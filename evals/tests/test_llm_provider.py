"""Unit tests for RepoPilot LLM Provider subsystem.

Tests cover:
1. Chain selection (chain order, picking first healthy, forced provider override)
2. 503 vs 429 handling (separate backoffs, Retry-After header, daily quota exhaustion)
3. Refusing mid-run judge switching (never falling back per call, halting run)
4. Generator fallback policy (--allow-generator-fallback vs failing the row)
5. Ollama JSON parse failure handling (tracking parse_failure, retry once, never turning to 0)
6. Comparison guard (refusing comparison when judge provider or model differs)
7. Security & API key redaction (no keys printed or logged)
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))
sys.path.insert(0, os.path.join(REPO_ROOT, "evals", "scripts"))

from app.llm_provider import (
    LLMResponse,
    OverloadedError,
    ProviderError,
    ProviderJudgeLLM,
    QuotaExhaustedError,
    RateLimitError,
    acomplete,
    assert_matching_judges,
    check_provider_health,
    complete,
    get_aux_chain,
    get_generator_chain,
    get_judge_chain,
    get_model_for_provider,
    JudgeEvaluationError,
    JudgeMismatchError,
    JudgeRunHaltedError,
    sanitize_url,
    select_healthy_judge_provider,
    _call_gemini_raw,
    _call_openai_compatible_raw,
    _extract_retry_after,
)


# ---------------------------------------------------------------------------
# 1. Chain Selection Tests
# ---------------------------------------------------------------------------

def test_chain_parsing_defaults_and_env(monkeypatch):
    monkeypatch.delenv("JUDGE_CHAIN", raising=False)
    monkeypatch.delenv("GENERATOR_CHAIN", raising=False)
    monkeypatch.delenv("AUX_CHAIN", raising=False)

    assert get_judge_chain() == ["gemini", "groq", "ollama"]
    assert get_generator_chain() == ["gemini", "groq", "ollama"]
    assert get_aux_chain() == ["gemini", "groq", "ollama"]

    monkeypatch.setenv("JUDGE_CHAIN", "groq, ollama, gemini")
    assert get_judge_chain() == ["groq", "ollama", "gemini"]


@pytest.mark.asyncio
async def test_chain_selection_picks_first_healthy(monkeypatch):
    monkeypatch.setenv("JUDGE_CHAIN", "gemini,groq,ollama")

    # Mock health checks: gemini fails, groq succeeds
    async def mock_health(prov: str, timeout: float = 4.0) -> bool:
        return prov.lower() == "groq"

    with patch("app.llm_provider.check_provider_health", side_effect=mock_health):
        chosen_prov, chosen_model = select_healthy_judge_provider(chain=["gemini", "groq", "ollama"])
        assert chosen_prov == "groq"
        assert chosen_model == get_model_for_provider("groq")


def test_chain_selection_forced_provider():
    with patch("app.llm_provider.check_provider_health", new_callable=AsyncMock) as mock_health:
        mock_health.return_value = True
        chosen_prov, chosen_model = select_healthy_judge_provider(forced_provider="ollama")
        assert chosen_prov == "ollama"
        assert chosen_model == get_model_for_provider("ollama")


def test_chain_selection_forced_unhealthy_raises():
    with patch("app.llm_provider.check_provider_health", new_callable=AsyncMock) as mock_health:
        mock_health.return_value = False
        with pytest.raises(RuntimeError, match="failed health check"):
            select_healthy_judge_provider(forced_provider="groq")


def test_chain_selection_raises_when_all_fail():
    with patch("app.llm_provider.check_provider_health", new_callable=AsyncMock) as mock_health:
        mock_health.return_value = False
        with pytest.raises(RuntimeError, match="No healthy judge provider found"):
            select_healthy_judge_provider(chain=["gemini", "groq", "ollama"])


# ---------------------------------------------------------------------------
# 2. 503 vs 429 Handling Tests
# ---------------------------------------------------------------------------

def test_extract_retry_after_header():
    res = httpx.Response(status_code=429, headers={"retry-after": "14.5"})
    assert _extract_retry_after(res) == 14.5


def test_extract_retry_after_gemini_json_body():
    body = {
        "error": {
            "code": 429,
            "message": "Resource exhausted",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "18.5s"}],
        }
    }
    res = httpx.Response(status_code=429, json=body)
    assert _extract_retry_after(res) == 18.5


def test_extract_retry_after_huge_delay_raises_quota_exhausted():
    body = {
        "error": {
            "code": 429,
            "message": "Resource exhausted",
            "details": [{"retryDelay": "84148s"}],
        }
    }
    res = httpx.Response(status_code=429, json=body)
    with pytest.raises(QuotaExhaustedError, match="Daily quota exhausted"):
        _extract_retry_after(res)


@pytest.mark.asyncio
async def test_call_raw_distinguishes_429_and_503():
    # 429 on Gemini
    res_429 = httpx.Response(status_code=429, json={"error": {"message": "Rate limit"}})
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = res_429
        with pytest.raises(RateLimitError):
            await _call_gemini_raw("hi", "gemini-model", "fake-key", "https://api")

    # 503 on OpenAI-compatible (Groq/Ollama)
    res_503 = httpx.Response(status_code=503, json={"error": {"message": "Overloaded"}})
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = res_503
        with pytest.raises(OverloadedError):
            await _call_openai_compatible_raw("hi", "groq", "llama-model", "fake-key", "https://api")


@pytest.mark.asyncio
async def test_exponential_backoff_exhaustion_raises():
    with patch("app.llm_provider._call_gemini_raw", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = RateLimitError("Gemini 429 rate limit (retry_after=None)")
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(ProviderError, match="failed after 3 attempts"):
                await acomplete(prompt="test", role="judge", provider="gemini", max_retries=3, base_backoff=0.01)
            assert mock_sleep.call_count == 2  # slept between attempt 1->2 and 2->3


# ---------------------------------------------------------------------------
# 3. Refusing Mid-Run Judge Switching (Rule 2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_judge_refuses_mid_run_switch_and_halts():
    judge = ProviderJudgeLLM(provider="gemini", model="gemini-flash-lite-latest")

    with patch("app.llm_provider.acomplete", new_callable=AsyncMock) as mock_complete:
        mock_complete.side_effect = RateLimitError("Gemini quota exhausted")
        with pytest.raises(JudgeRunHaltedError, match="Halting run to preserve consistent judge scoring"):
            await judge.a_generate("Judge this test case")


def test_judge_never_falls_back_per_call():
    with pytest.raises(ValueError, match="Judge completions must specify the designated provider"):
        complete(prompt="hi", role="judge", provider=None)


# ---------------------------------------------------------------------------
# 4. Generator Fallback Policy (Rule 3)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generator_fails_row_when_fallback_disallowed(monkeypatch):
    monkeypatch.setenv("GENERATOR_CHAIN", "gemini,groq,ollama")

    with patch("app.llm_provider._call_provider_with_backoff", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = ProviderError("Primary Gemini generation failed")

        # When allow_fallback=False, should NOT try Groq; fail immediately
        with pytest.raises(ProviderError, match="Primary Gemini generation failed"):
            await acomplete(prompt="Answer this", role="generator", allow_fallback=False)
        assert mock_call.call_count == 1


@pytest.mark.asyncio
async def test_generator_falls_back_when_allowed(monkeypatch):
    monkeypatch.setenv("GENERATOR_CHAIN", "gemini,groq,ollama")

    # Call 1 (gemini) fails, Call 2 (groq) succeeds
    async def mock_call(provider, *args, **kwargs):
        if provider == "gemini":
            raise ProviderError("Gemini unavailable")
        return LLMResponse(text="Grounded answer from Groq", provider="groq", model="llama-3.3-70b-versatile")

    with patch("app.llm_provider._call_provider_with_backoff", side_effect=mock_call):
        res = await acomplete(prompt="Answer this", role="generator", allow_fallback=True)
        assert res.text == "Grounded answer from Groq"
        assert res.provider == "groq"
        assert res.model == "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# 5. Ollama JSON Parse Failure Handling (Rule 6)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ollama_parse_failure_retry_and_error():
    judge = ProviderJudgeLLM(provider="ollama", model="qwen2.5-coder:1.5b")
    judge.set_current_metric("contextual_relevancy")

    # Return invalid JSON on both attempts
    invalid_response = LLMResponse(text="Here is my verdict: it looks mostly good!", provider="ollama", model="qwen2.5-coder:1.5b")

    with patch("app.llm_provider.acomplete", new_callable=AsyncMock) as mock_complete:
        mock_complete.return_value = invalid_response
        with pytest.raises(JudgeEvaluationError, match="Evaluation error for metric 'contextual_relevancy'"):
            await judge.a_generate("Provide JSON score and reason")

        # Verify parse failure was counted
        assert judge.get_parse_failures()["contextual_relevancy"] == 1
        # Verify it retried once (total 2 attempts)
        assert mock_complete.call_count == 2


@pytest.mark.asyncio
async def test_ollama_parse_failure_recovers_on_retry():
    judge = ProviderJudgeLLM(provider="ollama", model="qwen2.5-coder:1.5b")
    judge.set_current_metric("faithfulness")

    res_invalid = LLMResponse(text="Sure! Score is 1.0", provider="ollama", model="qwen2.5-coder:1.5b")
    res_valid = LLMResponse(text='{"score": 1.0, "reason": "Accurate"}', provider="ollama", model="qwen2.5-coder:1.5b")

    with patch("app.llm_provider.acomplete", new_callable=AsyncMock) as mock_complete:
        mock_complete.side_effect = [res_invalid, res_valid]
        out = await judge.a_generate("Provide JSON score and reason")
        assert json.loads(out)["score"] == 1.0
        # Recorded the 1 parse failure from first attempt
        assert judge.get_parse_failures()["faithfulness"] == 1
        assert mock_complete.call_count == 2


# ---------------------------------------------------------------------------
# 6. Comparison Guard Tests (Rule 8)
# ---------------------------------------------------------------------------

def test_comparison_guard_matching_judges():
    run_a = {"judge_provider": "gemini", "judge_model": "gemini-flash-lite-latest"}
    run_b = {"judge_provider": "gemini", "judge_model": "gemini-flash-lite-latest"}
    # Should execute cleanly with no exception
    assert_matching_judges(run_a, run_b)


def test_comparison_guard_provider_mismatch_raises():
    run_a = {"judge_provider": "gemini", "judge_model": "gemini-flash-lite-latest"}
    run_b = {"judge_provider": "ollama", "judge_model": "gemini-flash-lite-latest"}

    with pytest.raises(JudgeMismatchError, match="judge mismatch detected.*used judge 'gemini/.*used judge 'ollama/"):
        assert_matching_judges(run_a, run_b, label_a="Baseline", label_b="Candidate")


def test_comparison_guard_model_mismatch_raises():
    run_a = {"judge_provider": "gemini", "judge_model": "gemini-flash-lite-latest"}
    run_b = {"judge_provider": "gemini", "judge_model": "gemini-3-flash-preview"}

    with pytest.raises(JudgeMismatchError, match="judge mismatch detected.*gemini-flash-lite-latest.*gemini-3-flash-preview"):
        assert_matching_judges(run_a, run_b)


# ---------------------------------------------------------------------------
# 7. Privacy & Security Tests (Rule 10)
# ---------------------------------------------------------------------------

def test_sanitize_url_redacts_api_key():
    raw_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent?key=AQ.Secret12345&other=1"
    cleaned = sanitize_url(raw_url)
    assert "AQ.Secret12345" not in cleaned
    assert "key=[REDACTED]" in cleaned


def test_llm_response_repr_does_not_leak_keys():
    resp = LLMResponse(text="output", provider="groq", model="llama", raw={"secret": "abc"})
    assert str(resp) == "output"
