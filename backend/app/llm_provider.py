"""RepoPilot LLM Provider Subsystem.

Provides a unified, resilient multi-provider interface (Gemini, Groq, Ollama)
for Judge, Generator, and Auxiliary LLM calls with role-based fallback policies,
strict provenance tracking, separate 429/503 handling, and DeepEval integration.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
import json
import logging
import os
import random
import re
import sys
import threading
import time
from typing import Any, Callable

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration & Chain Resolution
# ---------------------------------------------------------------------------

DEFAULT_CHAIN = ["gemini", "groq", "ollama"]


def _parse_chain(env_var_name: str, default: list[str]) -> list[str]:
    raw = os.getenv(env_var_name, "")
    if not raw.strip():
        return list(default)
    chain = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return chain if chain else list(default)


def get_judge_chain() -> list[str]:
    return _parse_chain("JUDGE_CHAIN", DEFAULT_CHAIN)


def get_generator_chain() -> list[str]:
    return _parse_chain("GENERATOR_CHAIN", DEFAULT_CHAIN)


def get_aux_chain() -> list[str]:
    return _parse_chain("AUX_CHAIN", DEFAULT_CHAIN)


def get_model_for_provider(provider: str) -> str:
    prov = provider.lower()
    if prov == "gemini":
        return os.getenv("GEMINI_MODEL") or os.getenv("EVAL_JUDGE_MODEL") or "gemini-3.5-flash-lite"
    if prov == "groq":
        return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if prov == "ollama":
        return os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b")
    raise ValueError(f"Unknown LLM provider: '{provider}'")


def get_base_url_for_provider(provider: str) -> str:
    prov = provider.lower()
    if prov == "gemini":
        return "https://generativelanguage.googleapis.com/v1beta"
    if prov == "groq":
        return os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    if prov == "ollama":
        url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        if not url.endswith("/v1"):
            url = f"{url}/v1"
        return url
    raise ValueError(f"Unknown LLM provider: '{provider}'")


def get_api_key_for_provider(provider: str) -> str:
    prov = provider.lower()
    if prov == "gemini":
        return os.getenv("GEMINI_API_KEY", "").strip()
    if prov == "groq":
        return os.getenv("GROQ_API_KEY", "").strip()
    if prov == "ollama":
        return "ollama"  # Ollama OpenAI endpoint accepts any string
    return ""


# ---------------------------------------------------------------------------
# Data Structures & Exceptions
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    raw: dict[str, Any] | None = None
    model_version: str | None = None

    def __str__(self) -> str:
        return self.text


class ProviderError(Exception):
    """Base exception for provider calls."""


class RateLimitError(ProviderError):
    """HTTP 429 rate limit exceeded."""


class QuotaExhaustedError(RateLimitError):
    """Daily or monthly API quota fully exhausted (unrecoverable today)."""


class OverloadedError(ProviderError):
    """HTTP 503 service unavailable / overloaded."""


class JudgeRunHaltedError(ProviderError):
    """Raised when the designated judge fails mid-run; switching is strictly refused."""


class JudgeMismatchError(ValueError):
    """Raised when comparing runs evaluated with different judges."""


class JudgeEvaluationError(ProviderError):
    """Raised when a judge evaluation cannot produce valid parseable metric data."""
    def __init__(self, metric: str, message: str = "") -> None:
        super().__init__(f"Evaluation error for metric '{metric}': {message}")
        self.metric = metric


# ---------------------------------------------------------------------------
# HTTP Error & Retry Handling (Rule 5)
# ---------------------------------------------------------------------------

def _extract_retry_after(response: httpx.Response | None) -> float | None:
    if response is None:
        return None

    # 1. Standard Retry-After header
    hdr = response.headers.get("retry-after")
    if hdr:
        try:
            val = float(hdr)
            return val
        except ValueError:
            pass

    # 2. Check JSON payload for Gemini style retryDelay: "19s" or "84148s"
    try:
        body = response.json()
        error_info = body.get("error", {})
        details = error_info.get("details", [])
        for d in details:
            if "retryDelay" in d:
                delay_str = str(d["retryDelay"]).rstrip("s")
                val = float(delay_str)
                if val > 120.0:
                    raise QuotaExhaustedError(f"Daily quota exhausted (retryDelay={val:.0f}s).")
                return val
    except (ValueError, KeyError, TypeError):
        pass

    return None


def sanitize_url(url: str) -> str:
    """Mask any API keys embedded in query strings for safe logging."""
    return re.sub(r"key=[^&]+", "key=[REDACTED]", url)


# ---------------------------------------------------------------------------
# Low-level Provider Callers (Gemini, Groq, Ollama)
# ---------------------------------------------------------------------------

async def _call_gemini_raw(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 30.0,
) -> tuple[str, dict[str, Any]]:
    if not api_key:
        raise ProviderError("GEMINI_API_KEY is not set.")

    url = f"{base_url}/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens),
        },
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        res = await client.post(url, json=payload)
        status = res.status_code

        if status == 200:
            data = res.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts).strip()
                return text, data
            return "", data

        retry_after = _extract_retry_after(res)

        if status == 429:
            raise RateLimitError(f"Gemini 429 rate limit (retry_after={retry_after})")
        if status == 503:
            raise OverloadedError(f"Gemini 503 service overloaded (retry_after={retry_after})")
        if status in (401, 403):
            raise ProviderError(f"Gemini authentication/permission failure: HTTP {status}")

        err_msg = ""
        try:
            err_msg = res.json().get("error", {}).get("message", res.text[:200])
        except Exception:
            err_msg = res.text[:200]
        raise ProviderError(f"Gemini HTTP {status}: {err_msg}")


async def _call_openai_compatible_raw(
    prompt: str,
    provider_name: str,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 60.0,
) -> tuple[str, dict[str, Any]]:
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        res = await client.post(url, json=payload, headers=headers)
        status = res.status_code

        if status == 200:
            data = res.json()
            choices = data.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "").strip()
                return text, data
            return "", data

        retry_after = _extract_retry_after(res)

        if status == 429:
            raise RateLimitError(f"{provider_name} 429 rate limit (retry_after={retry_after})")
        if status == 503:
            raise OverloadedError(f"{provider_name} 503 service overloaded (retry_after={retry_after})")
        if status in (401, 403):
            raise ProviderError(f"{provider_name} authentication/permission failure: HTTP {status}")

        err_msg = ""
        try:
            err_msg = res.json().get("error", {}).get("message", res.text[:200])
        except Exception:
            err_msg = res.text[:200]
        raise ProviderError(f"{provider_name} HTTP {status}: {err_msg}")


# ---------------------------------------------------------------------------
# Resilient Caller with Backoff & Jitter
# ---------------------------------------------------------------------------

async def _call_provider_with_backoff(
    provider: str,
    prompt: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_retries: int = 4,
    base_backoff: float = 2.0,
) -> LLMResponse:
    prov = provider.lower()
    actual_model = model or get_model_for_provider(prov)
    base_url = get_base_url_for_provider(prov)
    api_key = get_api_key_for_provider(prov)

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            if prov == "gemini":
                text, raw = await _call_gemini_raw(
                    prompt=prompt,
                    model=actual_model,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            elif prov in ("groq", "ollama"):
                text, raw = await _call_openai_compatible_raw(
                    prompt=prompt,
                    provider_name=prov,
                    model=actual_model,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                raise ProviderError(f"Unsupported provider: {provider}")

            model_ver = (raw or {}).get("modelVersion") or (raw or {}).get("model") if isinstance(raw, dict) else None
            return LLMResponse(text=text, provider=prov, model=actual_model, raw=raw, model_version=model_ver)

        except QuotaExhaustedError:
            raise

        except RateLimitError as exc:
            last_error = exc
            if attempt == max_retries:
                break
            # Rule 5: 429 exponential backoff with jitter
            retry_match = re.search(r"retry_after=([0-9.]+)", str(exc))
            hdr_val = float(retry_match.group(1)) if (retry_match and retry_match.group(1) != "None") else None
            delay = hdr_val if hdr_val is not None else (base_backoff * (2 ** (attempt - 1))) + random.uniform(0.1, 1.0)
            logger.warning("Rate limit (429) on %s (attempt %d/%d) — backing off %.1fs", prov, attempt, max_retries, delay)
            await asyncio.sleep(delay)

        except OverloadedError as exc:
            last_error = exc
            if attempt == max_retries:
                break
            # Rule 5: 503 exponential backoff with jitter
            retry_match = re.search(r"retry_after=([0-9.]+)", str(exc))
            hdr_val = float(retry_match.group(1)) if (retry_match and retry_match.group(1) != "None") else None
            delay = hdr_val if hdr_val is not None else (base_backoff * (1.5 ** (attempt - 1))) + random.uniform(0.5, 2.0)
            logger.warning("Service overloaded (503) on %s (attempt %d/%d) — backing off %.1fs", prov, attempt, max_retries, delay)
            await asyncio.sleep(delay)

        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.TimeoutException) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            delay = (base_backoff * (1.5 ** (attempt - 1))) + random.uniform(0.2, 1.0)
            logger.warning("Connection error on %s: %s (attempt %d/%d) — retrying in %.1fs", prov, exc, attempt, max_retries, delay)
            await asyncio.sleep(delay)

        except ProviderError:
            raise

        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            delay = 1.0 + random.uniform(0.1, 0.5)
            await asyncio.sleep(delay)

    raise ProviderError(f"Provider '{prov}' failed after {max_retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Health Checks & Startup Selection (Rule 2)
# ---------------------------------------------------------------------------

async def check_provider_health(provider: str, timeout: float = 4.0) -> bool:
    """Non-destructive health-check probe for a provider."""
    prov = provider.lower()
    base_url = get_base_url_for_provider(prov)
    api_key = get_api_key_for_provider(prov)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if prov == "gemini":
                if not api_key:
                    return False
                model = get_model_for_provider(prov)
                url = f"{base_url}/models/{model}:generateContent?key={api_key}"
                res = await client.post(
                    url,
                    json={"contents": [{"parts": [{"text": "ping"}]}], "generationConfig": {"maxOutputTokens": 2}},
                )
                return res.status_code == 200

            if prov == "groq":
                if not api_key:
                    return False
                res = await client.get(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"})
                return res.status_code == 200

            if prov == "ollama":
                # Check models endpoint
                res = await client.get(f"{base_url}/models")
                return res.status_code == 200

    except Exception as exc:
        logger.debug("Health check failed for %s: %s", prov, exc)
        return False

    return False


def _run_coroutine_sync(coro: Any) -> Any:
    """Run an async coroutine synchronously, safely handling existing active event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def select_healthy_judge_provider(
    chain: list[str] | None = None,
    forced_provider: str | None = None,
) -> tuple[str, str]:
    """Pick judge provider once at start of run. Refuses mid-run switching (Rule 2)."""
    if forced_provider:
        prov = forced_provider.strip().lower()
        model = get_model_for_provider(prov)
        is_healthy = _run_coroutine_sync(check_provider_health(prov))
        if not is_healthy:
            raise RuntimeError(f"Forced judge provider '{prov}' failed health check or is unconfigured.")
        logger.info("Using forced judge provider: %s (model: %s)", prov, model)
        return prov, model

    candidate_chain = chain or get_judge_chain()
    for prov in candidate_chain:
        prov_clean = prov.strip().lower()
        if _run_coroutine_sync(check_provider_health(prov_clean)):
            model = get_model_for_provider(prov_clean)
            logger.info("Selected first healthy judge in chain: %s (model: %s)", prov_clean, model)
            return prov_clean, model

    raise RuntimeError(f"No healthy judge provider found in JUDGE_CHAIN: {candidate_chain}")


# ---------------------------------------------------------------------------
# Common Interface: complete & acomplete (Rules 3, 4)
# ---------------------------------------------------------------------------

async def acomplete(
    prompt: str,
    role: str = "aux",
    provider: str | None = None,
    model: str | None = None,
    allow_fallback: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> LLMResponse:
    """Execute LLM completion according to role-based fallback policies."""
    role_norm = role.lower()

    max_retries = kwargs.get("max_retries", 4)
    base_backoff = kwargs.get("base_backoff", 2.0)

    if role_norm == "judge":
        # Rule 2: Judge NEVER falls back per call.
        if not provider:
            raise ValueError("Judge completions must specify the designated provider.")
        return await _call_provider_with_backoff(
            provider=provider,
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            base_backoff=base_backoff,
        )

    if role_norm == "generator":
        # Rule 3: Per-call fallback allowed only when allow_fallback is True
        chain = [provider] if provider else get_generator_chain()
        if not allow_fallback:
            target_prov = chain[0]
            target_model = model or get_model_for_provider(target_prov)
            return await _call_provider_with_backoff(
                provider=target_prov,
                prompt=prompt,
                model=target_model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                base_backoff=base_backoff,
            )

        # Fallback allowed across chain
        last_exc: Exception | None = None
        for prov in chain:
            try:
                prov_model = model if (provider and prov == provider) else get_model_for_provider(prov)
                res = await _call_provider_with_backoff(
                    provider=prov,
                    prompt=prompt,
                    model=prov_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    base_backoff=base_backoff,
                )
                if prov != chain[0]:
                    logger.info("Generator fell back to provider=%s model=%s", res.provider, res.model)
                return res
            except Exception as exc:
                last_exc = exc
                logger.warning("Generator provider '%s' failed, trying next provider in chain: %s", prov, exc)
                continue
        raise ProviderError(f"All generator providers in chain failed: {last_exc}")

    # Rule 4: AUX calls allow fallback and log provider used
    chain = [provider] if provider else get_aux_chain()
    last_exc: Exception | None = None
    for prov in chain:
        try:
            prov_model = model if (provider and prov == provider) else get_model_for_provider(prov)
            res = await _call_provider_with_backoff(
                provider=prov,
                prompt=prompt,
                model=prov_model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                base_backoff=base_backoff,
            )
            logger.info("AUX LLM call successful using provider=%s model=%s", res.provider, res.model)
            return res
        except Exception as exc:
            last_exc = exc
            continue
    raise ProviderError(f"All aux providers in chain failed: {last_exc}")


def complete(
    prompt: str,
    role: str = "aux",
    provider: str | None = None,
    model: str | None = None,
    allow_fallback: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> LLMResponse:
    """Synchronous wrapper for acomplete."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run,
                acomplete(
                    prompt=prompt,
                    role=role,
                    provider=provider,
                    model=model,
                    allow_fallback=allow_fallback,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ),
            ).result()

    return loop.run_until_complete(
        acomplete(
            prompt=prompt,
            role=role,
            provider=provider,
            model=model,
            allow_fallback=allow_fallback,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    )


# ---------------------------------------------------------------------------
# DeepEval Judge Wrapper (Rules 2, 6)
# ---------------------------------------------------------------------------

try:
    from deepeval.models import DeepEvalBaseLLM as _DeepEvalBaseLLM
except (ImportError, Exception):
    _DeepEvalBaseLLM = object  # type: ignore[assignment,misc]


class ProviderJudgeLLM(_DeepEvalBaseLLM):  # type: ignore[misc]
    """DeepEval-compatible judge LLM backed by the provider subsystem.

    Enforces:
    - Fixed provider and model for the ENTIRE run (refuses mid-run switches).
    - Ollama invalid JSON parse failure tracking & single retry.
    - Zero evaluation score turning protection (never turns failure into 0.0).
    """

    def __init__(
        self,
        provider: str,
        model: str | None = None,
        temperature: float = 0.0,
        cache_path: str | None = None,
        cache_prefix: str = "",
    ) -> None:
        self.provider = provider.lower()
        self.model = model or get_model_for_provider(self.provider)
        self.temperature = float(temperature)
        self.cache_path = cache_path
        self.cache_prefix = cache_prefix
        self.parse_failures: dict[str, int] = defaultdict(int)
        self.observed_model_versions: list[str] = []
        self.current_metric_name: str = "general"
        self._cache: dict[str, str] = {}
        self._cache_lock = threading.Lock()

        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

        # DeepEval compatibility attributes
        self._model_name = f"{self.provider}/{self.model}"

    def get_model_name(self) -> str:
        return self._model_name

    def load_model(self) -> Any:
        return self

    def set_current_metric(self, metric_name: str) -> None:
        self.current_metric_name = metric_name

    def get_parse_failures(self) -> dict[str, int]:
        return dict(self.parse_failures)

    def _get_cache_key(self, prompt: str) -> str:
        import hashlib
        raw_key = f"{self.cache_prefix}{self.provider}:{self.model}:{self.temperature}:{prompt}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    async def a_generate(self, prompt: str) -> str:
        cache_key = self._get_cache_key(prompt)
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        try:
            res = await acomplete(
                prompt=prompt,
                role="judge",
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                max_tokens=2048,
            )
            raw_text = res.text
            if res.model_version and res.model_version not in self.observed_model_versions:
                self.observed_model_versions.append(res.model_version)

            # Rule 6: Ollama JSON verification
            if self.provider == "ollama":
                is_valid = self._verify_json_structure(raw_text)
                if not is_valid:
                    self.parse_failures[self.current_metric_name] += 1
                    logger.warning(
                        "Ollama JSON parse failure for metric '%s'. Retrying once with schema reminder...",
                        self.current_metric_name,
                    )
                    prompt_reinforced = (
                        f"{prompt}\n\nIMPORTANT: Output ONLY a valid JSON object matching the requested schema. "
                        "Do not include markdown fences, comments, or preamble."
                    )
                    res_retry = await acomplete(
                        prompt=prompt_reinforced,
                        role="judge",
                        provider=self.provider,
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=2048,
                    )
                    raw_text = res_retry.text
                    if not self._verify_json_structure(raw_text):
                        # Never turn into 0 score: raise evaluation error
                        raise JudgeEvaluationError(
                            self.current_metric_name,
                            f"Ollama failed to generate valid JSON after retry. Raw: {raw_text[:120]}",
                        )

            with self._cache_lock:
                self._cache[cache_key] = raw_text
                if self.cache_path:
                    try:
                        with open(self.cache_path, "w", encoding="utf-8") as f:
                            json.dump(self._cache, f, indent=2)
                    except Exception:
                        pass

            return raw_text

        except JudgeEvaluationError:
            raise
        except (RateLimitError, OverloadedError, ProviderError) as exc:
            # Rule 2: If judge fails mid-run, halt rather than switching
            raise JudgeRunHaltedError(
                f"Designated judge '{self.provider}/{self.model}' failed mid-run ({exc}). "
                "Halting run to preserve consistent judge scoring."
            ) from exc

    def generate(self, prompt: str) -> str:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.a_generate(prompt)).result()
        return loop.run_until_complete(self.a_generate(prompt))

    def _verify_json_structure(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        clean = text.strip()
        # Strip common markdown fencing if present
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean)
        try:
            parsed = json.loads(clean.strip())
            return isinstance(parsed, (dict, list))
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Comparison Guard (Rule 8)
# ---------------------------------------------------------------------------

def assert_matching_judges(
    run_a: dict[str, Any],
    run_b: dict[str, Any],
    label_a: str = "Baseline",
    label_b: str = "Candidate",
) -> None:
    """Refuse to compare runs evaluated with different judge providers or models."""
    prov_a = run_a.get("judge_provider")
    model_a = run_a.get("judge_model")
    prov_b = run_b.get("judge_provider")
    model_b = run_b.get("judge_model")

    if not prov_a or not prov_b or not model_a or not model_b:
        # Fall back to legacy key check if present
        meta_a = run_a.get("judge", {})
        meta_b = run_b.get("judge", {})
        prov_a = prov_a or meta_a.get("provider")
        model_a = model_a or meta_a.get("model")
        prov_b = prov_b or meta_b.get("provider")
        model_b = model_b or meta_b.get("model")

    if prov_a != prov_b or model_a != model_b:
        raise JudgeMismatchError(
            f"Cannot compare {label_a} vs {label_b}: judge mismatch detected. "
            f"{label_a} used judge '{prov_a}/{model_a}' while {label_b} used judge '{prov_b}/{model_b}'. "
            "Pipeline comparison requires the exact same judge provider and model to ensure valid comparison."
        )
