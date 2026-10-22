"""LLM integration for RepoPilot Phase 3 RAG Q&A.

Supports both:
1. Ollama (100% local, free, unlimited, private, no API key needed).
2. Google Gemini / Generative Language API (cloud-based).

If an Ollama model is installed locally (e.g. qwen2.5-coder, llama3.2),
RepoPilot automatically uses it without hitting any daily rate limits.
"""

import logging
import os
import httpx

from app.search import SearchResult

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PREFERRED_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")

CANDIDATE_GEMINI_MODELS = [
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemma-4-26b-a4b-it",
]

_SYSTEM_INSTRUCTION = """You are RepoPilot, an expert AI coding assistant.
You answer questions about codebases by reasoning over retrieved code snippets.

Rules:
- Base your answer ONLY on the provided code snippets. Do not use outside knowledge.
- If the snippets do not contain enough information to answer, say so explicitly.
- Only cite specific files and line numbers using this format: `path/to/file.py:L14-L58` if necessary.
- Be concise, precise, and developer-friendly.
- Use markdown: backticks for inline code, fenced blocks for multi-line code.
"""


def _format_snippet_context(snippets: list[SearchResult]) -> str:
    parts: list[str] = []
    for i, snippet in enumerate(snippets, start=1):
        header = f"[{i}] `{snippet.file_path}` lines {snippet.start_line}–{snippet.end_line}"
        if snippet.language:
            header += f" ({snippet.language})"
        code_block = f"```{snippet.language or ''}\n{snippet.content}\n```"
        parts.append(f"{header}\n{code_block}")
    return "\n\n---\n\n".join(parts)


def _build_prompt(question: str, snippets: list[SearchResult]) -> str:
    context = _format_snippet_context(snippets)
    return (
        f"{_SYSTEM_INSTRUCTION}\n\n"
        "The following code snippets were retrieved from the repository:\n\n"
        f"{context}\n\n"
        "---\n\n"
        f"Question: {question}\n\n"
        "Answer (grounded in the code snippets above, with file citations):"
    )


async def _try_ollama(prompt: str) -> tuple[str, str] | None:
    """Check if Ollama is running and has any models available to generate an answer."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            tags_res = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if tags_res.status_code != 200:
                return None
            models_info = tags_res.json().get("models", [])
            if not models_info:
                return None

            # Pick preferred model if configured and present, else pick first installed
            installed_names = [m.get("name", "") for m in models_info]
            chosen_model = None
            if PREFERRED_OLLAMA_MODEL and PREFERRED_OLLAMA_MODEL in installed_names:
                chosen_model = PREFERRED_OLLAMA_MODEL
            else:
                chosen_model = installed_names[0]

        logger.info("Using local Ollama model: %s", chosen_model)
        # Call generate with a generous timeout for local CPU inference
        async with httpx.AsyncClient(timeout=120.0) as client:
            gen_res = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": chosen_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 1024,
                    },
                },
            )
            if gen_res.status_code == 200:
                answer = gen_res.json().get("response", "").strip()
                return answer, f"ollama/{chosen_model}"

    except Exception as exc:
        logger.debug("Ollama generation skipped or failed (%s)", exc)
    return None


async def _try_gemini(prompt: str) -> tuple[str, str]:
    """Call Google Gemini / Generative Language API with fallback models."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "No local Ollama model found and GEMINI_API_KEY is not set in backend/.env."
        )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
        },
    }

    last_error = ""
    async with httpx.AsyncClient(timeout=15.0) as client:
        for model_name in CANDIDATE_GEMINI_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            try:
                res = await client.post(url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        raw_text = candidates[0]["content"]["parts"][0]["text"].strip()
                        return raw_text, model_name

                if res.status_code == 429:
                    last_error = "Free-tier daily quota exceeded on Gemini (20 requests/day). Switch to Ollama or create a new API key."
                    continue

                if res.status_code == 404:
                    continue

                err_msg = res.json().get("error", {}).get("message", res.text[:150])
                last_error = f"{model_name}: HTTP {res.status_code} ({err_msg})"

            except Exception as exc:
                last_error = str(exc)
                continue

    raise RuntimeError(f"{last_error} Retrieved code citations are displayed below.")


async def ask_with_llm(
    question: str,
    snippets: list[SearchResult],
) -> tuple[str, str]:
    """Generate a grounded answer using either local Ollama or Google Gemini."""
    prompt = _build_prompt(question, snippets)

    # 1. Try local Ollama first (no quota limits, 100% private and offline)
    ollama_result = await _try_ollama(prompt)
    if ollama_result is not None:
        return ollama_result

    # 2. Fall back to cloud Gemini API
    return await _try_gemini(prompt)
