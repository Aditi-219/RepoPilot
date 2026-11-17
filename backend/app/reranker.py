"""Reranker module for RepoPilot Phase 3 RAG pipeline.

Orders retrieved candidates for precision. Supports two modes:

- **Heuristic mode** (default): fast deduplication + RRF rank order preserved.
  Suitable for local CPU-only setups with no extra dependencies.

- **Neural mode** (opt-in via ENABLE_NEURAL_RERANKER=true): scores each
  (query, chunk) pair through a cross-encoder model
  (cross-encoder/ms-marco-MiniLM-L-6-v2) and re-sorts by the raw logit score.
  Requires sentence-transformers >= 2.2 and sufficient RAM/CPU (or GPU).
  Adds ~200–500 ms latency on CPU for a set of 20 candidates.

Set ENABLE_NEURAL_RERANKER=true in your .env to activate neural reranking.
"""

import logging
import os

from dotenv import load_dotenv

from app.search import SearchResult

# Ensure backend .env is loaded
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv()

logger = logging.getLogger(__name__)

# Read once at module load; avoids re-reading the env var on every call.
_NEURAL_RERANKER_ENABLED: bool = os.getenv("ENABLE_NEURAL_RERANKER", "false").lower() == "true"

# Model name for the cross-encoder. This is a small, fast model that works well
# for code retrieval tasks (trained on MS MARCO passage ranking).
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Lazy-loaded cross-encoder — only initialised when neural mode is first used.
_cross_encoder = None


class _SafetensorsCrossEncoder:
    """Offline cross-encoder wrapper using cached safetensors weights."""

    def __init__(self, model_dir: str):
        import safetensors.torch
        from transformers import AutoConfig, AutoTokenizer, BertForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        config = AutoConfig.from_pretrained(os.path.join(model_dir, "config.json"))
        weights = safetensors.torch.load_file(os.path.join(model_dir, "model.safetensors"))
        self.model = BertForSequenceClassification(config)
        self.model.load_state_dict(weights, strict=False)
        self.model.eval()

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        import torch

        if not pairs:
            return []
        queries = [p[0] for p in pairs]
        texts = [p[1] for p in pairs]
        inputs = self.tokenizer(
            queries,
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1)
            if logits.ndim == 0:
                return [float(logits.item())]
            return logits.tolist()


def _get_cross_encoder():
    """Lazily load the cross-encoder model on first use.

    Returns:
        A CrossEncoder instance or _SafetensorsCrossEncoder fallback, or None on error.
    """
    global _cross_encoder  # noqa: PLW0603
    if _cross_encoder is not None:
        return _cross_encoder

    # 1. First, check local HuggingFace cache snapshot (avoids network timeouts & auth blocks)
    try:
        cache_base = os.path.expanduser(
            "~/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots"
        )
        if os.path.isdir(cache_base):
            for snap in os.listdir(cache_base):
                snap_dir = os.path.join(cache_base, snap)
                if os.path.isfile(os.path.join(snap_dir, "model.safetensors")):
                    _cross_encoder = _SafetensorsCrossEncoder(snap_dir)
                    logger.info("Neural cross-encoder loaded from local snapshot: %s", snap_dir)
                    return _cross_encoder
    except Exception as exc:
        logger.debug("Local safetensors cross-encoder load skipped: %s", exc)

    # 2. Fall back to standard sentence_transformers CrossEncoder
    try:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        logger.info("Loading neural cross-encoder: %s", _CROSS_ENCODER_MODEL)
        _cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL)
        logger.info("Neural cross-encoder loaded successfully via sentence-transformers.")
        return _cross_encoder
    except Exception as exc:
        logger.warning("Standard CrossEncoder init failed: %s", exc)

    logger.warning("Could not load cross-encoder model '%s' (falling back to heuristic).", _CROSS_ENCODER_MODEL)
    _cross_encoder = None
    return _cross_encoder


def _neural_rerank(
    query: str,
    candidates: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """Score candidates with a cross-encoder and return the top-k by score.

    Args:
        query:      The user's search query.
        candidates: Pre-filtered list of SearchResult objects to score.
        top_k:      Maximum number of results to return.

    Returns:
        Candidates sorted by descending cross-encoder logit score, truncated to top_k.
        Falls back to heuristic ordering if the model is unavailable.
    """
    encoder = _get_cross_encoder()
    if encoder is None:
        logger.warning("Cross-encoder unavailable; falling back to heuristic reranking.")
        return candidates[:top_k]

    pairs = [(query, r.content) for r in candidates]
    try:
        scores = encoder.predict(pairs)
    except Exception as exc:
        logger.warning("Cross-encoder prediction failed (falling back): %s", exc)
        return candidates[:top_k]

    for result, score in zip(candidates, scores, strict=False):
        result.score = float(score)

    ranked = sorted(candidates, key=lambda r: r.score, reverse=True)
    logger.info(
        "Neural reranker scored %d candidates; returning top %d.",
        len(ranked),
        top_k,
    )
    return ranked[:top_k]


def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int = 8,
) -> list[SearchResult]:
    """Rerank retrieval candidates and return the top-k results.

    Uses neural cross-encoder scoring when ENABLE_NEURAL_RERANKER=true;
    otherwise applies fast heuristic deduplication preserving RRF order.

    Args:
        query:   The user's search query (used only by the neural path).
        results: RRF-ranked list of SearchResult objects from hybrid_search().
        top_k:   Maximum number of results to return.

    Returns:
        Up to top_k SearchResult objects ordered by relevance.
    """
    if not results:
        return []

    # Deduplicate by (file_path, start_line, end_line) to remove exact duplicates
    # that can appear when semantic and keyword searches retrieve the same chunk.
    seen: set[tuple[str, int, int]] = set()
    unique_results: list[SearchResult] = []
    for r in results:
        key = (r.file_path, r.start_line, r.end_line)
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    if _NEURAL_RERANKER_ENABLED:
        return _neural_rerank(query, unique_results, top_k)

    # Hybrid RRF rank order is already high quality; just truncate.
    return unique_results[:top_k]
