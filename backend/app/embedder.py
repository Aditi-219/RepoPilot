"""Embedding generation module for RepoPilot's indexing pipeline.

Uses sentence-transformers/all-MiniLM-L6-v2 via HuggingFace transformers
to generate 384-dimensional dense vector embeddings from code and text chunks.

Model characteristics:
- Model: sentence-transformers/all-MiniLM-L6-v2
- Dimensions: 384
- Max input tokens: 256
- Mean pooling + L2 normalization (cosine similarity ready for pgvector)
"""

import logging
import os
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer, BertModel

from app.chunker import Chunk

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model: Any = None
_tokenizer: Any = None


def _load_model_and_tokenizer():
    """Load BERT model and tokenizer with weights from HuggingFace cache."""
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    logger.info("Initializing embedding model '%s'...", MODEL_NAME)

    # 1. Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2. Check for cached safetensors file
    import safetensors.torch
    cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots")
    weights_path = None
    if os.path.isdir(cache_base):
        for snap in os.listdir(cache_base):
            candidate = os.path.join(cache_base, snap, "model.safetensors")
            if os.path.isfile(candidate):
                weights_path = candidate
                break

    config = AutoConfig.from_pretrained(MODEL_NAME)
    _model = BertModel(config)

    if weights_path and os.path.exists(weights_path):
        weights = safetensors.torch.load_file(weights_path)
        _model.load_state_dict(weights, strict=False)
    else:
        # Fallback to standard from_pretrained if not in expected cache path
        _model = BertModel.from_pretrained(MODEL_NAME)

    _model.eval()
    logger.info("Embedding model loaded successfully! Output dimension: %d", EMBEDDING_DIM)
    return _model, _tokenizer


def _mean_pooling(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    """Perform mean pooling on BERT token embeddings taking attention mask into account."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def embed_chunks(chunks: list[Chunk], batch_size: int = 64) -> list[list[float]]:
    """Generate 384-dim normalized embeddings for a list of Chunk objects."""
    if not chunks:
        return []

    model, tokenizer = _load_model_and_tokenizer()
    texts = [chunk.content for chunk in chunks]
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        with torch.no_grad():
            output = model(**encoded)
            sentence_embeddings = _mean_pooling(output, encoded["attention_mask"])
            normalized = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            all_embeddings.extend(normalized.tolist())

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Generate a single 384-dim normalized embedding for a search query."""
    model, tokenizer = _load_model_and_tokenizer()
    encoded = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    with torch.no_grad():
        output = model(**encoded)
        sentence_embeddings = _mean_pooling(output, encoded["attention_mask"])
        normalized = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

    return normalized[0].tolist()
