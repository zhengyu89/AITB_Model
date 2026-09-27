"""Sentence embedding service backing POST /api/v1/embeddings.

Loaded once per process and reused across requests. Callers (AI-Travel-Buddy's
RAG search) depend on these vectors matching what is already stored in their
Qdrant collection, so `encode` is always called with its library defaults: no
prefixes, no normalization, no other preprocessing.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings


class TextEncoderUnavailable(RuntimeError):
    """Raised when the text embedding model is not loaded.

    `get_text_encoder` is not memoized on failure (`lru_cache` does not cache a
    raised exception), so the next call retries the load. A transient failure
    at startup (for example, a network hiccup while downloading the model)
    self-heals on the next request instead of wedging the service.
    """


@lru_cache(maxsize=1)
def get_text_encoder() -> SentenceTransformer:
    settings = get_settings()
    try:
        return SentenceTransformer(settings.text_embedding_model_name)
    except Exception as exc:  # pragma: no cover - depends on network/model cache
        raise TextEncoderUnavailable(
            f"Failed to load text embedding model '{settings.text_embedding_model_name}': {exc}"
        ) from exc


def encode_texts(texts: list[str]) -> list[list[float]]:
    """Encode texts with the model's default `encode` arguments.

    Raises `TextEncoderUnavailable` if the model could not be loaded. This is a
    blocking, CPU-bound call: run it off the event loop (see the route).
    """
    encoder = get_text_encoder()
    vectors = encoder.encode(texts)
    return vectors.tolist()
