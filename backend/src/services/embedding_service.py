from __future__ import annotations

import logging
import os
import threading

import numpy as np


logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        self.model_name = os.getenv(
            "EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"
        ).strip()
        self._model = None
        self._lock = threading.Lock()

    def embed_query(self, text: str) -> np.ndarray | None:
        normalized = text.strip()
        if not normalized:
            return None
        prompt = (
            "Represent this profile for retrieving relevant books: "
            f"{normalized}"
        )
        embeddings = self._encode([prompt])
        return embeddings[0] if embeddings is not None and len(embeddings) else None

    def embed_documents(self, texts: list[str]) -> np.ndarray | None:
        normalized = [text.strip() for text in texts]
        if not normalized:
            return None
        return self._encode(normalized)

    def cosine_similarity(self, left: np.ndarray | None, right: np.ndarray | None) -> float:
        if left is None or right is None:
            return 0.0
        return float(np.dot(left, right))

    def _encode(self, texts: list[str]) -> np.ndarray | None:
        model = self._get_model()
        if model is None:
            return None
        try:
            return model.encode(
                texts,
                batch_size=min(16, max(1, len(texts))),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.warning("Embedding encode failed: %s", exc)
            return None

    def _get_model(self):
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            try:
                import torch
                from sentence_transformers import SentenceTransformer

                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = SentenceTransformer(self.model_name, device=device)
            except Exception as exc:
                logger.warning("Embedding model load failed: %s", exc)
                self._model = None
            return self._model


embedding_service = EmbeddingService()
