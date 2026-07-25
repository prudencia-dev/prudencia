from __future__ import annotations

from threading import Lock

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"

_model: SentenceTransformer | None = None
_model_lock = Lock()


def get_model() -> SentenceTransformer:
    """
    Charge le modèle BGE-M3 une seule fois.
    """

    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(MODEL_NAME)

    return _model


def get_embedding(text: str) -> list[float]:
    """
    Génère un embedding pour un texte.
    """

    model = get_model()

    embedding = model.encode(
        text,
        normalize_embeddings=True,
    )

    return embedding.tolist()