from __future__ import annotations

from threading import Lock

from app.ai.embedding_config import RAG_EMBEDDING_MODEL
from sentence_transformers import SentenceTransformer

MODEL_NAME = RAG_EMBEDDING_MODEL

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

    if not text or not text.strip():
        raise ValueError("Le texte ne peut pas être vide.")

    model = get_model()

    embedding = model.encode(
        text.strip(),
        normalize_embeddings=True,
    )

    return embedding.tolist()


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Génère les embeddings d'un lot en un seul appel au modèle."""
    if not texts:
        raise ValueError("Aucun texte à encoder.")
    normalized_texts = [text.strip() for text in texts]
    if any(not text for text in normalized_texts):
        raise ValueError("Les textes à encoder ne peuvent pas être vides.")

    embeddings = get_model().encode(
        normalized_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()
