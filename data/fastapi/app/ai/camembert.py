from __future__ import annotations

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "almanach/camembert-base"

_tokenizer = None
_model = None


def load_camembert():
    """Charge CamemBERT une seule fois en mémoire."""
    global _tokenizer, _model

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if _model is None:
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()

    return _tokenizer, _model


def get_embedding(text: str) -> list[float]:
    """Transforme un texte en vecteur normalisé."""
    if not text or not text.strip():
        raise ValueError("Le texte ne peut pas être vide.")

    tokenizer, model = load_camembert()

    inputs = tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"].unsqueeze(-1)

    masked_embeddings = token_embeddings * attention_mask
    summed_embeddings = masked_embeddings.sum(dim=1)
    token_counts = attention_mask.sum(dim=1).clamp(min=1)

    embedding = summed_embeddings / token_counts
    embedding = functional.normalize(embedding, p=2, dim=1)

    return embedding.squeeze(0).tolist()


def get_camembert_status() -> dict:
    return {
        "status": "loaded" if _model is not None else "not_loaded",
        "model": MODEL_NAME,
        "usage": "rag_embedding",
    }