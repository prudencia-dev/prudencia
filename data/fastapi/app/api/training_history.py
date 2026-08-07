from __future__ import annotations

from typing import Any

from app.api.error_responses import raise_api_error
from app.services.training_history_service import get_training_history
from fastapi import APIRouter

router = APIRouter(prefix="/training", tags=["Training"])


@router.get("/history")
def training_history() -> dict[str, Any]:
    """Retourne l'historique Deep Learning et Fine-Tuning."""

    try:
        return {"executions": get_training_history()}
    except Exception as error:
        raise_api_error(
            operation="training.history",
            error=error,
            detail="Impossible de récupérer l'historique des entraînements.",
        )
