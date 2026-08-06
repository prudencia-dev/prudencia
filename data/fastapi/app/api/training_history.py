from __future__ import annotations

from typing import Any

from app.services.training_history_service import (
    get_training_history,
)
from fastapi import APIRouter, HTTPException

router = APIRouter(
    prefix="/training",
    tags=["Training"],
)


@router.get("/history")
def training_history() -> dict[str, Any]:
    """
    Retourne l'historique des entraînements
    Deep Learning et Fine-Tuning.
    """

    try:
        return {
            "executions": get_training_history(),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur pendant la récupération "
                f"de l'historique : {error}"
            ),
        ) from error
