from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.services.analysis_history_service import AnalysisHistoryService
from app.services.training_history_service import (
    get_training_history,
)


router = APIRouter(
    prefix="/training",
    tags=["Training"],
)


@router.get("/history")
def training_history() -> dict[str, Any]:
    """
    Retourne l'historique des entraînements
    Fine-Tuning et Machine Learning.
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


@router.get("/analyses")
def analyses_history(
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    analysis_type: str | None = Query(
        default=None,
    ),
    status: str | None = Query(
        default=None,
    ),
    project_id: str | None = Query(
        default=None,
    ),
) -> dict[str, Any]:
    """
    Retourne l'historique des rapports PRUDENCIA.

    Les analyses peuvent provenir :

    - du questionnaire Machine Learning ;
    - du rapport documentaire ;
    - des futures analyses ajoutées au projet.
    """

    try:
        service = AnalysisHistoryService()

        analyses = service.list_analyses(
            limit=limit,
            offset=offset,
            analysis_type=analysis_type,
            status=status,
            project_id=project_id,
        )

        return {
            "success": True,
            "count": len(analyses),
            "analyses": analyses,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur pendant la récupération "
                f"de l'historique des analyses : {error}"
            ),
        ) from error


@router.get("/analyses/{analysis_id}")
def analysis_detail(
    analysis_id: str,
) -> dict[str, Any]:
    """
    Retourne le détail d'une analyse et ses exécutions de modèles.
    """

    try:
        service = AnalysisHistoryService()
        analysis = service.get_analysis(analysis_id)

        if analysis is None:
            raise HTTPException(
                status_code=404,
                detail="Analyse introuvable.",
            )

        return {
            "success": True,
            "analysis": analysis,
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur pendant la récupération "
                f"de l'analyse : {error}"
            ),
        ) from error