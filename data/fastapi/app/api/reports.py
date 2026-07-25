from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.analysis_orchestrator import (
    AnalysisOrchestrator,
)


router = APIRouter(
    prefix="/reports",
    tags=["Reports"],
)


class ReportGenerationRequest(BaseModel):
    """
    Données nécessaires à la génération du rapport PRUDENCIA.

    Les résultats Machine Learning, Deep Learning et RAG
    restent indépendants. L'orchestrateur les transmet au
    Report Builder afin de produire un rapport JSON unique.
    """

    project: dict[str, Any] = Field(
        default_factory=dict,
    )

    machine_learning_result: dict[str, Any] = Field(
        default_factory=dict,
    )

    deep_learning_result: dict[str, Any] = Field(
        default_factory=dict,
    )

    rag_result: dict[str, Any] = Field(
        default_factory=dict,
    )


@router.get("/health")
def reports_health() -> dict[str, Any]:
    """
    Vérifie que le service de génération de rapports
    est disponible.
    """

    return {
        "status": "ready",
        "service": "AnalysisOrchestrator",
        "report_builder": "PrudenciaReportBuilder",
        "schema_version": "1.0",
    }


@router.post("/generate")
def generate_report(
    request: ReportGenerationRequest,
) -> dict[str, Any]:
    """
    Génère le rapport JSON final de PRUDENCIA.
    """

    try:
        orchestrator = AnalysisOrchestrator()

        report = orchestrator.analyse(
            project=request.project,
            machine_learning_result=(
                request.machine_learning_result
            ),
            deep_learning_result=(
                request.deep_learning_result
            ),
            rag_result=request.rag_result,
        )

        return report

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur pendant la génération "
                f"du rapport PRUDENCIA : {error}"
            ),
        ) from error