from __future__ import annotations

from typing import Any

from app.api.error_responses import raise_api_error
from app.services.analysis_orchestrator import AnalysisOrchestrator
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/reports", tags=["Reports"])


class ReportGenerationRequest(BaseModel):
    """Résultats nécessaires à la génération d'un rapport documentaire."""

    project: dict[str, Any] = Field(default_factory=dict)
    deep_learning_result: dict[str, Any] = Field(default_factory=dict)
    rag_result: dict[str, Any] = Field(default_factory=dict)


@router.get("/health")
def reports_health() -> dict[str, Any]:
    """Retourne les capacités disponibles du générateur de rapports."""

    return {
        "status": "ready",
        "service": "AnalysisOrchestrator",
        "report_builder": "PrudenciaReportBuilder",
        "schema_version": "1.0",
        "available_reports": ["global", "documentary"],
    }


@router.post("/generate")
def generate_report(
    request: ReportGenerationRequest,
) -> dict[str, Any]:
    """Construit le rapport à partir des résultats JuriBERT et RAG."""

    try:
        return AnalysisOrchestrator().analyse(
            project=request.project,
            deep_learning_result=request.deep_learning_result,
            rag_result=request.rag_result,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except Exception as error:
        raise_api_error(
            operation="reports.generate",
            error=error,
            detail="Impossible de générer le rapport PRUDENCIA.",
        )
