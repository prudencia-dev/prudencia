from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.analysis_orchestrator import AnalysisOrchestrator


router = APIRouter(prefix="/reports", tags=["Reports"])


class ReportGenerationRequest(BaseModel):
    """Données nécessaires au rapport documentaire."""

    project: dict[str, Any] = Field(default_factory=dict)
    deep_learning_result: dict[str, Any] = Field(default_factory=dict)
    rag_result: dict[str, Any] = Field(default_factory=dict)


@router.get("/health")
def reports_health() -> dict[str, Any]:
    return {
        "status": "ready",
        "service": "AnalysisOrchestrator",
        "report_builder": "PrudenciaReportBuilder",
        "schema_version": "1.0",
        "available_reports": ["global", "documentary"],
    }


@router.post("/generate")
def generate_report(request: ReportGenerationRequest) -> dict[str, Any]:
    """Assemble l'analyse textuelle et les références juridiques."""

    try:
        orchestrator = AnalysisOrchestrator()
        return orchestrator.analyse(
            project=request.project,
            deep_learning_result=request.deep_learning_result,
            rag_result=request.rag_result,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la génération du rapport PRUDENCIA : {error}",
        ) from error
