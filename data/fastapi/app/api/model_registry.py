from uuid import UUID

from app.api.error_responses import raise_api_error
from app.services.model_registry_service import ModelRegistryService
from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/models", tags=["Model Registry"])


@router.get("/health")
def model_registry_health() -> dict:
    """Vérifie que le registre PostgreSQL est accessible."""

    try:
        models = ModelRegistryService.list_models()
        return {
            "status": "success",
            "service": "model_registry",
            "models_count": len(models),
        }
    except Exception as error:
        raise_api_error(
            operation="models.health",
            error=error,
            detail="Le registre des modèles est indisponible.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@router.get("")
def list_models() -> dict:
    """Liste les modèles enregistrés."""

    try:
        models = ModelRegistryService.list_models()
        return {"status": "success", "count": len(models), "models": models}
    except Exception as error:
        raise_api_error(
            operation="models.list",
            error=error,
            detail="Impossible de récupérer les modèles.",
        )


@router.get("/runs")
def list_training_runs() -> dict:
    """Liste les exécutions d'entraînement enregistrées."""

    try:
        runs = ModelRegistryService.list_training_runs()
        return {"status": "success", "count": len(runs), "runs": runs}
    except Exception as error:
        raise_api_error(
            operation="models.list_runs",
            error=error,
            detail="Impossible de récupérer les entraînements.",
        )


@router.get("/{model_id}")
def get_model(model_id: UUID) -> dict:
    """Retourne un modèle du registre."""

    try:
        model = ModelRegistryService.get_model(model_id)
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Modèle introuvable.",
            )
        return {"status": "success", "model": model}
    except HTTPException:
        raise
    except Exception as error:
        raise_api_error(
            operation="models.get",
            error=error,
            detail="Impossible de récupérer le modèle.",
        )


@router.post("/{model_id}/activate")
def activate_model(model_id: UUID) -> dict:
    """Active exclusivement un modèle pour sa tâche."""

    try:
        model = ModelRegistryService.activate_model(model_id)
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Modèle introuvable.",
            )
        return {"status": "success", "message": "Modèle activé.", "model": model}
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except HTTPException:
        raise
    except Exception as error:
        raise_api_error(
            operation="models.activate",
            error=error,
            detail="Impossible d'activer le modèle.",
        )


@router.post("/{model_id}/archive")
def archive_model(model_id: UUID) -> dict:
    """Archive un modèle du registre."""

    try:
        model = ModelRegistryService.archive_model(model_id)
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Modèle introuvable.",
            )
        return {"status": "success", "message": "Modèle archivé.", "model": model}
    except HTTPException:
        raise
    except Exception as error:
        raise_api_error(
            operation="models.archive",
            error=error,
            detail="Impossible d'archiver le modèle.",
        )
