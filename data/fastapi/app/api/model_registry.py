from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from app.services.model_registry_service import ModelRegistryService


router = APIRouter(
    prefix="/models",
    tags=["Model Registry"],
)


@router.get("/health")
def model_registry_health():
    try:
        models = ModelRegistryService.list_models()

        return {
            "status": "success",
            "service": "model_registry",
            "models_count": len(models),
        }

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


@router.get("")
def list_models():
    try:
        models = ModelRegistryService.list_models()

        return {
            "status": "success",
            "count": len(models),
            "models": models,
        }

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error


@router.get("/runs")
def list_training_runs():
    try:
        runs = ModelRegistryService.list_training_runs()

        return {
            "status": "success",
            "count": len(runs),
            "runs": runs,
        }

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error


@router.get("/{model_id}")
def get_model(model_id: UUID):
    try:
        model = ModelRegistryService.get_model(model_id)

        if model is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Modèle introuvable.",
            )

        return {
            "status": "success",
            "model": model,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error


@router.post("/{model_id}/activate")
def activate_model(model_id: UUID):
    try:
        model = ModelRegistryService.activate_model(model_id)

        if model is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Modèle introuvable.",
            )

        return {
            "status": "success",
            "message": "Modèle activé.",
            "model": model,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error


@router.post("/{model_id}/archive")
def archive_model(model_id: UUID):
    try:
        model = ModelRegistryService.archive_model(model_id)

        if model is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Modèle introuvable.",
            )

        return {
            "status": "success",
            "message": "Modèle archivé.",
            "model": model,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error