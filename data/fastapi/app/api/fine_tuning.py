from __future__ import annotations

import shutil
import time

from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.ai.fine_tuning.trainer import FineTuningTrainer
from app.config import AVAILABLE_MODELS
from app.services.training_history_service import (
    get_training_history,
    save_model_reset,
    save_training_execution,
)


router = APIRouter(
    prefix="/fine-tuning",
    tags=["Fine-Tuning"],
)

UPLOAD_DATASET_DIR = Path("/app/uploads/datasets")
UPLOAD_DATASET_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def _extract_metrics(
    training_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Extrait les métriques quelle que soit leur position.
    """

    metrics = training_result.get("metrics")

    if isinstance(metrics, dict):
        return metrics

    evaluation = training_result.get("evaluation")

    if isinstance(evaluation, dict):
        return evaluation

    return training_result


@router.get("/models")
def list_available_models() -> dict[str, Any]:
    """
    Retourne les modèles disponibles pour le Fine-Tuning.
    """

    return {
        "models": [
            {
                "id": model_id,
                "name": configuration["name"],
                "huggingface_id": configuration["hf_id"],
                "description": configuration.get(
                    "description",
                    "",
                ),
            }
            for model_id, configuration
            in AVAILABLE_MODELS.items()
            if configuration.get("hf_id")
        ]
    }


@router.get("/history")
def fine_tuning_history() -> dict[str, Any]:
    """
    Retourne l'historique des entraînements.
    """

    return {
        "history": get_training_history(),
    }


@router.get("/best-model")
def get_best_model() -> dict[str, Any]:
    """
    Retourne le meilleur Fine-Tuning selon le F1-score,
    puis l'accuracy.
    """

    history = get_training_history()

    successful_runs = [
        run
        for run in history
        if run.get("success")
        and run.get("execution_type") == "training"
        and run.get("model_type") in {
            "deep_learning",
            "fine_tuning",
        }
    ]

    if not successful_runs:
        return {
            "best_model": None,
            "message": "Aucun entraînement disponible.",
        }

    def score(run: dict[str, Any]) -> tuple[float, float]:
        output_data = run.get("output_data") or {}

        if not isinstance(output_data, dict):
            output_data = {}

        metrics = output_data.get("metrics", output_data)

        if not isinstance(metrics, dict):
            metrics = {}

        f1_score = (
            metrics.get("f1")
            or metrics.get("eval_f1")
            or 0
        )

        accuracy = (
            metrics.get("accuracy")
            or metrics.get("eval_accuracy")
            or 0
        )

        return (
            float(f1_score or 0),
            float(accuracy or 0),
        )

    best_run = max(
        successful_runs,
        key=score,
    )

    return {
        "best_model": best_run,
    }


@router.post("/train")
async def train_model(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    text_column: str = Form(...),
    label_column: str = Form(...),
    epochs: int = Form(3),
    batch_size: int = Form(8),
    learning_rate: float = Form(2e-5),
) -> dict[str, Any]:
    """
    Reçoit un CSV puis lance et journalise le Fine-Tuning.
    """

    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Modèle inconnu : {model_name}. "
                f"Modèles disponibles : "
                f"{list(AVAILABLE_MODELS.keys())}"
            ),
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Le fichier CSV doit avoir un nom.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Seuls les fichiers CSV sont acceptés.",
        )

    if text_column == label_column:
        raise HTTPException(
            status_code=400,
            detail=(
                "La colonne texte et la colonne label "
                "doivent être différentes."
            ),
        )

    safe_filename = Path(file.filename).name
    dataset_path = UPLOAD_DATASET_DIR / safe_filename

    trainer = FineTuningTrainer()
    started_at = time.perf_counter()

    preparation_dict: dict[str, Any] = {}

    try:
        with dataset_path.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        preparation = trainer.prepare_training(
            csv_path=str(dataset_path),
            text_column=text_column,
            label_column=label_column,
        )

        preparation_dict = preparation.to_dict()

        if not preparation.is_trainable:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "Le dataset ne peut pas être utilisé "
                        "pour l'entraînement."
                    ),
                    "preparation": preparation_dict,
                    "quality_report": (
                        trainer.get_quality_report()
                    ),
                },
            )

        result = trainer.train(
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        elapsed_ms = int(
            (time.perf_counter() - started_at) * 1000
        )

        metrics = _extract_metrics(result)

        model_version = str(
            result.get("model_version")
            or result.get("version")
            or result.get("saved_model_path")
            or result.get("model_path")
            or f"{model_name}-latest"
        )

        dataset_rows = int(
            preparation_dict.get("row_count")
            or preparation_dict.get("total_rows")
            or preparation_dict.get("rows")
            or 0
        )

        save_training_execution(
            model_type="deep_learning",
            model_name=model_name,
            model_version=model_version,
            task_name="sequence_classification",
            dataset_name=safe_filename,
            dataset_rows=dataset_rows,
            input_data={
                "text_column": text_column,
                "label_column": label_column,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
            output_data={
                "metrics": metrics,
                "training": result,
            },
            execution_time_ms=elapsed_ms,
            success=True,
        )

        return {
            "preparation": preparation_dict,
            "quality_report": trainer.get_quality_report(),
            "training": result,
            "execution_time_ms": elapsed_ms,
        }

    except HTTPException:
        raise

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Une erreur inattendue est survenue pendant "
                f"le Fine-Tuning : {exc}"
            ),
        ) from exc

    finally:
        await file.close()

        if dataset_path.exists():
            dataset_path.unlink()


@router.post("/reset/{model_name}")
def reset_fine_tuned_model(
    model_name: str,
) -> dict[str, Any]:
    """
    Réinitialise un modèle vers sa version pré-entraînée.
    """

    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Modèle inconnu : {model_name}",
        )

    try:
        trainer = FineTuningTrainer()

        result = trainer.reset_model(
            model_name=model_name,
        )

        save_model_reset(
            model_type="deep_learning",
            model_name=model_name,
            model_version="base",
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur pendant la réinitialisation "
                f"du modèle Fine-Tuning : {error}"
            ),
        ) from error