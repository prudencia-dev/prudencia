from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from app.ai.fine_tuning.trainer import FineTuningTrainer
from app.api.error_responses import raise_api_error
from app.config import AVAILABLE_MODELS
from app.services.training_history_service import (
    get_training_history,
    save_training_execution,
)
from app.services.upload_security import (
    MAX_CSV_UPLOAD_BYTES,
    UploadTooLargeError,
    build_stored_filename,
    confined_path,
    copy_limited_upload,
    sanitized_filename,
)
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel

router = APIRouter(
    prefix="/fine-tuning",
    tags=["Fine-Tuning"],
)

UPLOAD_DATASET_DIR = Path("/app/uploads/datasets")
UPLOAD_DATASET_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_CODE_VERSION = os.getenv("PRUDENCIA_CODE_VERSION", "1.2.0-local")
SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    """Calcule l'empreinte stable du dataset réellement entraîné."""

    digest = hashlib.sha256()
    with path.open("rb") as dataset_file:
        for block in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _benchmark_signature(configuration: dict[str, Any]) -> str:
    """Identifie les runs comparables, indépendamment du modèle BERT."""

    serialized = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


def _source_code_sha256(root: Path = SOURCE_ROOT) -> str:
    """Empreinte le code Python réellement utilisé par l'API."""

    digest = hashlib.sha256()
    for source_file in sorted(root.rglob("*.py")):
        digest.update(str(source_file.relative_to(root)).encode("utf-8"))
        digest.update(source_file.read_bytes())
    return digest.hexdigest()


class PredictionRequest(BaseModel):
    """Corps JSON attendu par la route de prédiction."""

    model_name: str
    text: str


def _extract_metrics(
    training_result: dict[str, Any],
) -> dict[str, Any]:
    """Extrait les métriques quelle que soit leur position."""

    metrics = training_result.get("metrics")

    if isinstance(metrics, dict):
        return metrics

    evaluation = training_result.get("evaluation")

    if isinstance(evaluation, dict):
        return evaluation

    return training_result


def _validate_training_parameters(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    max_length: int,
    gradient_accumulation_steps: int,
    early_stopping_patience: int,
    weight_decay: float,
    warmup_ratio: float,
    metric_for_best_model: str,
) -> None:
    """
    Contrôle les hyperparamètres reçus depuis Streamlit.

    Ces garde-fous empêchent une valeur incohérente de lancer un entraînement
    extrêmement long ou de provoquer une erreur peu lisible côté PyTorch.
    """

    if not 1 <= epochs <= 50:
        raise HTTPException(
            status_code=400,
            detail="Le nombre d'epochs doit être compris entre 1 et 50.",
        )

    if batch_size not in {1, 2, 4, 8, 16, 32}:
        raise HTTPException(
            status_code=400,
            detail="La taille de batch est invalide.",
        )

    if not 0 < learning_rate <= 0.01:
        raise HTTPException(
            status_code=400,
            detail=(
                "Le learning rate doit être supérieur à 0 "
                "et inférieur ou égal à 0,01."
            ),
        )

    if seed < 0:
        raise HTTPException(
            status_code=400,
            detail="La graine aléatoire doit être positive ou nulle.",
        )

    if max_length not in {128, 256, 320, 384, 512}:
        raise HTTPException(
            status_code=400,
            detail="La longueur maximale doit être 128, 256, 320, 384 ou 512.",
        )

    if gradient_accumulation_steps not in {1, 2, 4, 8, 16}:
        raise HTTPException(
            status_code=400,
            detail="L'accumulation de gradients est invalide.",
        )

    if not 0 <= early_stopping_patience <= 20:
        raise HTTPException(
            status_code=400,
            detail="La patience de l'early stopping doit être comprise entre 0 et 20.",
        )

    if not 0 <= weight_decay <= 1:
        raise HTTPException(
            status_code=400,
            detail="Le weight decay doit être compris entre 0 et 1.",
        )

    if not 0 <= warmup_ratio <= 0.5:
        raise HTTPException(
            status_code=400,
            detail="Le warmup ratio doit être compris entre 0 et 0,5.",
        )

    allowed_metrics = {
        "macro_f1",
        "f1_weighted",
        "accuracy",
        "loss",
    }

    if metric_for_best_model not in allowed_metrics:
        raise HTTPException(
            status_code=400,
            detail=(
                "La métrique du meilleur modèle doit être l'une de : "
                + ", ".join(sorted(allowed_metrics))
            ),
        )


@router.get("/models")
def list_available_models() -> dict[str, Any]:
    """Retourne les modèles disponibles pour le Fine-Tuning."""

    return {
        "models": [
            {
                "id": model_id,
                "name": configuration["name"],
                "huggingface_id": configuration["hf_id"],
                "description": configuration.get("description", ""),
            }
            for model_id, configuration in AVAILABLE_MODELS.items()
            if configuration.get("hf_id")
        ]
    }


@router.get("/history")
def fine_tuning_history() -> dict[str, Any]:
    """Retourne l'historique des entraînements et réinitialisations."""

    return {
        "history": get_training_history(),
    }


@router.get("/best-model")
def get_best_model() -> dict[str, Any]:
    """
    Retourne le meilleur Fine-Tuning.

    Le macro-F1 est prioritaire, puis le F1 pondéré et enfin l'accuracy. Cette
    hiérarchie évite qu'une classe majoritaire masque une classe rare mal
    reconnue.
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

    def score(run: dict[str, Any]) -> tuple[float, float, float]:
        output_data = run.get("output_data") or {}
        metrics = output_data.get("metrics", {})

        if not isinstance(metrics, dict):
            metrics = {}

        return (
            float(metrics.get("macro_f1") or 0),
            float(
                metrics.get("f1_weighted")
                or metrics.get("f1")
                or 0
            ),
            float(metrics.get("accuracy") or 0),
        )

    return {
        "best_model": max(successful_runs, key=score),
    }


@router.post("/train")
async def train_model(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    text_column: str = Form(...),
    label_column: str = Form(...),
    epochs: int = Form(10),
    batch_size: int = Form(4),
    learning_rate: float = Form(2e-5),
    seed: int = Form(42),
    max_length: int = Form(256),
    gradient_accumulation_steps: int = Form(2),
    early_stopping_patience: int = Form(3),
    weight_decay: float = Form(0.01),
    warmup_ratio: float = Form(0.10),
    metric_for_best_model: str = Form("macro_f1"),
    use_class_weights: bool = Form(True),
    training_profile: str = Form("Personnalisé"),
) -> dict[str, Any]:
    """
    Reçoit un CSV, lance le Fine-Tuning puis journalise l'expérience.

    Tous les hyperparamètres reçus sont sauvegardés dans ``input_data`` de
    ``prudencia.model_executions``. L'historique peut donc reproduire exactement
    les conditions de chaque run.
    """

    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Modèle inconnu : {model_name}. "
                f"Modèles disponibles : {list(AVAILABLE_MODELS.keys())}"
            ),
        )

    try:
        safe_filename = sanitized_filename(file.filename, ".csv")
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    if text_column == label_column:
        raise HTTPException(
            status_code=400,
            detail=(
                "La colonne texte et la colonne label doivent être différentes."
            ),
        )

    _validate_training_parameters(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        max_length=max_length,
        gradient_accumulation_steps=(
            gradient_accumulation_steps
        ),
        early_stopping_patience=early_stopping_patience,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        metric_for_best_model=metric_for_best_model,
    )

    stored_filename = build_stored_filename(safe_filename, ".csv")
    dataset_path = confined_path(UPLOAD_DATASET_DIR, stored_filename)

    trainer = FineTuningTrainer()
    started_at = time.perf_counter()
    preparation_dict: dict[str, Any] = {}

    try:
        copy_limited_upload(
            file,
            dataset_path,
            max_bytes=MAX_CSV_UPLOAD_BYTES,
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
                        "Le dataset ne peut pas être utilisé pour l'entraînement."
                    ),
                    "preparation": preparation_dict,
                    "quality_report": trainer.get_quality_report(),
                },
            )

        result = trainer.train(
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            max_length=max_length,
            gradient_accumulation_steps=(
                gradient_accumulation_steps
            ),
            early_stopping_patience=early_stopping_patience,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            metric_for_best_model=metric_for_best_model,
            use_class_weights=use_class_weights,
        )

        elapsed_ms = int(
            (time.perf_counter() - started_at) * 1000
        )
        metrics = _extract_metrics(result)

        model_version = str(
            result.get("model_version")
            or result.get("version")
            or result.get("model_path")
            or f"{model_name}-latest"
        )

        dataset_rows = int(
            preparation_dict.get("total_examples")
            or preparation_dict.get("row_count")
            or preparation_dict.get("total_rows")
            or preparation_dict.get("rows")
            or 0
        )

        effective_batch_size = (
            batch_size * gradient_accumulation_steps
        )

        dataset_sha256 = _sha256_file(dataset_path)
        source_code_sha256 = _source_code_sha256()
        reproducibility_configuration = {
            "dataset_sha256": dataset_sha256,
            "source_code_sha256": source_code_sha256,
            "text_column": text_column,
            "label_column": label_column,
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "seed": seed,
            "max_length": max_length,
            "early_stopping_patience": early_stopping_patience,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "metric_for_best_model": metric_for_best_model,
            "use_class_weights": use_class_weights,
        }
        benchmark_signature = _benchmark_signature(
            reproducibility_configuration
        )

        input_data = {
            "training_profile": training_profile[:100],
            "model_huggingface_id": AVAILABLE_MODELS[model_name]["hf_id"],
            "source_code_version": SOURCE_CODE_VERSION,
            "source_code_sha256": source_code_sha256,
            "dataset_sha256": dataset_sha256,
            "dataset_size_bytes": dataset_path.stat().st_size,
            "benchmark_signature": benchmark_signature,
            "text_column": text_column,
            "label_column": label_column,
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": (
                gradient_accumulation_steps
            ),
            "effective_batch_size": effective_batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "max_length": max_length,
            "early_stopping_patience": early_stopping_patience,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "metric_for_best_model": metric_for_best_model,
            "use_class_weights": use_class_weights,
        }

        execution = save_training_execution(
            model_type="deep_learning",
            model_name=model_name,
            model_version=model_version,
            task_name="sequence_classification",
            dataset_name=safe_filename,
            dataset_rows=dataset_rows,
            input_data=input_data,
            output_data={
                "metrics": metrics,
                "training": result,
            },
            execution_time_ms=elapsed_ms,
            success=True,
        )

        return {
            "traceability": execution
            | {
                "dataset_sha256": dataset_sha256,
                "benchmark_signature": benchmark_signature,
                "source_code_version": SOURCE_CODE_VERSION,
                "source_code_sha256": source_code_sha256,
            },
            "preparation": preparation_dict,
            "quality_report": trainer.get_quality_report(),
            "training": result,
            "execution_time_ms": elapsed_ms,
            "input_data": input_data,
        }

    except HTTPException:
        raise

    except UploadTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Le fichier nécessaire au Fine-Tuning est introuvable.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise_api_error(
            operation="fine_tuning.train.runtime",
            error=exc,
            detail="Le Fine-Tuning n'a pas pu être exécuté.",
        )

    except Exception as exc:
        raise_api_error(
            operation="fine_tuning.train",
            error=exc,
            detail="Une erreur inattendue est survenue pendant le Fine-Tuning.",
        )

    finally:
        await file.close()

        if dataset_path.exists():
            dataset_path.unlink()


@router.post("/reset/{model_name}")
def reset_fine_tuned_model(
    model_name: str,
) -> dict[str, Any]:
    """Réinitialise un modèle vers sa version pré-entraînée."""

    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Modèle inconnu : {model_name}",
        )

    try:
        trainer = FineTuningTrainer()
        return trainer.reset_model(model_name=model_name)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail="Aucun modèle Fine-Tuning à réinitialiser n'a été trouvé.",
        ) from error

    except Exception as error:
        raise_api_error(
            operation="fine_tuning.reset",
            error=error,
            detail="Impossible de réinitialiser le modèle Fine-Tuning.",
        )


@router.post("/predict")
def predict_with_fine_tuned_model(
    request: PredictionRequest,
) -> dict[str, Any]:
    """Réalise une prédiction avec un modèle fine-tuné."""

    if request.model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Modèle inconnu : {request.model_name}. "
                f"Modèles disponibles : {list(AVAILABLE_MODELS.keys())}"
            ),
        )

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Le texte à analyser est obligatoire.",
        )

    try:
        trainer = FineTuningTrainer()
        return trainer.predict(
            model_name=request.model_name,
            text=request.text.strip(),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail="Le modèle Fine-Tuning demandé est introuvable.",
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise_api_error(
            operation="fine_tuning.predict.runtime",
            error=error,
            detail="Le modèle Fine-Tuning ne peut pas effectuer la prédiction.",
        )

    except Exception as error:
        raise_api_error(
            operation="fine_tuning.predict",
            error=error,
            detail="Impossible d'effectuer la prédiction Fine-Tuning.",
        )
