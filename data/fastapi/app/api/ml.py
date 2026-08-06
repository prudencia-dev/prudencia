from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd
from app.ai.machine_learning.trainer import MachineLearningTrainer
from app.services.upload_security import (
    MAX_CSV_UPLOAD_BYTES,
    UploadTooLargeError,
    read_limited_file,
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


class PredictionRequest(BaseModel):
    """Données envoyées au modèle pour effectuer une prédiction."""

    data: list[dict[str, Any]]


router = APIRouter(
    prefix="/ml",
    tags=["Machine Learning"],
)

MODEL_PATH = Path("models") / "machine_learning" / "random_forest.joblib"


def _parse_optional_integer(value: str, field_name: str) -> int | None:
    """Transforme une chaîne vide en None, ou convertit la valeur en entier."""

    normalized_value = value.strip()
    if not normalized_value:
        return None

    try:
        parsed_value = int(normalized_value)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=f"Le paramètre '{field_name}' doit être un nombre entier.",
        ) from error

    if parsed_value < 1:
        raise HTTPException(
            status_code=400,
            detail=f"Le paramètre '{field_name}' doit être supérieur ou égal à 1.",
        )

    return parsed_value


def _parse_optional_choice(
    value: str,
    allowed_values: set[str],
    field_name: str,
) -> str | None:
    """Convertit les valeurs 'none' ou vides en None et valide le choix."""

    normalized_value = value.strip().lower()
    if normalized_value in {"", "none", "aucun", "automatique"}:
        return None

    if normalized_value not in allowed_values:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Valeur invalide pour '{field_name}'. Valeurs autorisées : "
                f"{', '.join(sorted(allowed_values))}."
            ),
        )

    return normalized_value


@router.get("/health")
def ml_health() -> dict[str, Any]:
    """Retourne l'état du moteur Machine Learning."""

    model_exists = MODEL_PATH.exists()

    return {
        "status": "ready" if model_exists else "not_trained",
        "model": "Random Forest",
        "version": (
            MachineLearningTrainer.MODEL_VERSION
            if model_exists
            else "Non disponible"
        ),
        "model_path": str(MODEL_PATH),
        "available": model_exists,
    }


@router.post("/train")
def train_machine_learning(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    feature_columns: str = Form(...),
    n_estimators: int = Form(300),
    max_depth: str = Form(""),
    min_samples_split: int = Form(2),
    min_samples_leaf: int = Form(1),
    max_features: str = Form("sqrt"),
    criterion: str = Form("gini"),
    bootstrap: bool = Form(True),
    class_weight: str = Form("none"),
    random_state: int = Form(42),
    test_size: float = Form(0.20),
) -> dict[str, Any]:
    """Entraîne un Random Forest à partir d'un fichier CSV."""

    try:
        sanitized_filename(file.filename, ".csv")
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    selected_features = [
        column.strip()
        for column in feature_columns.split(",")
        if column.strip()
    ]

    if not selected_features:
        raise HTTPException(
            status_code=400,
            detail="Sélectionnez au minimum une Feature.",
        )

    parsed_max_depth = _parse_optional_integer(
        value=max_depth,
        field_name="max_depth",
    )

    parsed_max_features = _parse_optional_choice(
        value=max_features,
        allowed_values={"sqrt", "log2"},
        field_name="max_features",
    )

    parsed_class_weight = _parse_optional_choice(
        value=class_weight,
        allowed_values={"balanced", "balanced_subsample"},
        field_name="class_weight",
    )

    normalized_criterion = criterion.strip().lower()
    if normalized_criterion not in {"gini", "entropy", "log_loss"}:
        raise HTTPException(
            status_code=400,
            detail=(
                "Le paramètre 'criterion' doit valoir "
                "'gini', 'entropy' ou 'log_loss'."
            ),
        )

    if n_estimators < 10 or n_estimators > 5000:
        raise HTTPException(
            status_code=400,
            detail="n_estimators doit être compris entre 10 et 5000.",
        )

    if min_samples_split < 2:
        raise HTTPException(
            status_code=400,
            detail="min_samples_split doit être supérieur ou égal à 2.",
        )

    if min_samples_leaf < 1:
        raise HTTPException(
            status_code=400,
            detail="min_samples_leaf doit être supérieur ou égal à 1.",
        )

    if not 0.10 <= test_size <= 0.40:
        raise HTTPException(
            status_code=400,
            detail="test_size doit être compris entre 0.10 et 0.40.",
        )

    temporary_path: Path | None = None

    try:
        file_content = read_limited_file(
            file.file,
            max_bytes=MAX_CSV_UPLOAD_BYTES,
        )

        with NamedTemporaryFile(
            mode="wb",
            suffix=".csv",
            delete=False,
        ) as temporary_file:
            temporary_file.write(file_content)
            temporary_path = Path(temporary_file.name)

        trainer = MachineLearningTrainer(
            n_estimators=n_estimators,
            max_depth=parsed_max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=parsed_max_features,
            criterion=normalized_criterion,
            bootstrap=bootstrap,
            class_weight=parsed_class_weight,
            random_state=random_state,
            test_size=test_size,
        )

        return trainer.train(
            csv_path=str(temporary_path),
            feature_columns=selected_features,
            target_column=target_column,
        )

    except UploadTooLargeError as error:
        raise HTTPException(
            status_code=413,
            detail=str(error),
        ) from error

    except KeyError as error:
        raise HTTPException(
            status_code=400,
            detail=f"Une colonne demandée est introuvable : {error}",
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant l'entraînement Machine Learning : {error}",
        ) from error

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


@router.post("/reset")
def reset_machine_learning() -> dict[str, Any]:
    """Réinitialise le modèle Random Forest."""

    try:
        trainer = MachineLearningTrainer()
        return trainer.reset_model()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur pendant la réinitialisation "
                f"du modèle Machine Learning : {error}"
            ),
        ) from error


@router.post("/predict")
def predict_machine_learning(
    request: PredictionRequest,
) -> dict[str, Any]:
    """Réalise une prédiction avec le modèle entraîné."""

    try:
        dataframe = pd.DataFrame(request.data)
        trainer = MachineLearningTrainer()
        return trainer.predict(dataframe)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction Machine Learning : {error}",
        ) from error
