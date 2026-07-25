from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel

from app.ai.machine_learning.trainer import (
    MachineLearningTrainer,
)


class PredictionRequest(BaseModel):
    """
    Données envoyées au modèle pour effectuer une prédiction.

    Chaque dictionnaire représente une ligne de données.
    """

    data: list[dict[str, Any]]


router = APIRouter(
    prefix="/ml",
    tags=["Machine Learning"],
)


MODEL_PATH = (
    Path("models")
    / "machine_learning"
    / "random_forest.joblib"
)


@router.get("/health")
def ml_health() -> dict[str, Any]:
    """
    Retourne l'état du moteur Machine Learning.

    Le modèle est considéré comme disponible uniquement
    si le fichier Joblib existe.
    """

    model_exists = MODEL_PATH.exists()

    return {
        "status": (
            "ready"
            if model_exists
            else "not_trained"
        ),
        "model": "Random Forest",
        "version": (
            "v1.0.0"
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
    n_estimators: int = Form(100),
    max_depth: str = Form(""),
    random_state: int = Form(42),
) -> dict[str, Any]:
    """
    Entraîne un Random Forest à partir d'un fichier CSV.

    feature_columns est reçu sous forme de texte :

    colonne_1,colonne_2,colonne_3
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Le fichier ne possède pas de nom.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Seuls les fichiers CSV sont acceptés.",
        )

    # Transformation de la liste reçue depuis Streamlit.
    selected_features = [
        column.strip()
        for column in feature_columns.split(",")
        if column.strip()
    ]

    if not selected_features:
        raise HTTPException(
            status_code=400,
            detail=(
                "Sélectionnez au minimum une Feature."
            ),
        )

    parsed_max_depth: int | None

    if max_depth.strip():
        try:
            parsed_max_depth = int(max_depth)
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=(
                    "La profondeur maximale doit être "
                    "un nombre entier."
                ),
            ) from error
    else:
        parsed_max_depth = None

    temporary_path: Path | None = None

    try:
        # Le fichier est lu en binaire pour préserver
        # son encodage d'origine.
        file_content = file.file.read()

        with NamedTemporaryFile(
            mode="wb",
            suffix=".csv",
            delete=False,
        ) as temporary_file:
            temporary_file.write(
                file_content
            )

            temporary_path = Path(
                temporary_file.name
            )

        trainer = MachineLearningTrainer(
            n_estimators=n_estimators,
            max_depth=parsed_max_depth,
            random_state=random_state,
        )

        return trainer.train(
            csv_path=str(temporary_path),
            feature_columns=selected_features,
            target_column=target_column,
        )

    except KeyError as error:
        raise HTTPException(
            status_code=400,
            detail=(
                "Une colonne demandée est introuvable : "
                f"{error}"
            ),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur pendant l'entraînement "
                f"Machine Learning : {error}"
            ),
        ) from error

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


@router.post("/reset")
def reset_machine_learning() -> dict[str, Any]:
    """
    Réinitialise le modèle Random Forest.
    """

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
    """
    Réalise une prédiction avec le modèle entraîné.
    """

    try:
        dataframe = pd.DataFrame(
            request.data
        )

        trainer = MachineLearningTrainer()

        return trainer.predict(
            dataframe,
        )

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
            detail=(
                "Erreur pendant la prédiction "
                f"Machine Learning : {error}"
            ),
        ) from error