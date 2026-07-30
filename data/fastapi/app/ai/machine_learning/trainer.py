from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.services.training_history_service import (
    save_model_reset,
    save_training_execution,
)


class MachineLearningTrainer:
    """
    Gère l'entraînement, la prédiction et la réinitialisation
    du modèle Machine Learning de PRUDENCIA.

    Pipeline général :

    Dataset CSV
        ↓
    Sélection des Features
        ↓
    Sélection de la Target
        ↓
    Prétraitement
        ↓
    Random Forest
        ↓
    Évaluation
        ↓
    Sauvegarde Joblib
    """

    MODEL_PATH = (
        Path("models")
        / "machine_learning"
        / "random_forest.joblib"
    )

    # Version unique utilisée par l'API, l'historique,
    # les rapports et la réinitialisation.
    MODEL_VERSION = "v1.1.0"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | None = "sqrt",
        criterion: str = "gini",
        bootstrap: bool = True,
        class_weight: str | None = None,
        random_state: int = 42,
        test_size: float = 0.20,
    ) -> None:
        """
        Initialise les hyperparamètres du Random Forest.

        n_estimators :
            Nombre d'arbres de décision.

        max_depth :
            Profondeur maximale de chaque arbre.
            None signifie que la profondeur n'est pas limitée.

        random_state :
            Graine aléatoire permettant de reproduire
            les mêmes résultats.
        """

        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.criterion = criterion
        self.bootstrap = bool(bootstrap)
        self.class_weight = class_weight
        self.random_state = int(random_state)
        self.test_size = float(test_size)

        if self.n_estimators < 1:
            raise ValueError("n_estimators doit être supérieur ou égal à 1.")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth doit être supérieur ou égal à 1.")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split doit être supérieur ou égal à 2.")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf doit être supérieur ou égal à 1.")
        if self.max_features not in {"sqrt", "log2", None}:
            raise ValueError("max_features doit valoir sqrt, log2 ou None.")
        if self.criterion not in {"gini", "entropy", "log_loss"}:
            raise ValueError("criterion invalide.")
        if self.class_weight not in {None, "balanced", "balanced_subsample"}:
            raise ValueError("class_weight invalide.")
        if not 0.10 <= self.test_size <= 0.40:
            raise ValueError("test_size doit être compris entre 0.10 et 0.40.")

        self.model: Pipeline | None = None

    # ================================================================
    # Lecture du Dataset
    # ================================================================

    def _read_dataset(
        self,
        csv_path: str,
    ) -> pd.DataFrame:
        """
        Lit un fichier CSV en testant plusieurs encodages
        et plusieurs séparateurs.

        Cette méthode accepte notamment les CSV produits
        par Excel sous Windows.
        """

        encodings = [
            "utf-8",
            "utf-8-sig",
            "cp1252",
            "latin-1",
        ]

        separators = [
            ";",
            ",",
            "\t",
        ]

        last_error: Exception | None = None

        for encoding in encodings:
            for separator in separators:
                try:
                    dataframe = pd.read_csv(
                        csv_path,
                        encoding=encoding,
                        sep=separator,
                    )

                    if len(dataframe.columns) >= 2:
                        return dataframe

                except (
                    UnicodeDecodeError,
                    pd.errors.ParserError,
                    pd.errors.EmptyDataError,
                ) as error:
                    last_error = error

        raise ValueError(
            "Impossible de lire le fichier CSV. "
            "Vérifiez son encodage, son séparateur "
            "et la présence d'au moins deux colonnes."
        ) from last_error

    # ================================================================
    # Validation
    # ================================================================

    def _validate_dataset(
        self,
        dataframe: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> None:
        """
        Vérifie que le Dataset et les colonnes sélectionnées
        permettent un apprentissage supervisé.
        """

        if dataframe.empty:
            raise ValueError(
                "Le Dataset est vide."
            )

        if target_column not in dataframe.columns:
            raise ValueError(
                f"La Target '{target_column}' n'existe pas "
                "dans le Dataset."
            )

        if not feature_columns:
            raise ValueError(
                "Sélectionnez au minimum une Feature."
            )

        missing_features = [
            column
            for column in feature_columns
            if column not in dataframe.columns
        ]

        if missing_features:
            raise ValueError(
                "Les Features suivantes n'existent pas "
                "dans le Dataset : "
                + ", ".join(missing_features)
            )

        if target_column in feature_columns:
            raise ValueError(
                "La Target ne peut pas également être utilisée "
                "comme Feature."
            )

        if dataframe[target_column].isna().all():
            raise ValueError(
                f"La Target '{target_column}' ne contient "
                "aucune valeur exploitable."
            )

        if dataframe[target_column].nunique() < 2:
            raise ValueError(
                f"La Target '{target_column}' doit contenir "
                "au minimum deux classes différentes."
            )

        if len(dataframe) < 5:
            raise ValueError(
                "Le Dataset doit contenir au minimum cinq lignes."
            )

    # ================================================================
    # Prétraitement
    # ================================================================

    def _build_pipeline(
        self,
        features: pd.DataFrame,
    ) -> Pipeline:
        """
        Construit le Pipeline Machine Learning.

        Les Features numériques :
            valeurs manquantes
                ↓
            remplacement par la médiane

        Les Features catégorielles :
            valeurs manquantes
                ↓
            valeur la plus fréquente
                ↓
            OneHotEncoder

        Toutes les Features préparées :
            ↓
        Random Forest
        """

        numeric_columns = features.select_dtypes(
            include=[
                "number",
                "bool",
            ],
        ).columns.tolist()

        categorical_columns = features.select_dtypes(
            exclude=[
                "number",
                "bool",
            ],
        ).columns.tolist()

        transformers: list[
            tuple[str, Any, list[str]]
        ] = []

        if numeric_columns:
            numeric_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                        ),
                    ),
                ],
            )

            transformers.append(
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_columns,
                )
            )

        if categorical_columns:
            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent",
                        ),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                        ),
                    ),
                ],
            )

            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns,
                )
            )

        if not transformers:
            raise ValueError(
                "Aucune Feature exploitable n'a été détectée."
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            criterion=self.criterion,
            bootstrap=self.bootstrap,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )

        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    classifier,
                ),
            ],
        )

    # ================================================================
    # Séparation Train / Test
    # ================================================================

    def _split_dataset(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        bool,
    ]:
        """
        Sépare les données en deux ensembles.

        Train :
            données utilisées pour apprendre.

        Test :
            données jamais vues pendant l'entraînement,
            utilisées pour calculer les métriques.

        La stratification conserve la proportion des classes,
        mais elle n'est possible que si chaque classe possède
        suffisamment d'exemples.
        """

        class_counts = target.value_counts()
        class_count = int(target.nunique())

        test_row_count = max(
            1,
            round(len(target) * self.test_size),
        )

        use_stratification = (
            class_count > 1
            and int(class_counts.min()) >= 2
            and test_row_count >= class_count
        )

        (
            x_train,
            x_test,
            y_train,
            y_test,
        ) = train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=(
                target
                if use_stratification
                else None
            ),
        )

        return (
            x_train,
            x_test,
            y_train,
            y_test,
            use_stratification,
        )

    # ================================================================
    # Entraînement
    # ================================================================

    def train(
        self,
        csv_path: str,
        feature_columns: list[str],
        target_column: str,
    ) -> dict[str, Any]:
        """
        Entraîne le Random Forest.

        feature_columns :
            colonnes utilisées pour effectuer la prédiction.

        target_column :
            bonne réponse connue pendant l'entraînement.
            La Target est aussi appelée Label.
        """

        start_time = time.perf_counter()

        dataframe = self._read_dataset(
            csv_path,
        )

        # Nettoyage des noms de colonnes.
        dataframe.columns = [
            str(column).strip()
            for column in dataframe.columns
        ]

        feature_columns = [
            column.strip()
            for column in feature_columns
        ]

        target_column = target_column.strip()

        self._validate_dataset(
            dataframe=dataframe,
            feature_columns=feature_columns,
            target_column=target_column,
        )

        # X représente les Features.
        features = dataframe[
            feature_columns
        ].copy()

        # y représente la Target ou Label.
        target = dataframe[
            target_column
        ].copy()

        # Suppression des lignes dont la Target est absente.
        valid_target_mask = target.notna()

        features = features.loc[
            valid_target_mask
        ].reset_index(drop=True)

        target = target.loc[
            valid_target_mask
        ].reset_index(drop=True)

        # Les colonnes constantes ne permettent pas
        # de distinguer les différentes classes.
        constant_columns = [
            column
            for column in features.columns
            if features[column].nunique(
                dropna=False,
            ) <= 1
        ]

        if constant_columns:
            features = features.drop(
                columns=constant_columns,
            )

        if features.empty:
            raise ValueError(
                "Aucune Feature exploitable ne reste "
                "après le nettoyage."
            )

        (
            x_train,
            x_test,
            y_train,
            y_test,
            stratification_used,
        ) = self._split_dataset(
            features=features,
            target=target,
        )

        self.model = self._build_pipeline(
            features,
        )

        # Apprentissage du modèle sur le jeu Train.
        self.model.fit(
            x_train,
            y_train,
        )

        # Prédiction sur le jeu Test.
        predictions = self.model.predict(
            x_test,
        )

        class_names = sorted({str(v) for v in list(y_test) + list(predictions)})
        report = classification_report(
            y_test, predictions, labels=class_names,
            output_dict=True, zero_division=0,
        )
        per_class = {
            name: {
                "precision": float(report.get(name, {}).get("precision", 0.0)),
                "recall": float(report.get(name, {}).get("recall", 0.0)),
                "f1": float(report.get(name, {}).get("f1-score", 0.0)),
                "support": int(report.get(name, {}).get("support", 0)),
            }
            for name in class_names
        }
        matrix = confusion_matrix(y_test, predictions, labels=class_names)

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
            "precision_weighted": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test, predictions)),
            "class_names": class_names,
            "per_class": per_class,
            "confusion_matrix": matrix.tolist(),
        }

        preprocessor = self.model.named_steps["preprocessor"]
        classifier = self.model.named_steps["classifier"]
        transformed_names = preprocessor.get_feature_names_out()
        feature_importance = sorted(
            [
                {"feature": str(name), "importance": float(value)}
                for name, value in zip(transformed_names, classifier.feature_importances_, strict=False)
            ],
            key=lambda row: row["importance"],
            reverse=True,
        )

        self.MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Le fichier Joblib contient à la fois :
        # - le prétraitement ;
        # - l'encodage ;
        # - le Random Forest entraîné.
        joblib.dump(
            self.model,
            self.MODEL_PATH,
        )

        execution_time_seconds = (
            time.perf_counter()
            - start_time
        )

        execution_time_ms = int(
            execution_time_seconds * 1000
        )

        duplicate_rows = int(dataframe.duplicated().sum())
        missing_values = int(features.isna().sum().sum())
        total_cells = int(features.shape[0] * features.shape[1])
        missing_values_ratio = missing_values / total_cells if total_cells else 0.0

        target_distribution = {
            str(label): int(count)
            for label, count
            in target.value_counts().items()
        }

        rare_classes = [
            str(label)
            for label, count
            in target.value_counts().items()
            if int(count) < 2
        ]

        warnings: list[str] = []

        if not stratification_used:
            warnings.append(
                "La séparation Train/Test n'a pas été stratifiée, "
                "car certaines classes contiennent trop peu "
                "d'exemples."
            )

        if rare_classes:
            warnings.append(
                "Certaines classes ne contiennent qu'un seul "
                "exemple : "
                + ", ".join(rare_classes)
            )

        if duplicate_rows:
            warnings.append(f"{duplicate_rows} ligne(s) dupliquée(s) détectée(s).")

        if missing_values_ratio >= 0.10:
            warnings.append("Plus de 10 % des valeurs des Features sont manquantes.")

        if len(dataframe) < 100:
            warnings.append(
                "Le Dataset contient moins de 100 lignes. "
                "Les métriques doivent être interprétées "
                "avec prudence."
            )

        save_training_execution(
            model_type="machine_learning",
            model_name="Random Forest",
            model_version=self.MODEL_VERSION,
            task_name="classification",
            execution_type="training",
            dataset_name=Path(csv_path).name,
            dataset_rows=len(dataframe),
            input_data={
                "algorithm": "RandomForestClassifier",
                "target_column": target_column,
                "feature_columns": (
                    features.columns.tolist()
                ),
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features,
                "criterion": self.criterion,
                "bootstrap": self.bootstrap,
                "class_weight": self.class_weight,
                "random_state": self.random_state,
                "test_size": self.test_size,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "stratification_used": (
                    stratification_used
                ),
            },
            output_data={
                "metrics": metrics,
                "target_distribution": (
                    target_distribution
                ),
                "feature_importance": feature_importance,
                "training": {
                    "training_examples": len(x_train),
                    "validation_examples": len(x_test),
                    "metrics": metrics,
                    "feature_importance": feature_importance,
                },
                "warnings": warnings,
            },
            execution_time_ms=execution_time_ms,
            success=True,
        )

        return {
            "success": True,
            "model_name": "Random Forest",
            "model_version": self.MODEL_VERSION,
            "model_path": str(
                self.MODEL_PATH
            ),
            "dataset_rows": len(dataframe),
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "feature_count": len(
                features.columns
            ),
            "feature_columns": (
                features.columns.tolist()
            ),
            "target_column": target_column,
            "class_count": int(
                target.nunique()
            ),
            "target_distribution": (
                target_distribution
            ),
            "stratification_used": (
                stratification_used
            ),
            "constant_columns_removed": (
                constant_columns
            ),
            "duplicate_rows": duplicate_rows,
            "missing_values": missing_values,
            "missing_values_ratio": missing_values_ratio,
            "feature_importance": feature_importance,
            "training_time": round(
                execution_time_seconds,
                3,
            ),
            "hyperparameters": {
                "algorithm": "RandomForestClassifier",
                "target_column": target_column,
                "feature_columns": features.columns.tolist(),
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features,
                "criterion": self.criterion,
                "bootstrap": self.bootstrap,
                "class_weight": self.class_weight,
                "random_state": self.random_state,
                "test_size": self.test_size,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "stratification_used": stratification_used,
            },
            "metrics": metrics,
            "expected_labels": [
                str(value)
                for value in y_test.tolist()
            ],
            "predicted_labels": [
                str(value)
                for value
                in predictions.tolist()
            ],
            "warnings": warnings,
        }

    # ================================================================
    # Prédiction
    # ================================================================

    def predict(
        self,
        input_dataframe: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Prédit la classe de nouvelles données
        avec le modèle sauvegardé.
        """

        if not self.MODEL_PATH.exists():
            raise RuntimeError(
                "Le modèle Machine Learning "
                "n'est pas entraîné."
            )

        if input_dataframe.empty:
            raise ValueError(
                "Aucune donnée n'a été fournie."
            )

        model: Pipeline = joblib.load(
            self.MODEL_PATH,
        )

        predictions = model.predict(
            input_dataframe,
        )

        probabilities = model.predict_proba(
            input_dataframe,
        )

        classifier = model.named_steps[
            "classifier"
        ]

        return {
            "success": True,
            "predictions": [
                str(value)
                for value
                in predictions.tolist()
            ],
            "probabilities": (
                probabilities.tolist()
            ),
            "classes": [
                str(value)
                for value
                in classifier.classes_.tolist()
            ],
        }

    # ================================================================
    # Réinitialisation
    # ================================================================

    def reset_model(
        self,
    ) -> dict[str, Any]:
        """
        Supprime le fichier Joblib et remet
        le moteur dans son état non entraîné.
        """

        model_deleted = False

        if self.MODEL_PATH.exists():
            self.MODEL_PATH.unlink()
            model_deleted = True

        self.model = None

        save_model_reset(
            model_type="machine_learning",
            model_name="Random Forest",
            model_version=self.MODEL_VERSION,
            reason=(
                "Réinitialisation du modèle "
                "Machine Learning depuis PRUDENCIA"
            ),
        )

        return {
            "success": True,
            "message": (
                "Le modèle Machine Learning "
                "a été réinitialisé."
            ),
            "model_name": "Random Forest",
            "model_version": self.MODEL_VERSION,
            "model_deleted": model_deleted,
        }