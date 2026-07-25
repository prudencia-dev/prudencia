from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from pathlib import Path

import pandas as pd


@dataclass
class DatasetInfo:
    filename: str
    total_rows: int
    usable_rows: int
    total_columns: int
    text_column: str
    label_column: str
    number_of_classes: int
    classes: list[str]
    missing_values: int
    duplicate_rows: int


@dataclass
class DatasetQualityReport:
    info: DatasetInfo
    quality_score: int
    quality_level: str
    checks: list[dict[str, Any]]
    warnings: list[str]
    recommendations: list[str]
    class_distribution: list[dict[str, Any]]
    is_trainable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetManager:
    """Charge, valide et évalue un dataset de Fine-Tuning."""

    MINIMUM_EXAMPLES = 10
    RECOMMENDED_EXAMPLES = 100
    MINIMUM_CLASSES = 2

    def __init__(self) -> None:
        self.dataframe: pd.DataFrame | None = None
        self.filename: str | None = None

    def load_csv(self, path: Path) -> pd.DataFrame:

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Le fichier Dataset est introuvable : {path}"
            )

        separators = [
            ";",
            ",",
            "\t",
        ]

        encodings = [
            "utf-8",
            "latin-1",
        ]

        dataframe = None

        for encoding in encodings:

            for separator in separators:

                try:

                    dataframe = pd.read_csv(
                        path,
                        sep=separator,
                        quotechar='"',
                        engine="python",
                        encoding=encoding,
                    )

                    if len(dataframe.columns) >= 2:
                        break

                except Exception:
                    dataframe = None

            if dataframe is not None:
                break

        if dataframe is None or dataframe.empty:

            raise ValueError(
                "Impossible de lire le Dataset."
            )

        self.filename = path.name
        self.dataframe = dataframe

        return dataframe

    def validate(
        self,
        text_column: str,
        label_column: str,
    ) -> DatasetQualityReport:
        if self.dataframe is None:
            raise RuntimeError(
                "Aucun dataset n'a été chargé."
            )

        if text_column == label_column:
            raise ValueError(
                "La colonne texte et la colonne cible "
                "doivent être différentes."
            )

        if text_column not in self.dataframe.columns:
            raise ValueError(
                f"Colonne texte absente : {text_column}"
            )

        if label_column not in self.dataframe.columns:
            raise ValueError(
                f"Colonne cible absente : {label_column}"
            )

        dataframe = self.dataframe.copy()

        total_rows = len(dataframe)
        total_columns = len(dataframe.columns)

        missing_values = int(
            dataframe[[text_column, label_column]]
            .isna()
            .sum()
            .sum()
        )

        cleaned_dataframe = (
            dataframe[[text_column, label_column]]
            .dropna()
            .copy()
        )

        cleaned_dataframe[text_column] = (
            cleaned_dataframe[text_column]
            .astype(str)
            .str.strip()
        )

        cleaned_dataframe[label_column] = (
            cleaned_dataframe[label_column]
            .astype(str)
            .str.strip()
        )

        cleaned_dataframe = cleaned_dataframe[
            (cleaned_dataframe[text_column] != "")
            & (cleaned_dataframe[label_column] != "")
        ]

        duplicate_rows = int(
            cleaned_dataframe.duplicated().sum()
        )

        usable_dataframe = (
            cleaned_dataframe
            .drop_duplicates()
            .reset_index(drop=True)
        )

        classes = sorted(
            usable_dataframe[label_column]
            .unique()
            .tolist()
        )

        class_counts = (
            usable_dataframe[label_column]
            .value_counts()
        )

        class_distribution = [
            {
                "class": str(class_name),
                "count": int(count),
                "percentage": round(
                    count / len(usable_dataframe) * 100,
                    2,
                )
                if len(usable_dataframe) > 0
                else 0,
            }
            for class_name, count in class_counts.items()
        ]

        info = DatasetInfo(
            filename=self.filename or "dataset.csv",
            total_rows=total_rows,
            usable_rows=len(usable_dataframe),
            total_columns=total_columns,
            text_column=text_column,
            label_column=label_column,
            number_of_classes=len(classes),
            classes=classes,
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
        )

        return self._build_quality_report(
            info=info,
            class_counts=class_counts,
            class_distribution=class_distribution,
        )

    def _build_quality_report(
        self,
        info: DatasetInfo,
        class_counts: pd.Series,
        class_distribution: list[dict[str, Any]],
    ) -> DatasetQualityReport:
        score = 0
        checks: list[dict[str, Any]] = []
        warnings: list[str] = []
        recommendations: list[str] = []

        def add_check(
            name: str,
            passed: bool,
            points: int,
            message: str,
        ) -> None:
            nonlocal score

            if passed:
                score += points

            checks.append(
                {
                    "name": name,
                    "passed": passed,
                    "points": points if passed else 0,
                    "maximum_points": points,
                    "message": message,
                }
            )

        add_check(
            name="Dataset chargé",
            passed=info.total_rows > 0,
            points=10,
            message=(
                "Le dataset contient des données."
                if info.total_rows > 0
                else "Le dataset est vide."
            ),
        )

        add_check(
            name="Colonne texte valide",
            passed=bool(info.text_column),
            points=10,
            message=f"Colonne texte : {info.text_column}",
        )

        add_check(
            name="Colonne cible valide",
            passed=bool(info.label_column),
            points=10,
            message=f"Colonne cible : {info.label_column}",
        )

        no_missing_values = info.missing_values == 0

        add_check(
            name="Valeurs manquantes",
            passed=no_missing_values,
            points=15,
            message=(
                "Aucune valeur manquante."
                if no_missing_values
                else (
                    f"{info.missing_values} valeur(s) "
                    "manquante(s) détectée(s)."
                )
            ),
        )

        if not no_missing_values:
            warnings.append(
                f"{info.missing_values} valeur(s) "
                "manquante(s) ont été détectée(s)."
            )
            recommendations.append(
                "Compléter ou supprimer les lignes "
                "contenant des valeurs manquantes."
            )

        enough_classes = (
            info.number_of_classes >= self.MINIMUM_CLASSES
        )

        add_check(
            name="Nombre de classes",
            passed=enough_classes,
            points=15,
            message=(
                f"{info.number_of_classes} classe(s) détectée(s)."
            ),
        )

        if not enough_classes:
            warnings.append(
                "Le dataset doit contenir au moins deux classes."
            )
            recommendations.append(
                "Ajouter des exemples appartenant "
                "à une seconde classe."
            )

        enough_examples = (
            info.usable_rows >= self.RECOMMENDED_EXAMPLES
        )

        minimum_examples_reached = (
            info.usable_rows >= self.MINIMUM_EXAMPLES
        )

        add_check(
            name="Taille du dataset",
            passed=enough_examples,
            points=15,
            message=(
                f"{info.usable_rows} exemple(s) exploitable(s)."
            ),
        )

        if not minimum_examples_reached:
            warnings.append(
                "Le dataset contient trop peu d'exemples "
                "pour lancer un entraînement."
            )
            recommendations.append(
                f"Ajouter au moins "
                f"{self.MINIMUM_EXAMPLES - info.usable_rows} "
                "exemple(s)."
            )
        elif not enough_examples:
            warnings.append(
                "Le dataset est exploitable, mais reste petit."
            )
            recommendations.append(
                "Ajouter davantage d'exemples "
                "pour améliorer la généralisation."
            )

        balanced = self._is_balanced(class_counts)

        add_check(
            name="Équilibre des classes",
            passed=balanced,
            points=15,
            message=(
                "Les classes sont suffisamment équilibrées."
                if balanced
                else "Les classes sont déséquilibrées."
            ),
        )

        if not balanced:
            warnings.append(
                "La répartition des classes est déséquilibrée."
            )
            recommendations.append(
                "Ajouter des exemples aux classes "
                "les moins représentées."
            )

        no_duplicates = info.duplicate_rows == 0

        add_check(
            name="Doublons",
            passed=no_duplicates,
            points=10,
            message=(
                "Aucun doublon détecté."
                if no_duplicates
                else (
                    f"{info.duplicate_rows} doublon(s) "
                    "détecté(s)."
                )
            ),
        )

        if not no_duplicates:
            warnings.append(
                f"{info.duplicate_rows} doublon(s) "
                "ont été détecté(s)."
            )
            recommendations.append(
                "Supprimer les doublons avant l'entraînement."
            )

        quality_level = self._get_quality_level(score)

        is_trainable = (
            minimum_examples_reached
            and enough_classes
            and info.usable_rows > 0
        )

        if is_trainable and not recommendations:
            recommendations.append(
                "Le dataset est prêt pour le Fine-Tuning."
            )

        return DatasetQualityReport(
            info=info,
            quality_score=score,
            quality_level=quality_level,
            checks=checks,
            warnings=warnings,
            recommendations=recommendations,
            class_distribution=class_distribution,
            is_trainable=is_trainable,
        )

    @staticmethod
    def _is_balanced(
        class_counts: pd.Series,
    ) -> bool:
        if class_counts.empty or len(class_counts) < 2:
            return False

        smallest_class = int(class_counts.min())
        largest_class = int(class_counts.max())

        if largest_class == 0:
            return False

        balance_ratio = smallest_class / largest_class

        return balance_ratio >= 0.50

    @staticmethod
    def _get_quality_level(score: int) -> str:
        if score >= 90:
            return "Excellent"

        if score >= 75:
            return "Bon"

        if score >= 60:
            return "Correct"

        if score >= 40:
            return "Faible"

        return "Insuffisant"

    def build_training_dataframe(
        self,
        text_column: str,
        label_column: str,
    ) -> pd.DataFrame:
        if self.dataframe is None:
            raise RuntimeError(
                "Aucun dataset n'a été chargé."
            )

        training_dataframe = (
            self.dataframe[[text_column, label_column]]
            .dropna()
            .copy()
        )

        training_dataframe[text_column] = (
            training_dataframe[text_column]
            .astype(str)
            .str.strip()
        )

        training_dataframe[label_column] = (
            training_dataframe[label_column]
            .astype(str)
            .str.strip()
        )

        training_dataframe = training_dataframe[
            (training_dataframe[text_column] != "")
            & (training_dataframe[label_column] != "")
        ]

        training_dataframe = (
            training_dataframe
            .drop_duplicates()
            .rename(
                columns={
                    text_column: "text",
                    label_column: "label",
                }
            )
            .reset_index(drop=True)
        )

        return training_dataframe