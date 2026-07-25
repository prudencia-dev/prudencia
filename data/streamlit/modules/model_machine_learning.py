from __future__ import annotations

import os
from io import BytesIO
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv(
    "PRUDENCIA_API_URL",
    os.getenv("API_URL", "http://api:8000"),
)

TRAINING_TIMEOUT = 600


@st.cache_data(ttl=30)
def get_ml_status() -> dict[str, Any]:
    """
    Récupère l'état actuel du modèle Machine Learning.
    """

    response = requests.get(
        f"{API_URL}/ml/health",
        timeout=5,
    )

    response.raise_for_status()

    return response.json()


def read_csv_file(
    uploaded_file: Any,
) -> pd.DataFrame | None:
    """
    Lit un fichier CSV en testant plusieurs encodages
    et plusieurs séparateurs.
    """

    file_content = uploaded_file.getvalue()

    attempts = [
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8"},
        {"sep": "\t", "encoding": "utf-8"},
        {"sep": ";", "encoding": "cp1252"},
        {"sep": ",", "encoding": "cp1252"},
        {"sep": ";", "encoding": "latin-1"},
        {"sep": ",", "encoding": "latin-1"},
    ]

    for options in attempts:
        try:
            dataframe = pd.read_csv(
                BytesIO(file_content),
                **options,
            )

            if len(dataframe.columns) >= 2:
                return dataframe

        except (
            UnicodeDecodeError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
        ):
            continue

    st.error(
        "Le fichier CSV n'a pas pu être lu. "
        "Vérifiez son encodage et son séparateur."
    )

    return None


def find_default_target_column(
    columns: list[str],
) -> int:
    """
    Recherche automatiquement une Target probable.
    """

    expected_names = [
        "grade_global",
        "niveau_risque",
        "classification",
        "target",
        "label",
        "classe",
        "category",
        "categorie",
    ]

    normalized_columns = {
        column.strip().lower(): index
        for index, column in enumerate(columns)
    }

    for expected_name in expected_names:
        if expected_name in normalized_columns:
            return normalized_columns[expected_name]

    return max(0, len(columns) - 1)


def find_default_features(
    columns: list[str],
    target_column: str,
) -> list[str]:
    """
    Présélectionne les Features pertinentes.

    Les colonnes techniques, les scores calculés et les textes libres
    sont exclus par défaut afin d'éviter le bruit et la fuite de données.
    """

    excluded_columns = {
        target_column,
        "id",
        "id_cas",
        "commentaire",
        "comments",
        "score_global",
        "note_juridique",
        "note_ethique",
        "note_responsabilite",
        "created_at",
        "updated_at",
    }

    return [
        column
        for column in columns
        if column.strip().lower() not in excluded_columns
    ]


def display_percentage_metric(
    container: Any,
    label: str,
    value: Any,
) -> None:
    """
    Affiche une métrique sous forme de pourcentage.
    """

    if isinstance(value, (int, float)):
        container.metric(
            label,
            f"{float(value):.2%}",
        )
    else:
        container.metric(
            label,
            "—",
        )


def display_training_report(
    result: dict[str, Any],
) -> None:
    """
    Affiche le rapport détaillé de l'entraînement.
    """

    metrics = result.get(
        "metrics",
        {},
    )

    st.divider()
    st.header("📊 Rapport d'entraînement")

    metric_columns = st.columns(4)

    display_percentage_metric(
        metric_columns[0],
        "Accuracy",
        metrics.get("accuracy"),
    )

    display_percentage_metric(
        metric_columns[1],
        "Precision",
        metrics.get("precision"),
    )

    display_percentage_metric(
        metric_columns[2],
        "Recall",
        metrics.get("recall"),
    )

    display_percentage_metric(
        metric_columns[3],
        "F1 Score",
        metrics.get("f1"),
    )

    st.subheader("Dataset utilisé")

    dataset_columns = st.columns(4)

    dataset_columns[0].metric(
        "Lignes",
        result.get(
            "dataset_rows",
            "—",
        ),
    )

    dataset_columns[1].metric(
        "Features",
        result.get(
            "feature_count",
            "—",
        ),
    )

    dataset_columns[2].metric(
        "Classes",
        result.get(
            "class_count",
            "—",
        ),
    )

    training_time = result.get(
        "training_time",
    )

    dataset_columns[3].metric(
        "Temps",
        (
            f"{training_time:.3f} s"
            if isinstance(
                training_time,
                (int, float),
            )
            else "—"
        ),
    )

    target_column = result.get(
        "target_column",
        "—",
    )

    feature_columns = result.get(
        "feature_columns",
        [],
    )

    st.info(
        f"🎯 Target utilisée : **{target_column}**"
    )

    with st.expander(
        "Features utilisées",
        expanded=False,
    ):
        for feature in feature_columns:
            st.write(
                f"- {feature}"
            )

    target_distribution = result.get(
        "target_distribution",
        {},
    )

    if target_distribution:
        st.subheader(
            "Distribution des classes"
        )

        distribution_dataframe = pd.DataFrame(
            {
                "Classe": list(
                    target_distribution.keys()
                ),
                "Nombre d'exemples": list(
                    target_distribution.values()
                ),
            }
        )

        st.dataframe(
            distribution_dataframe,
            use_container_width=True,
            hide_index=True,
        )

    warnings = result.get(
        "warnings",
        [],
    )

    if warnings:
        st.subheader(
            "⚠️ Avertissements"
        )

        for warning in warnings:
            st.warning(
                warning
            )
    else:
        st.success(
            "Aucune anomalie importante détectée."
        )

    with st.expander(
        "Afficher le résultat technique complet",
        expanded=False,
    ):
        st.json(
            result
        )


def render_machine_learning() -> None:
    """
    Affiche l'interface complète du moteur Machine Learning.
    """

    st.subheader(
        "Machine Learning"
    )

    st.info(
        """
        Cette page permet d'entraîner un modèle Random Forest
        à partir d'un Dataset structuré.

        Les Features correspondent aux informations utilisées
        pour effectuer la prédiction.

        La Target, également appelée Label, correspond à la bonne
        réponse connue pendant l'apprentissage supervisé.
        """
    )

    # ================================================================
    # État du modèle
    # ================================================================

    try:
        status = get_ml_status()

    except requests.RequestException as error:
        status = {
            "status": "unavailable",
            "available": False,
            "error": str(error),
        }

    model_available = bool(
        status.get(
            "available",
            False,
        )
    )

    status_columns = st.columns(3)

    status_columns[0].metric(
        "Modèle",
        status.get(
            "model",
            "Random Forest",
        ),
    )

    status_columns[1].metric(
        "Statut",
        (
            "🟢 Entraîné"
            if model_available
            else "⚪ À entraîner"
        ),
    )

    status_columns[2].metric(
        "Version",
        status.get(
            "version",
            "Non disponible",
        ),
    )

    st.divider()

    # ================================================================
    # Import du Dataset
    # ================================================================

    st.header(
        "📄 Dataset"
    )

    uploaded_file = st.file_uploader(
        "Dataset Machine Learning (.csv)",
        type=[
            "csv",
        ],
        key="ml_dataset",
        help=(
            "Importez un Dataset contenant des Features "
            "et une Target."
        ),
    )

    if uploaded_file is None:
        st.info(
            "Sélectionnez un Dataset CSV pour commencer."
        )
        return

    dataframe = read_csv_file(
        uploaded_file
    )

    if dataframe is None:
        return

    if dataframe.empty:
        st.error(
            "Le Dataset sélectionné est vide."
        )
        return

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    columns = dataframe.columns.tolist()

    if len(columns) < 2:
        st.error(
            "Le Dataset doit contenir au minimum "
            "une Feature et une Target."
        )
        return

    st.success(
        f"Dataset chargé : {len(dataframe)} lignes "
        f"et {len(columns)} colonnes."
    )

    with st.expander(
        "Prévisualiser le Dataset",
        expanded=True,
    ):
        st.dataframe(
            dataframe.head(20),
            use_container_width=True,
            hide_index=True,
        )

    # ================================================================
    # Sélection de la Target
    # ================================================================

    st.header(
        "🎯 Target — Label"
    )

    st.caption(
        "La Target est la valeur que le modèle doit apprendre "
        "à prédire. Elle est connue pendant l'entraînement."
    )

    default_target_index = find_default_target_column(
        columns
    )

    target_column = st.selectbox(
        "Target Column",
        options=columns,
        index=default_target_index,
        help=(
            "Exemple : grade_global, niveau_risque "
            "ou classification."
        ),
    )

    target_class_count = int(
        dataframe[target_column].nunique(
            dropna=True,
        )
    )

    target_distribution = (
        dataframe[target_column]
        .value_counts(
            dropna=False,
        )
        .to_dict()
    )

    target_columns = st.columns(3)

    target_columns[0].metric(
        "Classes",
        target_class_count,
    )

    target_columns[1].metric(
        "Classe majoritaire",
        (
            str(
                dataframe[target_column]
                .value_counts()
                .index[0]
            )
            if target_class_count > 0
            else "—"
        ),
    )

    minimum_class_count = (
        int(
            dataframe[target_column]
            .value_counts()
            .min()
        )
        if target_class_count > 0
        else 0
    )

    target_columns[2].metric(
        "Plus petite classe",
        minimum_class_count,
    )

    if target_class_count > max(
        20,
        len(dataframe) // 2,
    ):
        st.warning(
            "Cette colonne contient presque une valeur différente "
            "par ligne. Elle ressemble davantage à un identifiant "
            "ou à un score continu qu'à un Label de classification."
        )

    if minimum_class_count < 2:
        st.warning(
            "Certaines classes contiennent moins de deux exemples. "
            "Les performances du modèle seront peu fiables."
        )

    # ================================================================
    # Sélection des Features
    # ================================================================

    st.header(
        "📝 Features"
    )

    st.caption(
        "Les Features sont les informations utilisées "
        "pour prédire la Target."
    )

    available_features = [
        column
        for column in columns
        if column != target_column
    ]

    default_features = find_default_features(
        columns=columns,
        target_column=target_column,
    )

    default_features = [
        column
        for column in default_features
        if column in available_features
    ]

    feature_columns = st.multiselect(
        "Feature Columns",
        options=available_features,
        default=default_features,
        help=(
            "Évitez les identifiants, les commentaires libres "
            "et les colonnes calculées à partir de la Target."
        ),
    )

    if not feature_columns:
        st.warning(
            "Sélectionnez au minimum une Feature."
        )

    selected_dataframe = dataframe[
        feature_columns + [target_column]
    ]

    missing_values = int(
        selected_dataframe
        .isna()
        .sum()
        .sum()
    )

    configuration_columns = st.columns(3)

    configuration_columns[0].metric(
        "Features sélectionnées",
        len(feature_columns),
    )

    configuration_columns[1].metric(
        "Valeurs manquantes",
        missing_values,
    )

    ignored_columns = [
        column
        for column in columns
        if (
            column not in feature_columns
            and column != target_column
        )
    ]

    configuration_columns[2].metric(
        "Colonnes ignorées",
        len(ignored_columns),
    )

    with st.expander(
        "Résumé de la configuration",
        expanded=False,
    ):
        st.write(
            f"**Target :** {target_column}"
        )

        st.write(
            "**Features :**"
        )

        for feature in feature_columns:
            st.write(
                f"- {feature}"
            )

        if ignored_columns:
            st.write(
                "**Colonnes ignorées :**"
            )

            for column in ignored_columns:
                st.write(
                    f"- {column}"
                )

    # ================================================================
    # Paramètres du Random Forest
    # ================================================================

    st.header(
        "🌳 Random Forest"
    )

    parameter_columns = st.columns(3)

    n_estimators = parameter_columns[0].number_input(
        "Nombre d'arbres",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help=(
            "Plus le nombre d'arbres est élevé, plus le modèle "
            "peut être stable, mais l'entraînement sera plus long."
        ),
    )

    max_depth_choice = parameter_columns[1].selectbox(
        "Profondeur maximale",
        options=[
            "Automatique",
            "5",
            "10",
            "20",
            "30",
        ],
        index=0,
    )

    random_state = parameter_columns[2].number_input(
        "Random State",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
        help=(
            "Permet de reproduire les mêmes résultats."
        ),
    )

    max_depth = (
        ""
        if max_depth_choice == "Automatique"
        else max_depth_choice
    )

    st.divider()

    # ================================================================
    # Boutons
    # ================================================================

    action_columns = st.columns(2)

    launch_training = action_columns[0].button(
        "🚀 Lancer l'entraînement",
        type="primary",
        use_container_width=True,
        disabled=not feature_columns,
    )

    reset_model = action_columns[1].button(
        "🔄 Réinitialiser le modèle",
        use_container_width=True,
    )

    # ================================================================
    # Réinitialisation
    # ================================================================

    if reset_model:
        try:
            with st.spinner(
                "Réinitialisation du modèle..."
            ):
                response = requests.post(
                    f"{API_URL}/ml/reset",
                    timeout=60,
                )

            response.raise_for_status()

            get_ml_status.clear()

            st.success(
                "Le modèle Machine Learning a été réinitialisé."
            )

            st.rerun()

        except requests.RequestException as error:
            st.error(
                "Impossible de réinitialiser le modèle."
            )

            st.code(
                str(error)
            )

    # ================================================================
    # Entraînement
    # ================================================================

    if launch_training:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "text/csv",
            )
        }

        form_data = {
            "target_column": target_column,
            "feature_columns": ",".join(
                feature_columns
            ),
            "n_estimators": str(
                int(n_estimators)
            ),
            "max_depth": max_depth,
            "random_state": str(
                int(random_state)
            ),
        }

        try:
            with st.spinner(
                "Entraînement du Random Forest en cours..."
            ):
                response = requests.post(
                    f"{API_URL}/ml/train",
                    files=files,
                    data=form_data,
                    timeout=TRAINING_TIMEOUT,
                )

            if response.status_code == 200:
                result = response.json()

                get_ml_status.clear()

                st.success(
                    "Entraînement Machine Learning terminé "
                    "avec succès."
                )

                display_training_report(
                    result
                )

            else:
                try:
                    response_data = response.json()

                    error_detail = response_data.get(
                        "detail",
                        response_data,
                    )

                except ValueError:
                    error_detail = response.text

                st.error(
                    "Une erreur est survenue pendant "
                    "l'entraînement Machine Learning."
                )

                if isinstance(
                    error_detail,
                    (dict, list),
                ):
                    st.json(
                        error_detail
                    )
                else:
                    st.code(
                        str(error_detail)
                    )

        except requests.Timeout:
            st.error(
                "Le délai maximal d'attente a été dépassé."
            )

        except requests.ConnectionError:
            st.error(
                "Impossible de contacter l'API PRUDENCIA."
            )

        except requests.RequestException as error:
            st.error(
                "Une erreur réseau est survenue."
            )

            st.code(
                str(error)
            )
       