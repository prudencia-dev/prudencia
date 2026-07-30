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


# -------------------------------------------------------------------
# Communication avec l'API
# -------------------------------------------------------------------

@st.cache_data(ttl=30)
def get_ml_status() -> dict[str, Any]:
    """Récupère l'état actuel du modèle Machine Learning."""

    response = requests.get(
        f"{API_URL}/ml/health",
        timeout=5,
    )
    response.raise_for_status()
    return response.json()


# -------------------------------------------------------------------
# Lecture et préparation du CSV
# -------------------------------------------------------------------

def read_csv_file(
    uploaded_file: Any,
) -> pd.DataFrame | None:
    """
    Lit un CSV en testant plusieurs encodages et séparateurs.
    """

    file_content = uploaded_file.getvalue()

    attempts = [
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8"},
        {"sep": "\t", "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "utf-8-sig"},
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
    """Recherche automatiquement une Target probable."""

    expected_names = [
        "risk_level_aiact",
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

    Les identifiants, textes libres et scores calculés sont exclus
    par défaut pour limiter le bruit et les fuites de données.
    """

    excluded_columns = {
        target_column.strip().lower(),
        "id",
        "id_cas",
        "titre_cas",
        "description",
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


# -------------------------------------------------------------------
# Formatage
# -------------------------------------------------------------------

def display_percentage_metric(
    container: Any,
    label: str,
    value: Any,
) -> None:
    """Affiche une valeur comprise entre 0 et 1 en pourcentage."""

    if isinstance(value, (int, float)):
        container.metric(label, f"{float(value):.2%}")
    else:
        container.metric(label, "—")


def display_decimal_metric(
    container: Any,
    label: str,
    value: Any,
) -> None:
    """Affiche une métrique décimale."""

    if isinstance(value, (int, float)):
        container.metric(label, f"{float(value):.4f}")
    else:
        container.metric(label, "—")


# -------------------------------------------------------------------
# Rapport d'entraînement
# -------------------------------------------------------------------

def display_hyperparameters(
    result: dict[str, Any],
) -> None:
    """Affiche la configuration exacte du run."""

    hyperparameters = result.get("hyperparameters", {})

    if not isinstance(hyperparameters, dict):
        return

    st.subheader("⚙️ Configuration expérimentale")

    row_1 = st.columns(5)
    row_1[0].metric(
        "Arbres",
        hyperparameters.get("n_estimators", "—"),
    )
    row_1[1].metric(
        "Profondeur",
        (
            hyperparameters.get("max_depth")
            if hyperparameters.get("max_depth") is not None
            else "Automatique"
        ),
    )
    row_1[2].metric(
        "Min. division",
        hyperparameters.get("min_samples_split", "—"),
    )
    row_1[3].metric(
        "Min. feuille",
        hyperparameters.get("min_samples_leaf", "—"),
    )
    row_1[4].metric(
        "Features / division",
        (
            hyperparameters.get("max_features")
            if hyperparameters.get("max_features") is not None
            else "Toutes"
        ),
    )

    row_2 = st.columns(5)
    row_2[0].metric(
        "Critère",
        hyperparameters.get("criterion", "—"),
    )
    row_2[1].metric(
        "Bootstrap",
        (
            "Oui"
            if hyperparameters.get("bootstrap")
            else "Non"
        ),
    )
    row_2[2].metric(
        "Poids des classes",
        hyperparameters.get("class_weight") or "Aucun",
    )
    row_2[3].metric(
        "Random State",
        hyperparameters.get("random_state", "—"),
    )
    row_2[4].metric(
        "Jeu Test",
        (
            f"{float(hyperparameters.get('test_size')):.0%}"
            if isinstance(
                hyperparameters.get("test_size"),
                (int, float),
            )
            else "—"
        ),
    )


def display_global_metrics(
    metrics: dict[str, Any],
) -> None:
    """Affiche les métriques globales du modèle."""

    st.subheader("📈 Métriques globales")

    row_1 = st.columns(4)
    display_percentage_metric(
        row_1[0],
        "Accuracy",
        metrics.get("accuracy"),
    )
    display_percentage_metric(
        row_1[1],
        "Balanced Accuracy",
        metrics.get("balanced_accuracy"),
    )
    display_percentage_metric(
        row_1[2],
        "Precision pondérée",
        metrics.get(
            "precision_weighted",
            metrics.get("precision"),
        ),
    )
    display_percentage_metric(
        row_1[3],
        "Recall pondéré",
        metrics.get(
            "recall_weighted",
            metrics.get("recall"),
        ),
    )

    row_2 = st.columns(3)
    display_percentage_metric(
        row_2[0],
        "F1 pondéré",
        metrics.get(
            "f1_weighted",
            metrics.get("f1"),
        ),
    )
    display_percentage_metric(
        row_2[1],
        "Macro-F1",
        metrics.get("macro_f1"),
    )
    display_decimal_metric(
        row_2[2],
        "MCC",
        metrics.get("mcc"),
    )

    st.caption(
        "Le Macro-F1 traite toutes les classes de manière égale. "
        "La Balanced Accuracy et le MCC sont utiles lorsque les "
        "classes sont déséquilibrées."
    )


def display_per_class_metrics(
    metrics: dict[str, Any],
) -> None:
    """Affiche Precision, Recall, F1 et Support pour chaque classe."""

    per_class = metrics.get("per_class")

    if not isinstance(per_class, dict) or not per_class:
        return

    st.subheader("🎯 Métriques par classe")

    rows = []

    for class_name, values in per_class.items():
        rows.append(
            {
                "Classe": class_name,
                "Precision": (
                    f"{float(values.get('precision', 0)):.2%}"
                ),
                "Recall": (
                    f"{float(values.get('recall', 0)):.2%}"
                ),
                "F1": (
                    f"{float(values.get('f1', 0)):.2%}"
                ),
                "Support": int(values.get("support", 0)),
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Pour PRUDENCIA, le rappel par classe permet de vérifier "
        "que les catégories critiques ne sont pas négligées."
    )


def display_confusion_matrix(
    metrics: dict[str, Any],
) -> None:
    """Affiche la matrice de confusion du jeu Test."""

    matrix = metrics.get("confusion_matrix")
    class_names = metrics.get("class_names")

    if not matrix or not class_names:
        return

    st.subheader("🧩 Matrice de confusion")

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=[
            f"Réel : {name}"
            for name in class_names
        ],
        columns=[
            f"Prédit : {name}"
            for name in class_names
        ],
    )

    st.dataframe(
        matrix_dataframe,
        use_container_width=True,
    )


def display_feature_importance(
    result: dict[str, Any],
) -> None:
    """Affiche les variables les plus influentes du Random Forest."""

    importance = result.get("feature_importance")

    if not isinstance(importance, list) or not importance:
        return

    st.subheader("🌲 Importance des variables")

    importance_dataframe = pd.DataFrame(importance)

    if not {
        "feature",
        "importance",
    }.issubset(importance_dataframe.columns):
        return

    importance_dataframe = importance_dataframe[
        ["feature", "importance"]
    ].copy()

    importance_dataframe["importance"] = (
        importance_dataframe["importance"]
        .astype(float)
    )

    importance_dataframe = (
        importance_dataframe
        .sort_values(
            by="importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    top_n = min(20, len(importance_dataframe))

    st.dataframe(
        importance_dataframe.head(top_n).rename(
            columns={
                "feature": "Variable encodée",
                "importance": "Importance",
            }
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Importance": st.column_config.ProgressColumn(
                "Importance",
                min_value=0.0,
                max_value=max(
                    0.01,
                    float(
                        importance_dataframe[
                            "importance"
                        ].max()
                    ),
                ),
                format="%.4f",
            ),
        },
    )

    st.caption(
        "Les variables catégorielles sont décomposées par le "
        "OneHotEncoder. Une même Feature d'origine peut donc "
        "apparaître sous plusieurs modalités."
    )


def display_dataset_quality(
    result: dict[str, Any],
) -> None:
    """Affiche les informations de qualité du Dataset."""

    st.subheader("🔎 Qualité du Dataset")

    quality_columns = st.columns(5)

    quality_columns[0].metric(
        "Lignes",
        result.get("dataset_rows", "—"),
    )
    quality_columns[1].metric(
        "Train",
        result.get("train_rows", "—"),
    )
    quality_columns[2].metric(
        "Test",
        result.get("test_rows", "—"),
    )
    quality_columns[3].metric(
        "Doublons",
        result.get("duplicate_rows", 0),
    )
    quality_columns[4].metric(
        "Valeurs manquantes",
        result.get("missing_values", 0),
    )

    missing_ratio = result.get("missing_values_ratio")

    if isinstance(missing_ratio, (int, float)):
        st.caption(
            "Part de valeurs manquantes dans les Features : "
            f"{float(missing_ratio):.2%}."
        )

    removed_columns = result.get(
        "constant_columns_removed",
        [],
    )

    if removed_columns:
        st.warning(
            "Colonnes constantes supprimées automatiquement : "
            + ", ".join(map(str, removed_columns))
        )

    st.write(
        "**Stratification Train/Test :** "
        + (
            "Oui"
            if result.get("stratification_used")
            else "Non"
        )
    )


def display_training_report(
    result: dict[str, Any],
) -> None:
    """Affiche le rapport complet du dernier entraînement."""

    metrics = result.get("metrics", {})

    if not isinstance(metrics, dict):
        metrics = {}

    st.divider()
    st.header("📊 Rapport d'entraînement")

    model_columns = st.columns(4)
    model_columns[0].metric(
        "Modèle",
        result.get("model_name", "Random Forest"),
    )
    model_columns[1].metric(
        "Version",
        result.get("model_version", "—"),
    )
    model_columns[2].metric(
        "Features",
        result.get("feature_count", "—"),
    )

    training_time = result.get("training_time")
    model_columns[3].metric(
        "Durée",
        (
            f"{float(training_time):.3f} s"
            if isinstance(training_time, (int, float))
            else "—"
        ),
    )

    display_hyperparameters(result)
    display_global_metrics(metrics)
    display_per_class_metrics(metrics)
    display_confusion_matrix(metrics)
    display_feature_importance(result)
    display_dataset_quality(result)

    st.subheader("📚 Dataset utilisé")

    st.info(
        f"Target utilisée : "
        f"**{result.get('target_column', '—')}**"
    )

    feature_columns = result.get("feature_columns", [])

    with st.expander(
        "Features utilisées",
        expanded=False,
    ):
        for feature in feature_columns:
            st.write(f"- {feature}")

    target_distribution = result.get(
        "target_distribution",
        {},
    )

    if target_distribution:
        st.write("**Distribution des classes**")

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

    warnings = result.get("warnings", [])

    if warnings:
        st.subheader("⚠️ Avertissements")

        for warning in warnings:
            st.warning(warning)
    else:
        st.success(
            "Aucune anomalie importante détectée."
        )

    with st.expander(
        "Afficher le résultat technique complet",
        expanded=False,
    ):
        st.json(result)


# -------------------------------------------------------------------
# Interface principale
# -------------------------------------------------------------------

def render_machine_learning() -> None:
    """Affiche l'interface complète du moteur Machine Learning."""

    st.subheader("🌲 Machine Learning — Random Forest")

    st.info(
        """
        Cette page entraîne un modèle Random Forest à partir
        d'un Dataset tabulaire.

        Les **Features** sont les informations utilisées pour
        effectuer la prédiction.

        La **Target** ou **Label** est la bonne réponse connue
        pendant l'apprentissage supervisé.
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
        status.get("available", False)
    )

    status_columns = st.columns(3)
    status_columns[0].metric(
        "Modèle",
        status.get("model", "Random Forest"),
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

    if status.get("status") == "unavailable":
        st.warning(
            "L'API Machine Learning est actuellement "
            "indisponible."
        )

    st.divider()

    # ================================================================
    # Import du Dataset
    # ================================================================

    st.header("📄 Dataset")

    uploaded_file = st.file_uploader(
        "Dataset Machine Learning (.csv)",
        type=["csv"],
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

    dataframe = read_csv_file(uploaded_file)

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

    duplicate_rows = int(
        dataframe.duplicated().sum()
    )
    missing_values_total = int(
        dataframe.isna().sum().sum()
    )

    dataset_summary = st.columns(4)
    dataset_summary[0].metric(
        "Lignes",
        len(dataframe),
    )
    dataset_summary[1].metric(
        "Colonnes",
        len(columns),
    )
    dataset_summary[2].metric(
        "Doublons",
        duplicate_rows,
    )
    dataset_summary[3].metric(
        "Valeurs manquantes",
        missing_values_total,
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
    # Target
    # ================================================================

    st.header("🎯 Target — Label")

    st.caption(
        "La Target est la valeur que le modèle doit apprendre "
        "à prédire."
    )

    default_target_index = find_default_target_column(
        columns
    )

    target_column = st.selectbox(
        "Target Column",
        options=columns,
        index=default_target_index,
        help=(
            "Pour PRUDENCIA : par exemple "
            "risk_level_aiact."
        ),
    )

    target_counts = (
        dataframe[target_column]
        .value_counts(dropna=True)
    )

    target_class_count = int(
        target_counts.size
    )

    minimum_class_count = (
        int(target_counts.min())
        if not target_counts.empty
        else 0
    )

    majority_class = (
        str(target_counts.index[0])
        if not target_counts.empty
        else "—"
    )

    target_columns = st.columns(3)
    target_columns[0].metric(
        "Classes",
        target_class_count,
    )
    target_columns[1].metric(
        "Classe majoritaire",
        majority_class,
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
            "Cette colonne contient presque une valeur "
            "différente par ligne. Elle ressemble à un "
            "identifiant ou à une valeur continue."
        )

    if minimum_class_count < 2:
        st.warning(
            "Certaines classes contiennent moins de deux "
            "exemples. La stratification sera impossible."
        )

    # ================================================================
    # Features
    # ================================================================

    st.header("📝 Features")

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
            "Évitez les identifiants, les textes libres "
            "et les colonnes calculées depuis la Target."
        ),
    )

    if not feature_columns:
        st.warning(
            "Sélectionnez au minimum une Feature."
        )

    if feature_columns:
        selected_dataframe = dataframe[
            feature_columns + [target_column]
        ]

        selected_missing_values = int(
            selected_dataframe.isna().sum().sum()
        )
    else:
        selected_missing_values = 0

    ignored_columns = [
        column
        for column in columns
        if (
            column not in feature_columns
            and column != target_column
        )
    ]

    configuration_columns = st.columns(3)
    configuration_columns[0].metric(
        "Features sélectionnées",
        len(feature_columns),
    )
    configuration_columns[1].metric(
        "Valeurs manquantes",
        selected_missing_values,
    )
    configuration_columns[2].metric(
        "Colonnes ignorées",
        len(ignored_columns),
    )

    with st.expander(
        "Résumé de la sélection",
        expanded=False,
    ):
        st.write(f"**Target :** {target_column}")
        st.write("**Features :**")

        for feature in feature_columns:
            st.write(f"- {feature}")

        if ignored_columns:
            st.write("**Colonnes ignorées :**")

            for column in ignored_columns:
                st.write(f"- {column}")

    # ================================================================
    # Hyperparamètres
    # ================================================================

    st.header("⚙️ Hyperparamètres du Random Forest")

    st.caption(
        "Pour comparer deux expériences proprement, "
        "modifiez de préférence un seul paramètre à la fois."
    )

    row_1 = st.columns(3)

    n_estimators = row_1[0].number_input(
        "Nombre d'arbres",
        min_value=10,
        max_value=5000,
        value=300,
        step=50,
        help=(
            "Plus d'arbres améliore généralement la stabilité, "
            "au prix d'un temps de calcul plus long."
        ),
    )

    max_depth_choice = row_1[1].selectbox(
        "Profondeur maximale",
        options=[
            "Automatique",
            "5",
            "10",
            "20",
            "30",
            "50",
        ],
        index=0,
        help=(
            "Une profondeur limitée peut réduire "
            "le surapprentissage."
        ),
    )

    criterion = row_1[2].selectbox(
        "Critère de division",
        options=[
            "gini",
            "entropy",
            "log_loss",
        ],
        index=0,
    )

    row_2 = st.columns(3)

    min_samples_split = row_2[0].number_input(
        "Minimum pour diviser un nœud",
        min_value=2,
        max_value=100,
        value=2,
        step=1,
    )

    min_samples_leaf = row_2[1].number_input(
        "Minimum par feuille",
        min_value=1,
        max_value=100,
        value=1,
        step=1,
    )

    max_features_label = row_2[2].selectbox(
        "Features testées par division",
        options=[
            "sqrt",
            "log2",
            "Toutes",
        ],
        index=0,
    )

    row_3 = st.columns(4)

    bootstrap = row_3[0].toggle(
        "Bootstrap",
        value=True,
        help=(
            "Chaque arbre apprend sur un échantillon "
            "tiré avec remise."
        ),
    )

    class_weight_label = row_3[1].selectbox(
        "Poids des classes",
        options=[
            "Aucun",
            "balanced",
            "balanced_subsample",
        ],
        index=0,
        help=(
            "À tester lorsque les classes sont déséquilibrées. "
            "Ce réglage n'améliore pas automatiquement le modèle."
        ),
    )

    random_state = row_3[2].number_input(
        "Random State",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
        help=(
            "Permet de reproduire le même découpage "
            "et le même entraînement."
        ),
    )

    test_size_percent = row_3[3].select_slider(
        "Part du jeu Test",
        options=[10, 15, 20, 25, 30, 35, 40],
        value=20,
        format_func=lambda value: f"{value} %",
    )

    max_depth = (
        ""
        if max_depth_choice == "Automatique"
        else max_depth_choice
    )

    max_features = (
        "none"
        if max_features_label == "Toutes"
        else max_features_label
    )

    class_weight = (
        "none"
        if class_weight_label == "Aucun"
        else class_weight_label
    )

    test_size = test_size_percent / 100

    with st.expander(
        "Configuration recommandée pour la baseline",
        expanded=False,
    ):
        st.code(
            "\n".join(
                [
                    "n_estimators       = 300",
                    "max_depth          = None",
                    "min_samples_split  = 2",
                    "min_samples_leaf   = 1",
                    "max_features       = sqrt",
                    "criterion          = gini",
                    "bootstrap          = True",
                    "class_weight       = None",
                    "random_state       = 42",
                    "test_size          = 0.20",
                ]
            )
        )

    st.divider()

    # ================================================================
    # Actions
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
            st.code(str(error))

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
            "n_estimators": str(int(n_estimators)),
            "max_depth": max_depth,
            "min_samples_split": str(
                int(min_samples_split)
            ),
            "min_samples_leaf": str(
                int(min_samples_leaf)
            ),
            "max_features": max_features,
            "criterion": criterion,
            "bootstrap": str(bootstrap).lower(),
            "class_weight": class_weight,
            "random_state": str(int(random_state)),
            "test_size": str(float(test_size)),
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

                display_training_report(result)

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
                    st.json(error_detail)
                else:
                    st.code(str(error_detail))

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
            st.code(str(error))