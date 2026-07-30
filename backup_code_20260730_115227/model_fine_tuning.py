import os
from io import BytesIO
from typing import Any

import pandas as pd
import requests
import streamlit as st


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

API_URL = os.getenv(
    "PRUDENCIA_API_URL",
    os.getenv("API_URL", "http://api:8000"),
)

TRAINING_TIMEOUT = 3600


# -------------------------------------------------------------------
# Fonctions
# -------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_available_models() -> list[dict[str, str]]:
    """
    Récupère les modèles disponibles depuis FastAPI.
    """

    try:
        response = requests.get(
            f"{API_URL}/fine-tuning/models",
            timeout=15,
        )
        response.raise_for_status()

        models = response.json().get("models", [])

        if not models:
            raise ValueError(
                "Aucun modèle de Fine-Tuning n'est disponible."
            )

        return models

    except requests.RequestException as exc:
        st.error(
            "Impossible de récupérer la liste des modèles depuis l'API."
        )
        st.code(str(exc))
        return []

    except ValueError as exc:
        st.error(str(exc))
        return []


def read_csv_file(uploaded_file: Any) -> pd.DataFrame | None:
    """
    Lit le fichier CSV sélectionné.

    Plusieurs séparateurs et encodages courants sont testés.
    """

    file_content = uploaded_file.getvalue()

    attempts = [
        {
            "sep": ";",
            "encoding": "utf-8",
        },
        {
            "sep": ",",
            "encoding": "utf-8",
        },
        {
            "sep": "\t",
            "encoding": "utf-8",
        },
        {
            "sep": ";",
            "encoding": "latin-1",
        },
        {
            "sep": ",",
            "encoding": "latin-1",
        },
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
        "Vérifiez son séparateur, son encodage "
        "et la présence d'au moins deux colonnes."
    )

    return None


def find_default_column(
    columns: list[str],
    expected_names: list[str],
    fallback_index: int = 0,
) -> int:
    """
    Recherche une colonne par défaut dans le Dataset.
    """

    normalized_columns = {
        column.strip().lower(): index
        for index, column in enumerate(columns)
    }

    for expected_name in expected_names:
        if expected_name in normalized_columns:
            return normalized_columns[expected_name]

    if not columns:
        return 0

    return min(fallback_index, len(columns) - 1)


def extract_metrics(training_result: dict[str, Any]) -> dict[str, Any]:
    """
    Récupère les métriques, quelle que soit leur position
    dans la réponse du Trainer.
    """

    metrics = training_result.get("metrics")

    if isinstance(metrics, dict):
        return metrics

    evaluation = training_result.get("evaluation")

    if isinstance(evaluation, dict):
        return evaluation

    return training_result


def find_metric(
    metrics: dict[str, Any],
    metric_name: str,
) -> float | None:
    """
    Recherche une métrique avec ou sans préfixe 'eval_'.
    """

    possible_names = [
        metric_name,
        f"eval_{metric_name}",
    ]

    for name in possible_names:
        value = metrics.get(name)

        if isinstance(value, (int, float)):
            return float(value)

    return None


def display_percentage_metric(
    container: Any,
    label: str,
    value: float | None,
) -> None:
    """
    Affiche une métrique sous forme de pourcentage.
    """

    if value is None:
        container.metric(label, "—")
        return

    container.metric(label, f"{value:.2%}")


def find_value(
    sources: list[dict[str, Any]],
    possible_keys: list[str],
    default: Any = "—",
) -> Any:
    """
    Recherche une valeur dans plusieurs dictionnaires.
    """

    for source in sources:
        if not isinstance(source, dict):
            continue

        for key in possible_keys:
            value = source.get(key)

            if value is not None:
                return value

    return default


def display_messages(
    title: str,
    messages: Any,
    message_type: str,
) -> None:
    """
    Affiche une liste de recommandations ou d'avertissements.
    """

    if not messages:
        return

    st.subheader(title)

    if isinstance(messages, str):
        messages = [messages]

    if not isinstance(messages, list):
        messages = [str(messages)]

    for message in messages:
        if message_type == "warning":
            st.warning(str(message))
        else:
            st.info(str(message))


def display_training_report(
    result: dict[str, Any],
) -> None:
    """
    Affiche le rapport retourné par FastAPI.
    """

    preparation = result.get("preparation", {})
    quality_report = result.get("quality_report", {})
    training = result.get("training", {})

    metrics = extract_metrics(training)

    st.divider()
    st.header("📊 Training Report")

    st.subheader("Training Metrics")

    metric_columns = st.columns(4)

    display_percentage_metric(
        metric_columns[0],
        "Accuracy",
        find_metric(metrics, "accuracy"),
    )

    display_percentage_metric(
        metric_columns[1],
        "Precision",
        find_metric(metrics, "precision"),
    )

    display_percentage_metric(
        metric_columns[2],
        "Recall",
        find_metric(metrics, "recall"),
    )

    display_percentage_metric(
        metric_columns[3],
        "F1 Score",
        find_metric(metrics, "f1"),
    )

    st.subheader("Dataset Quality")

    dataset_columns = st.columns(3)

    total_rows = find_value(
        [preparation, quality_report],
        [
            "row_count",
            "total_rows",
            "rows",
            "number_of_rows",
        ],
    )

    class_count = find_value(
        [preparation, quality_report],
        [
            "class_count",
            "number_of_classes",
            "classes_count",
            "num_classes",
        ],
    )

    dataset_score = find_value(
        [preparation, quality_report],
        [
            "dataset_score",
            "score",
            "quality_score",
        ],
    )

    dataset_columns[0].metric(
        "Rows",
        total_rows,
    )

    dataset_columns[1].metric(
        "Classes",
        class_count,
    )

    dataset_columns[2].metric(
        "Dataset Score",
        dataset_score,
    )

    training_time = find_value(
        [training, metrics],
        [
            "training_time",
            "train_runtime",
            "runtime",
        ],
        default=None,
    )

    model_path = find_value(
        [training],
        [
            "model_path",
            "saved_model_path",
            "output_dir",
            "save_path",
        ],
        default=None,
    )

    information_columns = st.columns(2)

    if isinstance(training_time, (int, float)):
        information_columns[0].metric(
            "Training Time",
            f"{training_time:.2f} secondes",
        )
    else:
        information_columns[0].metric(
            "Training Time",
            training_time or "—",
        )

    information_columns[1].metric(
        "Saved Model",
        model_path or "—",
    )

    warnings = find_value(
        [quality_report, preparation],
        ["warnings"],
        default=[],
    )

    recommendations = find_value(
        [quality_report, preparation],
        ["recommendations"],
        default=[],
    )

    display_messages(
        "⚠️ Warnings",
        warnings,
        "warning",
    )

    display_messages(
        "💡 Recommendations",
        recommendations,
        "information",
    )

    if not warnings and not recommendations:
        st.success(
            "Le Dataset ne présente aucune anomalie importante détectée."
        )

    with st.expander("Afficher le résultat technique complet"):
        st.json(result)


def render_fine_tuning() -> None:
    """Affiche l’interface complète de Fine-Tuning."""

    st.subheader("Fine-Tuning")

    st.info(
        """
        Cette page permet de spécialiser le modèle CamemBERT à partir
        de vos propres exemples annotés.

        Le modèle apprend à reconnaître les catégories présentes dans
        le Dataset afin d'améliorer les analyses réalisées par PRUDENCIA.
        """
    )

    st.header("📄 Dataset")

    uploaded_file = st.file_uploader(
        "Dataset (.csv)",
        type=["csv"],
        help=(
            "Sélectionnez le fichier CSV contenant les exemples "
            "qui serviront à entraîner le modèle."
        ),
    )

    if uploaded_file is None:
        st.info(
            "Sélectionnez un Dataset CSV pour afficher sa configuration."
        )
        return


    dataframe = read_csv_file(uploaded_file)


    if dataframe is None:
        return

    if dataframe.empty:
        st.error("Le Dataset sélectionné est vide.")
        return

    columns = dataframe.columns.astype(str).tolist()

    if len(columns) < 2:
        st.error(
            "Le Dataset doit contenir au minimum une Text Column "
            "et une Label Column."
        )
        return


    st.success(
        f"Dataset chargé : {len(dataframe)} lignes et "
        f"{len(columns)} colonnes."
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


    st.header("🤖 Model")

    available_models = get_available_models()

    if not available_models:
        return

    model_ids = [
        model["id"]
        for model in available_models
    ]

    model_labels = {
        model["id"]: model.get("name", model["id"])
        for model in available_models
    }

    default_model_index = (
        model_ids.index("juribert")
        if "juribert" in model_ids
        else 0
    )

    model_name = st.selectbox(
        "Model",
        options=model_ids,
        index=default_model_index,
        format_func=lambda model_id: model_labels.get(
            model_id,
            model_id,
        ),
        help=(
            "JuriBERT est sélectionné par défaut car il est "
            "spécialisé dans les textes juridiques français."
        ),
    )
    
    st.header("📝 Dataset Configuration")

    configuration_columns = st.columns(2)

    default_text_index = find_default_column(
        columns,
        [
            "text",
            "texte",
            "content",
            "description",
            "question",
        ],
    )

    text_column = configuration_columns[0].selectbox(
        "Text Column",
        options=columns,
        index=default_text_index,
        help=(
            "Sélectionnez la colonne contenant les textes "
            "que le modèle doit analyser."
        ),
    )

    default_label_index = find_default_column(
        columns,
        [
            "label",
            "labels",
            "classe",
            "class",
            "category",
            "categorie",
            "target",
        ],
        fallback_index=1,
    )

    label_column = configuration_columns[1].selectbox(
        "Label Column",
        options=columns,
        index=default_label_index,
        help=(
            "Sélectionnez la colonne contenant la catégorie attendue "
            "pour chaque texte."
        ),
    )

    if text_column == label_column:
        st.warning(
            "La Text Column et la Label Column doivent être différentes."
        )


    st.header("⚙️ Training Parameters")

    parameter_columns = st.columns(3)

    epochs = parameter_columns[0].number_input(
        "Epochs",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
        help=(
            "Nombre de fois où le modèle parcourt l'ensemble du Dataset. "
            "La valeur 3 est recommandée pour commencer."
        ),
    )

    batch_size = parameter_columns[1].selectbox(
        "Batch Size",
        options=[2, 4, 8, 16],
        index=2,
        help=(
            "Nombre d'exemples traités simultanément. "
            "Une valeur élevée consomme davantage de mémoire. "
            "La valeur 8 est recommandée pour commencer."
        ),
    )

    learning_rate = parameter_columns[2].number_input(
        "Learning Rate",
        min_value=0.000001,
        max_value=0.001,
        value=0.00002,
        step=0.000001,
        format="%.6f",
        help=(
            "Détermine la vitesse à laquelle le modèle apprend. "
            "Conservez la valeur par défaut sauf pour des essais avancés."
        ),
    )


    st.divider()

    st.caption(
        "Le Fine-Tuning peut prendre plusieurs minutes selon la taille "
        "du Dataset et les performances du serveur."
    )

    col_train, col_reset = st.columns(2)

    with col_train:
        launch_training = st.button(
            "🚀 Launch Fine-Tuning",
            type="primary",
            use_container_width=True,
            disabled=text_column == label_column,
        )

    with col_reset:
        reset_model = st.button(
            "🔄 Réinitialiser le modèle",
            use_container_width=True,
        )

    if reset_model:

        try:

            with st.spinner("Réinitialisation du modèle..."):

                response = requests.post(
                    f"{API_URL}/fine-tuning/reset/{model_name}",
                    timeout=60,
                )

            response.raise_for_status()

            st.success(
                "Le modèle a été réinitialisé avec succès."
            )

            st.rerun()

        except requests.RequestException as exc:

            st.error(
                "Impossible de réinitialiser le modèle."
            )

            st.code(str(exc))

    # -------------------------------------------------------------------
    # Appel à FastAPI
    # -------------------------------------------------------------------

    if launch_training:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "text/csv",
            )
        }

        form_data = {
            "model_name": model_name,
            "text_column": text_column,
            "label_column": label_column,
            "epochs": str(int(epochs)),
            "batch_size": str(int(batch_size)),
            "learning_rate": str(float(learning_rate)),
        }

        try:
            with st.spinner(
                "Fine-Tuning en cours... "
                "Merci de ne pas fermer cette page."
            ):
                response = requests.post(
                    f"{API_URL}/fine-tuning/train",
                    files=files,
                    data=form_data,
                    timeout=TRAINING_TIMEOUT,
                )

            if response.status_code == 200:
                result = response.json()

                st.success(
                    "Fine-Tuning terminé avec succès."
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
                    "Une erreur est survenue pendant le Fine-Tuning."
                )

                if isinstance(error_detail, (dict, list)):
                    st.json(error_detail)
                else:
                    st.code(str(error_detail))

        except requests.Timeout:
            st.error(
                "Le délai maximal d'attente a été dépassé. "
                "L'entraînement est peut-être encore en cours sur le serveur."
            )

        except requests.ConnectionError:
            st.error(
                "Impossible de contacter l'API PRUDENCIA. "
                "Vérifiez que le conteneur API est démarré."
            )

        except requests.RequestException as exc:
            st.error(
                "Une erreur réseau est survenue pendant "
                "la communication avec l'API."
            )
            st.code(str(exc))

        except ValueError as exc:
            st.error(
                "La réponse reçue depuis l'API est invalide."
            )
            st.code(str(exc))
