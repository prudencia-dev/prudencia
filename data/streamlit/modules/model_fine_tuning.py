from __future__ import annotations

import os
from io import BytesIO
from typing import Any

import pandas as pd
import requests
import streamlit as st


# -------------------------------------------------------------------
# Configuration de l'accès à FastAPI
# -------------------------------------------------------------------

API_URL = os.getenv(
    "PRUDENCIA_API_URL",
    os.getenv("API_URL", "http://api:8000"),
)

# Un Fine-Tuning peut durer plusieurs minutes. Le timeout est volontairement
# élevé afin que Streamlit n'interrompe pas la requête pendant l'entraînement.
TRAINING_TIMEOUT = 3600


# -------------------------------------------------------------------
# Utilitaires de lecture et d'affichage
# -------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_available_models() -> list[dict[str, str]]:
    """Récupère les modèles Fine-Tuning déclarés dans FastAPI."""

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
    """Lit un CSV en testant plusieurs séparateurs et encodages."""

    file_content = uploaded_file.getvalue()
    attempts = [
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8"},
        {"sep": "\t", "encoding": "utf-8"},
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
        "Le fichier CSV n'a pas pu être lu. Vérifiez son séparateur, "
        "son encodage et la présence d'au moins deux colonnes."
    )
    return None


def find_default_column(
    columns: list[str],
    expected_names: list[str],
    fallback_index: int = 0,
) -> int:
    """Cherche automatiquement les colonnes texte et label habituelles."""

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
    """Récupère les métriques quelle que soit leur position dans la réponse."""

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
    """Recherche une métrique avec ou sans le préfixe Hugging Face ``eval_``."""

    for name in (metric_name, f"eval_{metric_name}"):
        value = metrics.get(name)

        if isinstance(value, (int, float)):
            return float(value)

    return None


def display_percentage_metric(
    container: Any,
    label: str,
    value: float | None,
) -> None:
    """Affiche une valeur comprise entre 0 et 1 sous forme de pourcentage."""

    container.metric(
        label,
        "—" if value is None else f"{value:.2%}",
    )


def find_value(
    sources: list[dict[str, Any]],
    possible_keys: list[str],
    default: Any = "—",
) -> Any:
    """Cherche une valeur dans plusieurs dictionnaires possibles."""

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
    """Affiche les avertissements et recommandations du contrôle qualité."""

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


def _display_parameter_grid(parameters: dict[str, Any]) -> None:
    """Présente les hyperparamètres du run dans un format lisible."""

    st.subheader("⚙️ Hyperparamètres utilisés")

    row_1 = st.columns(5)
    row_1[0].metric("Epochs demandés", parameters.get("epochs", "—"))
    row_1[1].metric("Batch réel", parameters.get("batch_size", "—"))
    row_1[2].metric(
        "Accumulation",
        parameters.get("gradient_accumulation_steps", "—"),
    )
    row_1[3].metric(
        "Batch effectif",
        parameters.get("effective_batch_size", "—"),
    )
    row_1[4].metric("Seed", parameters.get("seed", "—"))

    row_2 = st.columns(5)
    row_2[0].metric(
        "Learning rate",
        parameters.get("learning_rate", "—"),
    )
    row_2[1].metric(
        "Max tokens",
        parameters.get("max_length", "—"),
    )
    row_2[2].metric(
        "Early stopping",
        parameters.get("early_stopping_patience", "—"),
    )
    row_2[3].metric(
        "Weight decay",
        parameters.get("weight_decay", "—"),
    )
    row_2[4].metric(
        "Warmup ratio",
        parameters.get("warmup_ratio", "—"),
    )

    st.write(
        "**Métrique de sélection :** "
        f"`{parameters.get('metric_for_best_model', '—')}`"
    )
    st.write(
        "**Poids de classes :** "
        f"{'activés' if parameters.get('use_class_weights') else 'désactivés'}"
    )


def _display_per_class_metrics(metrics: dict[str, Any]) -> None:
    """Affiche precision, recall et F1 pour chaque catégorie AI Act."""

    per_class = metrics.get("per_class")

    if not isinstance(per_class, dict) or not per_class:
        return

    st.subheader("🎯 Résultats par classe")

    rows = []

    for class_name, values in per_class.items():
        rows.append(
            {
                "Classe": class_name,
                "Precision": values.get("precision"),
                "Recall": values.get("recall"),
                "F1": values.get("f1"),
                "Support": values.get("support"),
            }
        )

    dataframe = pd.DataFrame(rows)

    for column in ("Precision", "Recall", "F1"):
        dataframe[column] = dataframe[column].map(
            lambda value: (
                f"{float(value):.2%}"
                if isinstance(value, (int, float))
                else "—"
            )
        )

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )


def _display_confusion_matrix(metrics: dict[str, Any]) -> None:
    """Affiche la matrice de confusion sous forme de tableau."""

    matrix = metrics.get("confusion_matrix")
    class_names = metrics.get("class_names")

    if not matrix or not class_names:
        return

    st.subheader("🧩 Matrice de confusion")
    st.caption(
        "Les lignes correspondent aux classes réelles et les colonnes aux "
        "classes prédites."
    )

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=[f"Réel : {name}" for name in class_names],
        columns=[f"Prédit : {name}" for name in class_names],
    )

    st.dataframe(
        matrix_dataframe,
        use_container_width=True,
    )


def display_training_report(result: dict[str, Any]) -> None:
    """
    Affiche un rapport d'entraînement complet et directement exploitable.

    Le rapport distingue :
    - qualité du dataset ;
    - configuration expérimentale ;
    - métriques globales ;
    - performances par classe ;
    - état réel de l'entraînement et longueur des textes.
    """

    preparation = result.get("preparation", {})
    quality_report = result.get("quality_report", {})
    training = result.get("training", {})
    metrics = extract_metrics(training)
    parameters = training.get("parameters", {})
    training_state = training.get("training_state", {})
    token_statistics = training.get(
        "token_length_statistics",
        {},
    )

    st.divider()
    st.header("📊 Rapport d'entraînement Deep Learning")

    base_model = training.get("base_model", {})
    model_columns = st.columns(4)
    model_columns[0].metric(
        "Modèle",
        base_model.get("display_name", training.get("model_name", "—")),
    )
    model_columns[1].metric(
        "Train",
        training.get("training_examples", "—"),
    )
    model_columns[2].metric(
        "Validation",
        training.get("validation_examples", "—"),
    )
    model_columns[3].metric(
        "Durée",
        (
            f"{float(training.get('training_time')):.2f} s"
            if isinstance(training.get("training_time"), (int, float))
            else "—"
        ),
    )

    st.write(
        "**Checkpoint Hugging Face :** "
        f"`{base_model.get('huggingface_id', '—')}`"
    )
    st.write(
        "**Architecture :** "
        f"`{base_model.get('architecture', '—')}`"
    )

    st.subheader("📈 Métriques globales")
    global_metrics = st.columns(6)

    display_percentage_metric(
        global_metrics[0],
        "Accuracy",
        find_metric(metrics, "accuracy"),
    )
    display_percentage_metric(
        global_metrics[1],
        "Precision pondérée",
        find_metric(metrics, "precision_weighted"),
    )
    display_percentage_metric(
        global_metrics[2],
        "Recall pondéré",
        find_metric(metrics, "recall_weighted"),
    )
    display_percentage_metric(
        global_metrics[3],
        "F1 pondéré",
        find_metric(metrics, "f1_weighted"),
    )
    display_percentage_metric(
        global_metrics[4],
        "Macro-F1",
        find_metric(metrics, "macro_f1"),
    )

    loss = find_metric(metrics, "loss")
    global_metrics[5].metric(
        "Loss",
        "—" if loss is None else f"{loss:.4f}",
    )

    _display_parameter_grid(parameters)

    st.subheader("⏱️ État de l'entraînement")
    state_columns = st.columns(4)
    state_columns[0].metric(
        "Epochs demandés",
        training_state.get("epochs_requested", "—"),
    )
    state_columns[1].metric(
        "Epochs réalisés",
        training_state.get("epochs_completed", "—"),
    )
    state_columns[2].metric(
        "Meilleure métrique",
        (
            f"{float(training_state.get('best_metric')):.4f}"
            if isinstance(training_state.get("best_metric"), (int, float))
            else "—"
        ),
    )
    state_columns[3].metric(
        "Early stopping",
        (
            "Actif"
            if training_state.get("early_stopping_enabled")
            else "Inactif"
        ),
    )

    _display_per_class_metrics(metrics)
    _display_confusion_matrix(metrics)

    st.subheader("📏 Longueur des textes")

    if token_statistics:
        token_columns = st.columns(5)
        token_columns[0].metric(
            "Médiane",
            token_statistics.get("median", "—"),
        )
        token_columns[1].metric(
            "P95",
            token_statistics.get("p95", "—"),
        )
        token_columns[2].metric(
            "Maximum",
            token_statistics.get("maximum", "—"),
        )
        token_columns[3].metric(
            "Max utilisé",
            token_statistics.get("max_length_used", "—"),
        )
        token_columns[4].metric(
            "Textes tronqués",
            (
                f"{token_statistics.get('texts_truncated', 0)} / "
                f"{token_statistics.get('total_texts', 0)}"
            ),
        )
    else:
        st.info("Les statistiques de longueur ne sont pas disponibles.")

    st.subheader("🧪 Qualité du dataset")
    dataset_columns = st.columns(3)
    dataset_columns[0].metric(
        "Lignes",
        find_value(
            [preparation, quality_report],
            ["total_examples", "row_count", "total_rows", "rows"],
        ),
    )
    dataset_columns[1].metric(
        "Classes",
        find_value(
            [preparation, quality_report],
            ["number_of_classes", "class_count", "num_classes"],
        ),
    )
    dataset_columns[2].metric(
        "Score qualité",
        find_value(
            [preparation, quality_report],
            ["quality_score", "dataset_score", "score"],
        ),
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

    display_messages("⚠️ Avertissements", warnings, "warning")
    display_messages(
        "💡 Recommandations",
        recommendations,
        "information",
    )

    if not warnings and not recommendations:
        st.success(
            "Le dataset ne présente aucune anomalie importante détectée."
        )

    st.write(
        "**Modèle sauvegardé :** "
        f"`{training.get('model_path', '—')}`"
    )

    with st.expander("Afficher le résultat technique complet"):
        st.json(result)


# -------------------------------------------------------------------
# Interface principale
# -------------------------------------------------------------------

def render_fine_tuning() -> None:
    """Affiche l'interface complète de Fine-Tuning."""

    st.subheader("Fine-Tuning Deep Learning")
    st.info(
        "Cette page entraîne CamemBERT ou JuriBERT sur un dataset annoté. "
        "Tous les paramètres sont transmis à FastAPI, enregistrés dans "
        "l'historique et repris dans le rapport du run."
    )

    st.header("📄 Dataset")
    uploaded_file = st.file_uploader(
        "Dataset (.csv)",
        type=["csv"],
        help=(
            "Sélectionnez le CSV contenant une colonne texte et une "
            "colonne label."
        ),
    )

    if uploaded_file is None:
        st.info(
            "Sélectionnez un dataset CSV pour afficher sa configuration."
        )
        return

    dataframe = read_csv_file(uploaded_file)

    if dataframe is None:
        return

    if dataframe.empty:
        st.error("Le dataset sélectionné est vide.")
        return

    columns = dataframe.columns.astype(str).tolist()

    if len(columns) < 2:
        st.error(
            "Le dataset doit contenir au minimum une colonne texte et une "
            "colonne label."
        )
        return

    st.success(
        f"Dataset chargé : {len(dataframe)} lignes et "
        f"{len(columns)} colonnes."
    )

    with st.expander("Prévisualiser le dataset", expanded=True):
        st.dataframe(
            dataframe.head(20),
            use_container_width=True,
            hide_index=True,
        )

    st.header("🤖 Modèle")
    available_models = get_available_models()

    if not available_models:
        return

    model_ids = [model["id"] for model in available_models]
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
        "Modèle de base",
        options=model_ids,
        index=default_model_index,
        format_func=lambda model_id: model_labels.get(
            model_id,
            model_id,
        ),
        help=(
            "JuriBERT est conseillé pour les textes juridiques français."
        ),
    )

    st.header("📝 Configuration du dataset")
    configuration_columns = st.columns(2)

    text_column = configuration_columns[0].selectbox(
        "Colonne texte",
        options=columns,
        index=find_default_column(
            columns,
            ["text", "texte", "content", "description", "question"],
        ),
    )
    label_column = configuration_columns[1].selectbox(
        "Colonne label",
        options=columns,
        index=find_default_column(
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
        ),
    )

    if text_column == label_column:
        st.warning(
            "La colonne texte et la colonne label doivent être différentes."
        )

    st.header("⚙️ Hyperparamètres")

    basic_columns = st.columns(3)
    epochs = basic_columns[0].number_input(
        "Epochs maximum",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help=(
            "Nombre maximal de passages complets sur le dataset. "
            "L'early stopping peut arrêter le run avant."
        ),
    )
    batch_size = basic_columns[1].selectbox(
        "Batch réel",
        options=[1, 2, 4, 8, 16],
        index=2,
        help=(
            "Nombre d'exemples chargés simultanément. Une valeur élevée "
            "consomme davantage de RAM ou de mémoire GPU."
        ),
    )
    learning_rate = basic_columns[2].number_input(
        "Learning rate",
        min_value=0.000001,
        max_value=0.001,
        value=0.00002,
        step=0.000001,
        format="%.6f",
        help=(
            "Taille des corrections appliquées au modèle. 2e-5 est une "
            "bonne valeur initiale pour JuriBERT."
        ),
    )

    resource_columns = st.columns(3)
    max_length = resource_columns[0].selectbox(
        "Longueur maximale (tokens)",
        options=[128, 256, 320, 384, 512],
        index=1,
        help=(
            "Les textes plus longs sont tronqués. Une valeur plus faible "
            "réduit fortement la charge du serveur."
        ),
    )
    gradient_accumulation_steps = resource_columns[1].selectbox(
        "Accumulation de gradients",
        options=[1, 2, 4, 8, 16],
        index=1,
        help=(
            "Simule un batch plus grand sans charger tous les exemples en "
            "mémoire au même moment."
        ),
    )
    seed = resource_columns[2].number_input(
        "Graine aléatoire",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
        help=(
            "Fixe les opérations aléatoires afin de rendre les runs "
            "comparables et reproductibles."
        ),
    )

    effective_batch_size = (
        int(batch_size) * int(gradient_accumulation_steps)
    )
    st.caption(
        f"Batch effectif : {batch_size} × "
        f"{gradient_accumulation_steps} = {effective_batch_size}."
    )

    regularization_columns = st.columns(3)
    early_stopping_patience = regularization_columns[0].number_input(
        "Patience early stopping",
        min_value=0,
        max_value=20,
        value=3,
        step=1,
        help=(
            "Nombre d'epochs sans amélioration avant arrêt automatique. "
            "0 désactive l'early stopping."
        ),
    )
    weight_decay = regularization_columns[1].number_input(
        "Weight decay",
        min_value=0.0,
        max_value=1.0,
        value=0.01,
        step=0.01,
        format="%.3f",
        help="Régularisation qui limite le surapprentissage.",
    )
    warmup_ratio = regularization_columns[2].number_input(
        "Warmup ratio",
        min_value=0.0,
        max_value=0.5,
        value=0.10,
        step=0.05,
        format="%.2f",
        help=(
            "Fait monter progressivement le learning rate au début du run."
        ),
    )

    selection_columns = st.columns(2)
    metric_for_best_model = selection_columns[0].selectbox(
        "Métrique du meilleur checkpoint",
        options=["macro_f1", "f1_weighted", "accuracy", "loss"],
        index=0,
        help=(
            "Macro-F1 donne le même poids aux cinq classes et constitue "
            "le meilleur choix pour PRUDENCIA."
        ),
    )
    use_class_weights = selection_columns[1].checkbox(
        "Activer les poids de classes",
        value=True,
        help=(
            "Augmente le coût des erreurs sur les classes moins fréquentes."
        ),
    )

    st.divider()
    st.caption(
        "Configuration conseillée pour la démonstration : JuriBERT, 10 epochs, "
        "batch 4, accumulation 2, 2e-5, seed 42, 256 tokens, patience 3, "
        "macro-F1 et poids de classes activés."
    )

    col_train, col_reset = st.columns(2)
    launch_training = col_train.button(
        "🚀 Lancer le Fine-Tuning",
        type="primary",
        use_container_width=True,
        disabled=text_column == label_column,
    )
    reset_model = col_reset.button(
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
            st.success("Le modèle a été réinitialisé avec succès.")
            st.rerun()

        except requests.RequestException as exc:
            st.error("Impossible de réinitialiser le modèle.")
            st.code(str(exc))

    if not launch_training:
        return

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
        "seed": str(int(seed)),
        "max_length": str(int(max_length)),
        "gradient_accumulation_steps": str(
            int(gradient_accumulation_steps)
        ),
        "early_stopping_patience": str(
            int(early_stopping_patience)
        ),
        "weight_decay": str(float(weight_decay)),
        "warmup_ratio": str(float(warmup_ratio)),
        "metric_for_best_model": metric_for_best_model,
        "use_class_weights": str(bool(use_class_weights)).lower(),
    }

    try:
        with st.spinner(
            "Fine-Tuning en cours... Merci de ne pas fermer cette page."
        ):
            response = requests.post(
                f"{API_URL}/fine-tuning/train",
                files=files,
                data=form_data,
                timeout=TRAINING_TIMEOUT,
            )

        if response.status_code == 200:
            result = response.json()
            st.success("Fine-Tuning terminé avec succès.")
            display_training_report(result)
            return

        try:
            response_data = response.json()
            error_detail = response_data.get("detail", response_data)
        except ValueError:
            error_detail = response.text

        st.error("Une erreur est survenue pendant le Fine-Tuning.")

        if isinstance(error_detail, (dict, list)):
            st.json(error_detail)
        else:
            st.code(str(error_detail))

    except requests.Timeout:
        st.error(
            "Le délai maximal d'attente a été dépassé. L'entraînement est "
            "peut-être encore en cours sur le serveur."
        )

    except requests.ConnectionError:
        st.error(
            "Impossible de contacter l'API PRUDENCIA. Vérifiez que le "
            "conteneur API est démarré."
        )

    except requests.RequestException as exc:
        st.error(
            "Une erreur réseau est survenue pendant la communication avec l'API."
        )
        st.code(str(exc))

    except ValueError as exc:
        st.error("La réponse reçue depuis l'API est invalide.")
        st.code(str(exc))