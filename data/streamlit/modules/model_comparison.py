from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

from services.api import PrudenciaAPI

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

API_URL = os.getenv(
    "PRUDENCIA_API_URL",
    os.getenv("API_URL", "http://api:8000"),
)


# -------------------------------------------------------------------
# Accès API
# -------------------------------------------------------------------

@st.cache_data(ttl=30)
def get_training_history() -> list[dict[str, Any]]:
    """
    Récupère le journal des entraînements depuis FastAPI.
    """

    response = requests.get(
        f"{API_URL}/training/history",
        timeout=10,
    )

    response.raise_for_status()

    payload = response.json()

    # L'API peut retourner directement une liste.
    if isinstance(payload, list):
        return payload

    # Ou placer la liste dans une clé.
    if isinstance(payload, dict):
        for key in (
            "history",
            "executions",
            "items",
            "results",
        ):
            value = payload.get(key)

            if isinstance(value, list):
                return value

    return []


@st.cache_data(ttl=30)
def get_ml_status() -> dict[str, Any]:
    """
    Récupère l'état du moteur Random Forest.
    """

    response = requests.get(
        f"{API_URL}/ml/health",
        timeout=5,
    )

    response.raise_for_status()

    return response.json()


# -------------------------------------------------------------------
# Fonctions utilitaires
# -------------------------------------------------------------------

def normalize_boolean(value: Any) -> bool:
    """
    Convertit différentes représentations en booléen.
    """

    if value is True:
        return True

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "oui",
    }


def get_nested_dictionary(
    execution: dict[str, Any],
    possible_keys: list[str],
) -> dict[str, Any]:
    """
    Recherche un dictionnaire dans plusieurs clés possibles.
    """

    for key in possible_keys:
        value = execution.get(key)

        if isinstance(value, dict):
            return value

    return {}


def get_metric(
    execution: dict[str, Any],
    metric_name: str,
) -> float | None:
    """
    Recherche une métrique dans les différentes zones
    possibles du journal.
    """

    output_data = get_nested_dictionary(
        execution,
        [
            "output_data",
            "output",
            "result",
            "results",
        ],
    )

    possible_sources = [
        execution,
        output_data,
        output_data.get("metrics", {}),
        execution.get("metrics", {}),
    ]

    for source in possible_sources:
        if not isinstance(source, dict):
            continue

        for key in (
            metric_name,
            f"eval_{metric_name}",
            f"test_{metric_name}",
        ):
            value = source.get(key)

            if isinstance(value, (int, float)):
                return float(value)

    return None


def get_execution_time(
    execution: dict[str, Any],
) -> float | None:
    """
    Retourne la durée d'entraînement en secondes.
    """

    for key in (
        "execution_time_ms",
        "duration_ms",
    ):
        value = execution.get(key)

        if isinstance(value, (int, float)):
            return float(value) / 1000

    for key in (
        "training_time",
        "duration",
        "execution_time",
    ):
        value = execution.get(key)

        if isinstance(value, (int, float)):
            return float(value)

    output_data = get_nested_dictionary(
        execution,
        [
            "output_data",
            "output",
        ],
    )

    value = output_data.get("training_time")

    if isinstance(value, (int, float)):
        return float(value)

    return None


def get_dataset_rows(
    execution: dict[str, Any],
) -> int | None:
    """
    Retourne le nombre de lignes du Dataset.
    """

    for key in (
        "dataset_rows",
        "row_count",
        "rows",
    ):
        value = execution.get(key)

        if isinstance(value, int):
            return value

    return None


def is_training_execution(
    execution: dict[str, Any],
) -> bool:
    """
    Ignore les opérations de reset dans la comparaison.
    """

    execution_type = str(
        execution.get(
            "execution_type",
            "training",
        )
    ).lower()

    return execution_type in {
        "training",
        "fine_tuning",
        "train",
    }


def is_machine_learning_execution(
    execution: dict[str, Any],
) -> bool:
    """
    Identifie une exécution Machine Learning.
    """

    model_type = str(
        execution.get(
            "model_type",
            "",
        )
    ).lower()

    model_name = str(
        execution.get(
            "model_name",
            "",
        )
    ).lower()

    return (
        model_type == "machine_learning"
        or "random forest" in model_name
        or "randomforest" in model_name
    )


def is_deep_learning_execution(
    execution: dict[str, Any],
) -> bool:
    """
    Identifie une exécution Fine-Tuning / Deep Learning.
    """

    model_type = str(
        execution.get(
            "model_type",
            "",
        )
    ).lower()

    model_name = str(
        execution.get(
            "model_name",
            "",
        )
    ).lower()

    return (
        model_type in {
            "deep_learning",
            "fine_tuning",
        }
        or "camembert" in model_name
        or "juribert" in model_name
    )


def find_latest_execution(
    history: list[dict[str, Any]],
    execution_filter: Any,
) -> dict[str, Any] | None:
    """
    Recherche le dernier entraînement correspondant au filtre.
    """

    matching_executions = [
        execution
        for execution in history
        if (
            is_training_execution(execution)
            and execution_filter(execution)
        )
    ]

    if not matching_executions:
        return None

    # L'API retourne normalement les plus récentes en premier.
    # Si is_active existe, on privilégie toutefois le modèle actif.
    active_executions = [
        execution
        for execution in matching_executions
        if normalize_boolean(
            execution.get(
                "is_active",
                False,
            )
        )
    ]

    if active_executions:
        return active_executions[0]

    return matching_executions[0]


def format_percentage(
    value: float | None,
) -> str:
    """
    Formate une métrique en pourcentage.
    """

    if value is None:
        return "—"

    return f"{value:.2%}"


def format_duration(
    value: float | None,
) -> str:
    """
    Formate une durée en secondes.
    """

    if value is None:
        return "—"

    if value < 1:
        return f"{value:.3f} s"

    return f"{value:.2f} s"


def display_model_card(
    title: str,
    icon: str,
    execution: dict[str, Any] | None,
    fallback_name: str,
    model_family: str,
    data_type: str,
) -> None:
    """
    Affiche la synthèse d'un moteur IA.
    """

    st.subheader(
        f"{icon} {title}"
    )

    if execution is None:
        st.warning(
            "Aucun entraînement enregistré."
        )

        st.metric(
            "Modèle",
            fallback_name,
        )

        st.metric(
            "Statut",
            "⚪ Non entraîné",
        )

        return

    model_name = execution.get(
        "model_name",
        fallback_name,
    )

    model_version = execution.get(
        "model_version",
        "—",
    )

    is_active = normalize_boolean(
        execution.get(
            "is_active",
            False,
        )
    )

    status_label = (
        "🟢 Actif"
        if is_active
        else "⚪ Historique"
    )

    first_row = st.columns(3)

    first_row[0].metric(
        "Modèle",
        model_name,
    )

    first_row[1].metric(
        "Version",
        model_version,
    )

    first_row[2].metric(
        "Statut",
        status_label,
    )

    second_row = st.columns(3)

    second_row[0].metric(
        "Famille",
        model_family,
    )

    second_row[1].metric(
        "Données",
        data_type,
    )

    second_row[2].metric(
        "Dataset",
        execution.get(
            "dataset_name",
            "—",
        ),
    )


def display_metric_comparison(
    deep_learning: dict[str, Any] | None,
    machine_learning: dict[str, Any] | None,
) -> None:
    """
    Compare les métriques des deux derniers entraînements.
    """

    st.header(
        "📊 Comparaison des performances"
    )

    metric_definitions = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1"),
    ]

    comparison_rows: list[dict[str, Any]] = []

    for label, metric_name in metric_definitions:
        deep_learning_value = (
            get_metric(
                deep_learning,
                metric_name,
            )
            if deep_learning
            else None
        )

        machine_learning_value = (
            get_metric(
                machine_learning,
                metric_name,
            )
            if machine_learning
            else None
        )

        comparison_rows.append(
            {
                "Métrique": label,
                "Deep Learning": format_percentage(
                    deep_learning_value
                ),
                "Machine Learning": format_percentage(
                    machine_learning_value
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(
            comparison_rows
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Les métriques ne sont directement comparables que si les "
        "modèles utilisent un objectif, des classes et des jeux de test "
        "équivalents."
    )


def display_training_information(
    deep_learning: dict[str, Any] | None,
    machine_learning: dict[str, Any] | None,
) -> None:
    """
    Compare les informations techniques des entraînements.
    """

    st.header(
        "⏱️ Entraînement et Dataset"
    )

    rows = [
        {
            "Critère": "Durée",
            "Deep Learning": format_duration(
                get_execution_time(
                    deep_learning
                )
                if deep_learning
                else None
            ),
            "Machine Learning": format_duration(
                get_execution_time(
                    machine_learning
                )
                if machine_learning
                else None
            ),
        },
        {
            "Critère": "Lignes du Dataset",
            "Deep Learning": (
                get_dataset_rows(
                    deep_learning
                )
                if deep_learning
                else "—"
            ),
            "Machine Learning": (
                get_dataset_rows(
                    machine_learning
                )
                if machine_learning
                else "—"
            ),
        },
        {
            "Critère": "Type d'apprentissage",
            "Deep Learning": "Supervisé — Fine-Tuning",
            "Machine Learning": "Supervisé — Classification",
        },
        {
            "Critère": "Prétraitement",
            "Deep Learning": "Tokenisation",
            "Machine Learning": (
                "Imputation + encodage catégoriel"
            ),
        },
    ]

    st.dataframe(
        pd.DataFrame(
            rows
        ),
        use_container_width=True,
        hide_index=True,
    )


# -------------------------------------------------------------------
# Interface principale
# -------------------------------------------------------------------

def render_model_comparison() -> None:
    """
    Affiche la comparaison pédagogique des moteurs IA PRUDENCIA.
    """

    st.subheader(
        "Comparaison des moteurs"
    )

    st.info(
        """
        PRUDENCIA utilise deux approches complémentaires.

        Le Deep Learning analyse principalement le langage et les textes.

        Le Machine Learning exploite les réponses structurées
        du questionnaire pour produire une classification.
        """
    )

    try:
        history = get_training_history()

        best = PrudenciaAPI.get("/fine-tuning/best-model")

        best_model = best.get("best_model")

        if best_model:

            st.success(
                f"🏆 Modèle recommandé : "
                f"{best_model['model_name']}"
            )

            c1, c2, c3 = st.columns(3)

            metrics = (
                best_model.get("output_data", {})
                .get("metrics", {})
            )

            c1.metric(
                "Accuracy",
                format_percentage(
                    metrics.get("accuracy")
                    or metrics.get("eval_accuracy")
                ),
            )

            c2.metric(
                "F1",
                format_percentage(
                    metrics.get("f1")
                    or metrics.get("eval_f1")
                ),
            )

            c3.metric(
                "Version",
                best_model.get(
                    "model_version",
                    "—",
                ),
            )

            st.divider()

    except requests.RequestException as error:
        st.error(
            "Impossible de récupérer le journal des modèles."
        )

        st.code(
            str(error)
        )

        return

    deep_learning_execution = find_latest_execution(
        history,
        is_deep_learning_execution,
    )

    machine_learning_execution = find_latest_execution(
        history,
        is_machine_learning_execution,
    )

    # ---------------------------------------------------------------
    # Cartes des moteurs
    # ---------------------------------------------------------------

    model_columns = st.columns(2)

    with model_columns[0]:
        display_model_card(
            title="Deep Learning",
            icon="🧠",
            execution=deep_learning_execution,
            fallback_name="CamemBERT / JuriBERT",
            model_family="Transformer",
            data_type="Texte",
        )

    with model_columns[1]:
        display_model_card(
            title="Machine Learning",
            icon="🌳",
            execution=machine_learning_execution,
            fallback_name="Random Forest",
            model_family="Arbres de décision",
            data_type="Données structurées",
        )

    st.divider()

    # ---------------------------------------------------------------
    # Différences pédagogiques
    # ---------------------------------------------------------------

    st.header(
        "🔎 Différences principales"
    )

    differences = pd.DataFrame(
        [
            {
                "Critère": "Modèle",
                "Deep Learning": (
                    deep_learning_execution.get(
                        "model_name",
                        "CamemBERT / JuriBERT",
                    )
                    if deep_learning_execution
                    else "CamemBERT / JuriBERT"
                ),
                "Machine Learning": "Random Forest",
            },
            {
                "Critère": "Famille",
                "Deep Learning": "Transformer",
                "Machine Learning": (
                    "Ensemble d'arbres de décision"
                ),
            },
            {
                "Critère": "Données principales",
                "Deep Learning": "Textes annotés",
                "Machine Learning": (
                    "Questionnaires structurés"
                ),
            },
            {
                "Critère": "Entrées",
                "Deep Learning": (
                    "Texte + Label"
                ),
                "Machine Learning": (
                    "Features + Target"
                ),
            },
            {
                "Critère": "Apprentissage",
                "Deep Learning": (
                    "Ajustement des poids du Transformer"
                ),
                "Machine Learning": (
                    "Construction de plusieurs arbres"
                ),
            },
            {
                "Critère": "Prétraitement",
                "Deep Learning": "Tokenisation",
                "Machine Learning": (
                    "Encodage des catégories"
                ),
            },
            {
                "Critère": "Temps d'entraînement",
                "Deep Learning": "Plus long",
                "Machine Learning": "Rapide",
            },
            {
                "Critère": "Interprétabilité",
                "Deep Learning": "Plus complexe",
                "Machine Learning": "Plus accessible",
            },
        ]
    )

    st.dataframe(
        differences,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    display_metric_comparison(
        deep_learning=deep_learning_execution,
        machine_learning=machine_learning_execution,
    )

    st.divider()

    display_training_information(
        deep_learning=deep_learning_execution,
        machine_learning=machine_learning_execution,
    )

    st.divider()

    # ---------------------------------------------------------------
    # Complémentarité dans PRUDENCIA
    # ---------------------------------------------------------------

    st.header(
        "🔗 Complémentarité dans PRUDENCIA"
    )

    workflow_columns = st.columns(2)

    with workflow_columns[0]:
        st.markdown(
            """
            ### 🧠 Analyse textuelle

            Description du projet ou document juridique

            ↓

            CamemBERT / JuriBERT

            ↓

            Classification et compréhension du texte
            """
        )

    with workflow_columns[1]:
        st.markdown(
            """
            ### 🌳 Analyse structurée

            Réponses au questionnaire

            ↓

            Random Forest

            ↓

            Classification du niveau de conformité
            """
        )

    st.success(
        """
        Le rapport final PRUDENCIA pourra fusionner les résultats
        textuels du Deep Learning, les résultats structurés du
        Machine Learning et les références juridiques trouvées par le RAG.
        """
    )

    # ---------------------------------------------------------------
    # Message pédagogique
    # ---------------------------------------------------------------

    with st.expander(
        "Pourquoi utiliser deux moteurs ?",
        expanded=True,
    ):
        st.write(
            """
            Le Deep Learning est adapté à l'analyse du langage naturel.
            Il peut apprendre à reconnaître des catégories à partir
            de textes annotés.

            Le Random Forest est adapté aux données structurées.
            Il apprend à prédire une Target à partir de plusieurs Features.

            Les deux approches répondent donc à des besoins différents
            et complémentaires dans PRUDENCIA.
            """
        )

    comparison_export = []

    for execution in history:

        if not is_training_execution(execution):
            continue

        comparison_export.append(
            {
                "Model": execution.get("model_name"),
                "Version": execution.get("model_version"),
                "Dataset": execution.get("dataset_name"),
                "Accuracy": get_metric(execution, "accuracy"),
                "Precision": get_metric(execution, "precision"),
                "Recall": get_metric(execution, "recall"),
                "F1": get_metric(execution, "f1"),
                "Training Time (s)": get_execution_time(execution),
            }
        )

    if comparison_export:

        csv = pd.DataFrame(
            comparison_export
        ).to_csv(
            index=False
        )

        st.download_button(
            "📥 Export CSV",
            csv,
            "comparison_models.csv",
            "text/csv",
        )

    with st.expander(
        "Afficher les données techniques",
        expanded=False,
    ):
        st.json(
            {
                "deep_learning": deep_learning_execution,
                "machine_learning": machine_learning_execution,
            }
        )