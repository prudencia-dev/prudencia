from __future__ import annotations

from typing import Any

import streamlit as st

from services.api import PrudenciaAPI


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

BASE_MODEL_NAME = "JuriBERT Base"
BASE_MODEL_ID = "dascim/juribert-base"
BASE_MODEL_KEY = "juribert"
BASE_MODEL_ARCHITECTURE = "BERT"
BASE_MODEL_LANGUAGE = "Français"
BASE_MODEL_DOMAIN = "Juridique"

MODELS_ENDPOINT = "/fine-tuning/models"
PREDICT_ENDPOINT = "/fine-tuning/predict"


# -------------------------------------------------------------------
# Fonctions API
# -------------------------------------------------------------------

def _get_status(
    endpoint: str,
) -> tuple[bool, dict[str, Any]]:
    """
    Interroge un endpoint FastAPI.
    """

    try:
        result = PrudenciaAPI.get(endpoint)

        if isinstance(result, dict):
            return True, result

        return True, {
            "response": result,
        }

    except Exception as error:
        return False, {
            "error": str(error),
        }


def _find_value(
    data: dict[str, Any],
    names: list[str],
    default: Any = None,
) -> Any:
    """
    Recherche une valeur parmi plusieurs noms de clés possibles.
    """

    for name in names:
        value = data.get(name)

        if value is not None:
            return value

    return default


def _is_juribert_available(
    status: dict[str, Any],
) -> bool:
    """
    Vérifie que JuriBERT est présent dans les modèles disponibles.
    """

    models = status.get("models", [])

    if not isinstance(models, list):
        return False

    for model in models:
        if not isinstance(model, dict):
            continue

        model_id = str(
            model.get("id")
            or model.get("model_name")
            or model.get("key")
            or ""
        ).lower()

        if model_id == BASE_MODEL_KEY:
            return True

    return False


# -------------------------------------------------------------------
# Fonctions d'affichage
# -------------------------------------------------------------------

def _format_confidence(
    value: Any,
) -> str:
    """
    Formate un score de confiance.
    """

    if not isinstance(
        value,
        (int, float),
    ):
        return "—"

    numeric_value = float(value)

    if numeric_value <= 1:
        return f"{numeric_value:.1%}"

    return f"{numeric_value:.1f} %"


def _display_prediction(
    result: dict[str, Any],
) -> None:
    """
    Affiche le résultat retourné par le modèle Fine-Tuné.
    """

    prediction = _find_value(
        result,
        [
            "prediction",
            "predicted_label",
            "label",
            "class",
            "classification",
        ],
        default="Non disponible",
    )

    confidence = _find_value(
        result,
        [
            "confidence",
            "score",
            "probability",
            "confidence_score",
        ],
        default=None,
    )

    model_name = _find_value(
        result,
        [
            "model_name",
            "model",
            "selected_model",
        ],
        default=BASE_MODEL_NAME,
    )

    explanation = _find_value(
        result,
        [
            "explanation",
            "justification",
            "message",
            "description",
        ],
        default=None,
    )

    st.divider()
    st.subheader("Résultat de l'analyse")

    result_columns = st.columns(3)

    result_columns[0].metric(
        "Modèle",
        model_name,
    )

    result_columns[1].metric(
        "Classification",
        str(prediction),
    )

    result_columns[2].metric(
        "Confiance",
        _format_confidence(confidence),
    )

    if explanation:
        st.info(
            str(explanation)
        )

    probabilities = _find_value(
        result,
        [
            "probabilities",
            "scores",
            "class_probabilities",
        ],
        default=None,
    )

    if isinstance(
        probabilities,
        dict,
    ):
        st.write("### Probabilités par classe")

        probability_rows = []

        for label, probability in probabilities.items():
            probability_rows.append(
                {
                    "Classe": str(label),
                    "Probabilité": _format_confidence(
                        probability
                    ),
                }
            )

        st.dataframe(
            probability_rows,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander(
        "Afficher le résultat technique"
    ):
        st.json(result)


def _load_example() -> None:
    """
    Charge un exemple dans le champ de saisie.
    """

    st.session_state[
        "juribert_prediction_text"
    ] = (
        "Notre société souhaite développer une intelligence "
        "artificielle capable d'analyser automatiquement les CV "
        "des candidats, de les classer par pertinence et de "
        "recommander les meilleurs profils au recruteur."
    )


def _clear_prediction() -> None:
    """
    Réinitialise le test du modèle.
    """

    st.session_state[
        "juribert_prediction_text"
    ] = ""

    st.session_state.pop(
        "juribert_last_prediction",
        None,
    )


# -------------------------------------------------------------------
# Interface principale
# -------------------------------------------------------------------

def render_base_model() -> None:
    """
    Affiche les informations de JuriBERT et permet de tester
    le modèle Fine-Tuné.
    """

    st.subheader("JuriBERT")

    st.info(
        "JuriBERT est le modèle NLP de référence de PRUDENCIA. "
        "Il est spécialisé dans l'analyse des textes juridiques "
        "en français."
    )

    api_available, status = _get_status(
        MODELS_ENDPOINT
    )

    juribert_available = (
        api_available
        and _is_juribert_available(status)
    )

    model_columns = st.columns(3)

    model_columns[0].metric(
        "Modèle de référence",
        BASE_MODEL_NAME,
    )

    model_columns[1].metric(
        "Architecture",
        BASE_MODEL_ARCHITECTURE,
    )

    model_columns[2].metric(
        "Service Fine-Tuning",
        (
            "🟢 Disponible"
            if juribert_available
            else "🔴 Indisponible"
        ),
    )

    st.divider()

    st.write("### Informations du modèle")

    information_columns = st.columns(3)

    information_columns[0].metric(
        "Hugging Face",
        BASE_MODEL_ID,
    )

    information_columns[1].metric(
        "Langue",
        BASE_MODEL_LANGUAGE,
    )

    information_columns[2].metric(
        "Spécialité",
        BASE_MODEL_DOMAIN,
    )

    with st.expander(
        "Informations techniques"
    ):
        st.json(
            {
                "model_key": BASE_MODEL_KEY,
                "display_name": BASE_MODEL_NAME,
                "huggingface_id": BASE_MODEL_ID,
                "architecture": BASE_MODEL_ARCHITECTURE,
                "language": BASE_MODEL_LANGUAGE,
                "domain": BASE_MODEL_DOMAIN,
                "framework": "PyTorch / Transformers",
                "available": juribert_available,
                "models_api_response": status,
            }
        )

    st.divider()

    st.write("### Tester le modèle Fine-Tuné")

    st.caption(
        "Saisissez la description d'un projet IA. "
        "JuriBERT Fine-Tuné déterminera la classe apprise "
        "pendant l'entraînement."
    )

    if (
        "juribert_prediction_text"
        not in st.session_state
    ):
        st.session_state[
            "juribert_prediction_text"
        ] = ""

    st.text_area(
        "Description du projet IA",
        placeholder=(
            "Décrivez le fonctionnement et l'utilisation "
            "du projet d'intelligence artificielle..."
        ),
        height=180,
        key="juribert_prediction_text",
    )

    example_column, clear_column = st.columns(2)

    example_column.button(
        "Utiliser l'exemple",
        use_container_width=True,
        key="juribert_example_button",
        on_click=_load_example,
    )

    clear_column.button(
        "Effacer",
        use_container_width=True,
        key="juribert_clear_button",
        on_click=_clear_prediction,
    )

    text = str(
        st.session_state.get(
            "juribert_prediction_text",
            "",
        )
    )

    launch_prediction = st.button(
        "Analyser avec JuriBERT Fine-Tuné",
        type="primary",
        use_container_width=True,
        disabled=(
            not juribert_available
            or not text.strip()
        ),
        key="juribert_prediction_button",
    )

    if not juribert_available:
        st.warning(
            "Le service Fine-Tuning ou le modèle JuriBERT "
            "n'est pas disponible."
        )

    if launch_prediction:
        try:
            with st.spinner(
                "Analyse du texte avec JuriBERT..."
            ):
                result = PrudenciaAPI.post(
                    PREDICT_ENDPOINT,
                    json={
                        "model_name": BASE_MODEL_KEY,
                        "text": text.strip(),
                    },
                )

            if not isinstance(
                result,
                dict,
            ):
                raise ValueError(
                    "La réponse du serveur n'est pas "
                    "au format JSON attendu."
                )

            st.session_state[
                "juribert_last_prediction"
            ] = result

            st.success(
                "Analyse terminée."
            )

        except Exception as error:
            st.error(
                "Impossible de tester le modèle Fine-Tuné."
            )

            st.code(
                str(error)
            )

    last_prediction = st.session_state.get(
        "juribert_last_prediction"
    )

    if isinstance(
        last_prediction,
        dict,
    ):
        _display_prediction(
            last_prediction
        )

    st.divider()

    st.write("### Rôle dans PRUDENCIA")

    st.write(
        """
Le modèle **JuriBERT Base** constitue le point de départ du
pipeline Deep Learning de PRUDENCIA.

Après Fine-Tuning, il analyse la description d'un projet IA
et prédit une catégorie apprise dans le Dataset annoté.

Le moteur d'embeddings du RAG est un composant séparé :

- **JuriBERT Fine-Tuné** réalise la classification ;
- le moteur d'embeddings transforme les documents en vecteurs ;
- **ChromaDB** stocke les vecteurs ;
- le RAG retrouve les références juridiques pertinentes.
"""
    )