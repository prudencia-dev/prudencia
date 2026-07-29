from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
import streamlit as st


# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Rapport questionnaire",
    page_icon="📋",
    layout="wide",
)

API_URL = os.getenv(
    "PRUDENCIA_API_URL",
    os.getenv("API_URL", "http://api:8000"),
).rstrip("/")

QUESTIONNAIRE_ENDPOINT = "/questionnaire-mvp/active"
SUBMIT_ENDPOINT = "/questionnaire-mvp/submit"
REPORT_ML_ENDPOINT = "/reports/generate-ml"

ML_PREDICT_ENDPOINTS = (
    "/ml/predict",
    "/machine-learning/predict",
)

REQUEST_TIMEOUT = 120


# =============================================================================
# API
# =============================================================================

class APIError(RuntimeError):
    """Erreur API affichable dans Streamlit."""


def api_request(
    method: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        response = requests.request(
            method=method,
            url=f"{API_URL}{endpoint}",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as error:
        raise APIError(
            f"Impossible de contacter l'API : {error}"
        ) from error

    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text

        raise APIError(
            f"Erreur API {response.status_code} "
            f"sur {endpoint} : {detail}"
        )

    try:
        data = response.json()
    except ValueError as error:
        raise APIError(
            f"Réponse JSON invalide depuis {endpoint}."
        ) from error

    if not isinstance(data, dict):
        raise APIError(
            f"Format de réponse inattendu depuis {endpoint}."
        )

    return data


def ml_predict(
    features: dict[str, Any],
) -> dict[str, Any]:
    """
    L'endpoint ML attend une liste de lignes dans la clé data.
    """
    payload = {
        "data": [
            features
        ]
    }

    last_error: APIError | None = None

    for endpoint in ML_PREDICT_ENDPOINTS:
        try:
            return api_request(
                "POST",
                endpoint,
                payload,
            )
        except APIError as error:
            last_error = error

            if "404" not in str(error):
                raise

    raise last_error or APIError(
        "Aucune route de prédiction ML disponible."
    )


# =============================================================================
# NORMALISATION DE LA PRÉDICTION
# =============================================================================

def first_item(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None

    return value


def extract_prediction_result(
    raw_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Accepte plusieurs formats de réponse possibles du service ML.
    """

    source = raw_result

    if isinstance(raw_result.get("result"), dict):
        source = raw_result["result"]

    elif isinstance(raw_result.get("results"), list):
        results = raw_result["results"]

        if results and isinstance(results[0], dict):
            source = results[0]

    prediction = (
        source.get("prediction")
        or source.get("classification")
        or source.get("risk_level_aiact")
        or source.get("predicted_class")
        or source.get("predictions")
    )

    prediction = first_item(prediction)

    confidence = (
        source.get("confidence")
        or source.get("score")
        or source.get("probability")
    )

    confidence = first_item(confidence)

    probabilities = (
        source.get("probabilities")
        or source.get("class_probabilities")
        or source.get("prediction_probabilities")
        or {}
    )

    if isinstance(probabilities, list):
        probabilities = (
            probabilities[0]
            if probabilities
            and isinstance(probabilities[0], dict)
            else {}
        )

    if prediction is None:
        raise APIError(
            "La réponse du modèle ML ne contient aucune classification."
        )

    return {
        "prediction": str(prediction),
        "confidence": (
            float(confidence)
            if isinstance(confidence, (int, float))
            else None
        ),
        "probabilities": (
            probabilities
            if isinstance(probabilities, dict)
            else {}
        ),
        "raw_result": raw_result,
    }


# =============================================================================
# QUESTIONNAIRE
# =============================================================================

def question_number(code: str) -> int:
    match = re.fullmatch(r"Q(\d+)", code.strip())

    if match:
        return int(match.group(1))

    return 10_000


def is_answered(
    value: Any,
    answer_type: str,
) -> bool:
    """
    False est une réponse valide correspondant à « Non ».
    """
    if answer_type == "boolean":
        return value is not None

    if answer_type == "multiple_choice":
        return isinstance(value, list) and len(value) > 0

    if isinstance(value, str):
        return bool(value.strip())

    return value is not None


def question_label(
    question: dict[str, Any],
) -> str:
    code = str(question.get("code", "")).strip()
    label = str(question.get("label", "")).strip()
    required = bool(question.get("is_required"))

    if re.fullmatch(r"Q\d+", code):
        result = f"{code}. {label}"
    elif code == "ML_TYPE_IA":
        result = f"Complément ML — {label}"
    else:
        result = f"{code} — {label}" if code else label

    if required:
        result += " *"

    return result


def render_question(
    question: dict[str, Any],
) -> Any:
    code = str(question["code"])
    answer_type = str(question["answer_type"])
    label = question_label(question)
    description = question.get("description")
    key = f"prudencia_{code}"

    if answer_type == "boolean":
        selected = st.radio(
            label,
            options=["Non renseigné", "Oui", "Non"],
            horizontal=True,
            help=description,
            key=key,
        )

        if selected == "Oui":
            return True

        if selected == "Non":
            return False

        return None

    if answer_type == "text":
        return st.text_area(
            label,
            help=description,
            key=key,
            height=100,
        ).strip()

    if answer_type == "integer":
        return int(
            st.number_input(
                label,
                step=1,
                help=description,
                key=key,
            )
        )

    if answer_type == "decimal":
        return float(
            st.number_input(
                label,
                help=description,
                key=key,
            )
        )

    options = question.get("options") or []

    labels = [
        str(option["label"])
        for option in options
        if isinstance(option, dict)
        and "label" in option
    ]

    values_by_label = {
        str(option["label"]): option.get(
            "value",
            option["label"],
        )
        for option in options
        if isinstance(option, dict)
        and "label" in option
    }

    if answer_type == "single_choice":
        selected = st.selectbox(
            label,
            options=["— Sélectionner —", *labels],
            help=description,
            key=key,
        )

        if selected == "— Sélectionner —":
            return None

        return values_by_label[selected]

    if answer_type == "multiple_choice":
        selected = st.multiselect(
            label,
            options=labels,
            help=description,
            key=key,
        )

        return [
            values_by_label[item]
            for item in selected
        ]

    st.warning(
        f"Type de question non pris en charge : {answer_type}"
    )

    return None


# =============================================================================
# AFFICHAGE DU RAPPORT MÉTIER
# =============================================================================

def format_confidence(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "Non disponible"

    numeric = float(value)

    if numeric <= 1:
        return f"{numeric:.1%}"

    return f"{numeric:.1f} %"


def display_business_report(
    report: dict[str, Any],
) -> None:
    classification = report.get("classification") or {}
    project = report.get("project") or {}
    features = report.get("features") or {}

    classification_code = str(
        classification.get("code", "")
    ).lower()

    classification_label_value = str(
        classification.get(
            "label",
            "Classification non disponible",
        )
    )

    confidence = classification.get("confidence")
    probabilities = (
        classification.get("probabilities")
        or {}
    )

    st.divider()
    st.header("📊 Rapport de pré-diagnostic AI Act")

    if classification_code == "interdit":
        st.error(
            f"⛔ {classification_label_value}"
        )
    elif classification_code == "haut_risque":
        st.error(
            f"🔴 {classification_label_value}"
        )
    elif classification_code == "risque_limite":
        st.warning(
            f"🟠 {classification_label_value}"
        )
    elif classification_code == "risque_minimal":
        st.success(
            f"🟢 {classification_label_value}"
        )
    else:
        st.info(
            f"🔵 {classification_label_value}"
        )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Projet",
        project.get("name", "Non renseigné"),
    )

    col2.metric(
        "Classification",
        classification_label_value,
    )

    col3.metric(
        "Confiance",
        format_confidence(confidence),
    )

    st.subheader("Interprétation")
    st.write(
        report.get(
            "interpretation",
            "Aucune interprétation disponible.",
        )
    )

    st.subheader("Variables analysées")

    feature_labels = {
        "secteur_grp": "Secteur",
        "role": "Rôle",
        "donnees_perso": "Données personnelles",
        "donnees_sensibles": "Données sensibles",
        "type_ia_norm": "Type d'IA",
    }

    if features:
        columns = st.columns(
            min(len(features), 5)
        )

        for index, (name, value) in enumerate(
            features.items()
        ):
            columns[index % len(columns)].metric(
                feature_labels.get(name, name),
                str(value),
            )

    if isinstance(probabilities, dict) and probabilities:
        st.subheader("Probabilités par classe")

        for class_name, probability in sorted(
            probabilities.items(),
            key=lambda item: float(item[1]),
            reverse=True,
        ):
            numeric = float(probability)

            progress_value = (
                numeric
                if numeric <= 1
                else numeric / 100
            )

            st.progress(
                min(max(progress_value, 0.0), 1.0),
                text=(
                    f"{class_name} — "
                    f"{format_confidence(numeric)}"
                ),
            )

    obligations = report.get("obligations") or []

    st.subheader("Obligations principales")

    if obligations:
        for obligation in obligations:
            st.write(f"✅ {obligation}")
    else:
        st.write(
            "Aucune obligation indicative n'a été générée."
        )

    recommendations = (
        report.get("recommendations")
        or []
    )

    st.subheader("Recommandations")

    if recommendations:
        for recommendation in recommendations:
            st.write(f"• {recommendation}")
    else:
        st.write(
            "Aucune recommandation n'a été générée."
        )

    disclaimer = report.get("disclaimer")

    if disclaimer:
        st.warning(disclaimer)

    st.download_button(
        label="⬇️ Télécharger le rapport JSON",
        data=json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        file_name=(
            "rapport_questionnaire_prudencia.json"
        ),
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("Données techniques"):
        st.json(report)


# =============================================================================
# PAGE
# =============================================================================

st.title("📋 Rapport questionnaire")

st.markdown(
    """
    Ce parcours crée ou retrouve un projet, enregistre les réponses
    dans PostgreSQL, réalise une prédiction avec le modèle
    **Random Forest**, puis génère un rapport métier de pré-diagnostic.
    """
)

st.subheader("📁 Informations du projet")

project_name = st.text_input(
    "Nom du projet *",
    placeholder="Exemple : Assistant IA de recrutement",
)

project_description = st.text_area(
    "Description du projet",
    placeholder=(
        "Décrivez brièvement le système "
        "d'intelligence artificielle."
    ),
)

st.caption(
    "Les champs marqués d'un astérisque (*) sont obligatoires."
)


# =============================================================================
# CHARGEMENT DE L'API
# =============================================================================

try:
    api_data = api_request(
        "GET",
        QUESTIONNAIRE_ENDPOINT,
    )
except APIError as error:
    st.error(str(error))
    st.stop()

questionnaire = api_data.get("questionnaire")

if not isinstance(questionnaire, dict):
    st.error(
        "La réponse de l'API ne contient pas "
        "de questionnaire valide."
    )
    st.stop()

questions = questionnaire.get("questions")

if not isinstance(questions, list) or not questions:
    st.error(
        "Aucune question active n'est disponible."
    )
    st.stop()

questions = sorted(
    questions,
    key=lambda question: (
        question_number(
            str(question.get("code", ""))
        ),
        int(
            question.get("display_order")
            or 10_000
        ),
    ),
)

st.info(
    f"**{questionnaire.get('name', 'Questionnaire PRUDENCIA')}**  \n"
    f"{questionnaire.get('description') or ''}"
)


# =============================================================================
# QUESTIONS PAR SECTION
# =============================================================================

section_titles = {
    "ai_act": "⚖️ AI Act et contexte du système",
    "rgpd": "🔐 Données et RGPD",
    "ethics": "🧭 Éthique et impacts",
    "technical": "⚙️ Technique, sécurité et documentation",
    "governance": "🏛️ Gouvernance et supervision",
}

category_order = [
    "ai_act",
    "rgpd",
    "ethics",
    "technical",
    "governance",
]

answers: dict[str, Any] = {}
displayed_ids: set[str] = set()

for category in category_order:
    category_questions = [
        question
        for question in questions
        if question.get("category") == category
    ]

    if not category_questions:
        continue

    st.divider()
    st.subheader(
        section_titles.get(
            category,
            category.title(),
        )
    )

    for question in category_questions:
        displayed_ids.add(
            str(question["id"])
        )

        value = render_question(question)
        code = str(question["code"])

        if is_answered(
            value,
            str(question["answer_type"]),
        ):
            answers[code] = value

remaining_questions = [
    question
    for question in questions
    if str(question["id"]) not in displayed_ids
]

if remaining_questions:
    st.divider()
    st.subheader("🧩 Informations complémentaires")

    for question in remaining_questions:
        value = render_question(question)
        code = str(question["code"])

        if is_answered(
            value,
            str(question["answer_type"]),
        ):
            answers[code] = value


# =============================================================================
# PROGRESSION
# =============================================================================

required_questions = [
    question
    for question in questions
    if question.get("is_required")
]

missing_questions = [
    question
    for question in required_questions
    if str(question["code"]) not in answers
]

answered_required = (
    len(required_questions)
    - len(missing_questions)
)

st.divider()
st.subheader("Progression")

st.progress(
    (
        answered_required / len(required_questions)
        if required_questions
        else 1.0
    ),
    text=(
        f"{answered_required}/{len(required_questions)} "
        "questions obligatoires renseignées"
    ),
)

if missing_questions:
    st.warning(
        f"{len(missing_questions)} question(s) "
        "obligatoire(s) reste(nt) à compléter."
    )
else:
    st.success(
        "Toutes les questions obligatoires sont renseignées."
    )


# =============================================================================
# ANALYSE COMPLÈTE
# =============================================================================

if st.button(
    "🚀 Lancer l'analyse complète",
    type="primary",
    use_container_width=True,
):
    if not project_name.strip():
        st.error("Le nom du projet est obligatoire.")

    elif missing_questions:
        st.error(
            "Le questionnaire ne peut pas encore être envoyé."
        )

        with st.expander(
            "Questions obligatoires manquantes",
            expanded=True,
        ):
            for question in missing_questions:
                st.write(
                    "- "
                    + question_label(
                        question
                    ).replace(" *", "")
                )

    else:
        progress = st.progress(0)
        status = st.empty()

        try:
            status.info(
                "1/4 — Création du projet et enregistrement..."
            )
            progress.progress(25)

            submission = api_request(
                "POST",
                SUBMIT_ENDPOINT,
                {
                    "project_name": project_name.strip(),
                    "project_description": (
                        project_description.strip()
                        or None
                    ),
                    "respondent_id": None,
                    "answers": answers,
                },
            )

            features = submission["features"]

            status.info(
                "2/4 — Prédiction Random Forest..."
            )
            progress.progress(50)

            raw_prediction = ml_predict(
                features
            )

            prediction = extract_prediction_result(
                raw_prediction
            )

            status.info(
                "3/4 — Génération du rapport métier..."
            )
            progress.progress(75)

            report_response = api_request(
                "POST",
                REPORT_ML_ENDPOINT,
                {
                    "project_name": project_name.strip(),
                    "project_id": str(
                        submission["project_id"]
                    ),
                    "response_id": str(
                        submission["response_id"]
                    ),
                    "prediction": prediction["prediction"],
                    "confidence": prediction["confidence"],
                    "probabilities": prediction["probabilities"],
                    "features": features,
                },
            )

            report = report_response.get("report")

            if not isinstance(report, dict):
                raise APIError(
                    "Le générateur de rapport n'a pas "
                    "renvoyé de rapport valide."
                )

            progress.progress(100)
            status.success(
                "4/4 — Rapport généré."
            )

            st.session_state[
                "questionnaire_last_report"
            ] = report

        except (APIError, KeyError, TypeError) as error:
            progress.empty()
            status.empty()

            st.error(
                f"L'analyse n'a pas pu être terminée : {error}"
            )


# =============================================================================
# RAPPORT CONSERVÉ EN SESSION
# =============================================================================

saved_report = st.session_state.get(
    "questionnaire_last_report"
)

if isinstance(saved_report, dict):
    display_business_report(saved_report)