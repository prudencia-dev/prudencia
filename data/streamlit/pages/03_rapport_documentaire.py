from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import requests
import streamlit as st


API_URL = os.getenv(
    "PRUDENCIA_API_URL",
    os.getenv("API_URL", "http://api:8000"),
).rstrip("/")

COLLECTION_NAME = os.getenv(
    "PRUDENCIA_RAG_COLLECTION",
    "prudencia_legal_documents",
)

FINE_TUNED_MODEL_NAME = os.getenv(
    "PRUDENCIA_FINE_TUNED_MODEL_NAME",
    "juribert",
)

EXTRACT_ENDPOINT = "/document-analysis/extract"
PREDICT_ENDPOINT = "/fine-tuning/predict"
RAG_SEARCH_ENDPOINT = "/rag/search"
REPORT_ENDPOINT = "/reports/generate"

REQUEST_TIMEOUT = 120
DEFAULT_RAG_LIMIT = 5
MAX_MODEL_TEXT_LENGTH = 12000
MAX_RAG_QUERY_LENGTH = 6000


st.set_page_config(
    page_title="Rapport documentaire",
    page_icon="📘",
    layout="wide",
)


class PrudenciaAPIError(RuntimeError):
    pass


def parse_response(
    response: requests.Response,
    endpoint: str,
) -> dict[str, Any]:
    if response.status_code != 200:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text

        raise PrudenciaAPIError(
            f"Erreur API {response.status_code} "
            f"sur {endpoint} : {detail}"
        )

    try:
        result = response.json()
    except ValueError as error:
        raise PrudenciaAPIError(
            f"L'endpoint {endpoint} n'a pas renvoyé de JSON valide."
        ) from error

    if not isinstance(result, dict):
        raise PrudenciaAPIError(
            f"L'endpoint {endpoint} n'a pas renvoyé un objet JSON."
        )

    return result


def post_json(
    endpoint: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{API_URL}{endpoint}",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as error:
        raise PrudenciaAPIError(
            f"Impossible de contacter {endpoint} : {error}"
        ) from error

    return parse_response(response, endpoint)


def extract_pdf(
    uploaded_pdf: Any,
) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{API_URL}{EXTRACT_ENDPOINT}",
            files={
                "file": (
                    uploaded_pdf.name,
                    uploaded_pdf.getvalue(),
                    "application/pdf",
                )
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as error:
        raise PrudenciaAPIError(
            f"Impossible d'envoyer le PDF à l'API : {error}"
        ) from error

    return parse_response(
        response,
        EXTRACT_ENDPOINT,
    )


def normalize_prediction(
    result: dict[str, Any],
) -> dict[str, Any]:
    prediction = (
        result.get("prediction")
        or result.get("label")
        or result.get("classification")
        or "À déterminer"
    )

    normalized: dict[str, Any] = {
        "prediction": str(prediction),
        "classification": str(prediction),
        "confidence": result.get("confidence"),
    }

    probabilities = result.get("probabilities")

    if isinstance(probabilities, dict):
        normalized["probabilities"] = probabilities

    if result.get("model_name"):
        normalized["model_name"] = result["model_name"]

    return normalized


def normalize_references(
    rag_result: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_results = (
        rag_result.get("results")
        or rag_result.get("references")
        or []
    )

    if not isinstance(raw_results, list):
        return []

    references: list[dict[str, Any]] = []

    for item in raw_results:
        if not isinstance(item, dict):
            references.append(
                {
                    "text": str(item),
                    "filename": "Document réglementaire",
                }
            )
            continue

        metadata = item.get("metadata")

        if not isinstance(metadata, dict):
            metadata = {}

        reference: dict[str, Any] = {
            "text": str(
                item.get("text")
                or item.get("content")
                or item.get("excerpt")
                or ""
            ),
            "filename": str(
                item.get("filename")
                or item.get("source")
                or metadata.get("filename")
                or metadata.get("source")
                or "Document réglementaire"
            ),
        }

        article = (
            item.get("article")
            or metadata.get("article")
        )

        if article:
            reference["article"] = str(article)

        similarity = (
            item.get("similarity")
            if isinstance(item.get("similarity"), (int, float))
            else item.get("score")
        )

        if isinstance(similarity, (int, float)):
            reference["score"] = float(similarity)

        references.append(reference)

    return references


def format_confidence(
    value: Any,
) -> str:
    if not isinstance(value, (int, float)):
        return "—"

    numeric_value = float(value)

    if numeric_value <= 1:
        return f"{numeric_value:.1%}"

    return f"{numeric_value:.1f} %"


def display_classification(
    classification: str,
) -> None:
    normalized = classification.lower()

    if "interdit" in normalized:
        st.error(f"⛔ {classification}")
    elif "haut" in normalized or "élevé" in normalized:
        st.error(f"🔴 {classification}")
    elif "limité" in normalized or "modéré" in normalized:
        st.warning(f"🟠 {classification}")
    elif "minimal" in normalized or "faible" in normalized:
        st.success(f"🟢 {classification}")
    else:
        st.info(f"🔵 {classification}")


def display_report(
    report: dict[str, Any],
) -> None:
    project = report.get("project") or {}
    ai_act = report.get("ai_act") or {}
    risks = report.get("risks") or []
    recommendations = report.get("recommendations") or []
    legal_references = report.get("legal_references") or []

    st.divider()
    st.header("📘 Rapport de pré-diagnostic")

    st.subheader(
        str(project.get("title") or "Projet IA")
    )

    if project.get("description"):
        st.write(str(project["description"]))

    columns = st.columns(4)
    columns[0].metric("Version", report.get("version", "1.0"))
    columns[1].metric("Risques", len(risks))
    columns[2].metric("Recommandations", len(recommendations))
    columns[3].metric("Références", len(legal_references))

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Classification AI Act")

        classification = str(
            ai_act.get("classification")
            or ai_act.get("prediction")
            or "À déterminer"
        )

        display_classification(classification)

        st.metric(
            "Indice de confiance",
            format_confidence(
                ai_act.get("confidence")
            ),
        )

    with right:
        st.subheader("État de conformité")
        st.info(
            str(
                report.get("conformity_status")
                or "À déterminer"
            )
        )

    justification = ai_act.get("justification")

    if justification:
        st.subheader("Justification")
        st.write(str(justification))

    st.divider()
    st.subheader("Risques identifiés")

    if not risks:
        st.info("Aucun risque spécifique n'a été identifié.")
    else:
        for index, risk in enumerate(risks, start=1):
            if isinstance(risk, dict):
                with st.container(border=True):
                    st.markdown(
                        f"**{index}. "
                        f"{risk.get('category', 'Général')} — "
                        f"{risk.get('level', 'À évaluer')}**"
                    )
                    st.write(
                        str(risk.get("description") or "")
                    )
            else:
                st.write(f"{index}. {risk}")

    st.divider()
    st.subheader("Recommandations")

    if not recommendations:
        st.info("Aucune recommandation n'a été générée.")
    else:
        for index, recommendation in enumerate(
            recommendations,
            start=1,
        ):
            if isinstance(recommendation, dict):
                with st.container(border=True):
                    st.markdown(
                        f"**Priorité "
                        f"{recommendation.get('priority', '—')} — "
                        f"{recommendation.get('category', 'Général')}**"
                    )
                    st.write(
                        str(recommendation.get("action") or "")
                    )
            else:
                st.write(f"{index}. {recommendation}")

    st.divider()
    st.subheader("Références réglementaires")

    if not legal_references:
        st.info(
            "Aucune référence réglementaire n'a été associée."
        )
    else:
        for index, reference in enumerate(
            legal_references,
            start=1,
        ):
            if not isinstance(reference, dict):
                st.write(reference)
                continue

            document = (
                reference.get("document")
                or reference.get("filename")
                or "Document réglementaire"
            )

            article = reference.get("article")
            title = f"{index}. {document}"

            if article:
                title += f" — {article}"

            with st.expander(
                title,
                expanded=index == 1,
            ):
                st.write(
                    str(
                        reference.get("excerpt")
                        or reference.get("text")
                        or ""
                    )
                )

    conclusion = report.get("conclusion")

    if conclusion:
        st.divider()
        st.subheader("Conclusion")
        st.write(str(conclusion))

    report_json = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    st.divider()

    st.download_button(
        "⬇️ Télécharger le rapport JSON",
        data=report_json,
        file_name="rapport_documentaire_prudencia.json",
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("Afficher les données techniques"):
        st.json(report)


st.session_state.setdefault("documentary_report", None)
st.session_state.setdefault("documentary_pipeline", None)
st.session_state.setdefault("documentary_history", [])


st.title("PRUDENCIA")
st.subheader("Pré-diagnostic documentaire d'un projet IA")

st.write(
    "Déposez le document décrivant votre projet. "
    "PRUDENCIA extrait son contenu avec FastAPI, "
    "le classe avec JuriBERT, interroge le corpus juridique "
    "et génère automatiquement le rapport."
)

with st.form(
    "documentary_analysis_form",
    clear_on_submit=False,
):
    project_title = st.text_input(
        "Nom du projet",
        placeholder="Exemple : Assistant de recrutement",
    )

    uploaded_pdf = st.file_uploader(
        "Document de présentation du projet",
        type=["pdf"],
    )

    rag_limit = st.slider(
        "Nombre de références réglementaires",
        min_value=3,
        max_value=10,
        value=DEFAULT_RAG_LIMIT,
    )

    launch = st.form_submit_button(
        "🚀 Lancer l'analyse complète",
        type="primary",
        use_container_width=True,
    )


if launch:
    if not project_title.strip():
        st.error("Le nom du projet est obligatoire.")
    elif uploaded_pdf is None:
        st.error("Veuillez sélectionner un fichier PDF.")
    else:
        progress = st.progress(0)
        status = st.empty()

        try:
            status.info("1/5 — Envoi et extraction du PDF")
            progress.progress(15)

            extraction = extract_pdf(uploaded_pdf)
            extracted_text = str(
                extraction.get("text") or ""
            ).strip()

            if not extracted_text:
                raise PrudenciaAPIError(
                    "L'API n'a renvoyé aucun texte extrait."
                )

            status.info("2/5 — Analyse avec JuriBERT Fine-Tuné")
            progress.progress(35)

            prediction = post_json(
                PREDICT_ENDPOINT,
                {
                    "text": extracted_text[
                        :MAX_MODEL_TEXT_LENGTH
                    ],
                    "model_name": FINE_TUNED_MODEL_NAME,
                },
            )

            status.info(
                "3/5 — Recherche réglementaire dans le RAG"
            )
            progress.progress(60)

            rag_result = post_json(
                RAG_SEARCH_ENDPOINT,
                {
                    "query": extracted_text[
                        :MAX_RAG_QUERY_LENGTH
                    ],
                    "collection_name": COLLECTION_NAME,
                    "limit": rag_limit,
                },
            )

            status.info("4/5 — Assemblage du pré-diagnostic")
            progress.progress(80)

            report_payload = {
                "project": {
                    "title": project_title.strip(),
                    "description": extracted_text,
                    "source_filename": uploaded_pdf.name,
                    "page_count": extraction.get(
                        "page_count"
                    ),
                },
                "deep_learning_result": normalize_prediction(
                    prediction
                ),
                "rag_result": {
                    "references": normalize_references(
                        rag_result
                    ),
                },
            }

            status.info("5/5 — Génération du rapport")
            progress.progress(90)

            report = post_json(
                REPORT_ENDPOINT,
                report_payload,
            )

            progress.progress(100)
            status.success(
                "Analyse terminée et rapport généré."
            )

            st.session_state.documentary_report = report
            st.session_state.documentary_pipeline = {
                "generated_at": datetime.now().isoformat(),
                "source_filename": uploaded_pdf.name,
                "page_count": extraction.get("page_count"),
                "character_count": extraction.get(
                    "character_count"
                ),
                "prediction": prediction,
                "rag": rag_result,
            }

            st.session_state.documentary_history.insert(
                0,
                {
                    "project_title": project_title.strip(),
                    "filename": uploaded_pdf.name,
                    "report": report,
                },
            )

            del st.session_state.documentary_history[10:]

        except PrudenciaAPIError as error:
            progress.empty()
            status.empty()
            st.error(str(error))


report = st.session_state.documentary_report

if isinstance(report, dict):
    display_report(report)


pipeline = st.session_state.documentary_pipeline

if isinstance(pipeline, dict):
    with st.expander("Voir le pipeline exécuté"):
        columns = st.columns(3)

        columns[0].metric(
            "Pages",
            pipeline.get("page_count") or 0,
        )

        columns[1].metric(
            "Caractères",
            pipeline.get("character_count") or 0,
        )

        rag_results = pipeline.get("rag") or {}

        columns[2].metric(
            "Références",
            len(rag_results.get("results") or []),
        )

        st.json(pipeline)
