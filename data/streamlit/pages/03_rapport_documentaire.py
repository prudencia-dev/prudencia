from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
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
HISTORY_ENDPOINT = "/training/analyses"

REQUEST_TIMEOUT = 120
DEFAULT_RAG_LIMIT = 5
DOCUMENT_CHUNK_SIZE = 4000
DOCUMENT_CHUNK_OVERLAP = 500
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


def get_analysis_history() -> list[dict[str, Any]]:
    try:
        response = requests.get(
            f"{API_URL}{HISTORY_ENDPOINT}",
            params={
                "limit": 50,
                "analysis_type": "documentaire",
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as error:
        raise PrudenciaAPIError(
            f"Impossible de récupérer l'historique : {error}"
        ) from error

    result = parse_response(
        response,
        HISTORY_ENDPOINT,
    )

    analyses = result.get("analyses", [])

    if not isinstance(analyses, list):
        return []

    return analyses


def extract_pdf(
    uploaded_pdf: Any,
    project_name: str,
) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{API_URL}{EXTRACT_ENDPOINT}",
            data={
                "project_name": project_name.strip(),
            },
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


def split_document(
    text: str,
    chunk_size: int = DOCUMENT_CHUNK_SIZE,
    overlap: int = DOCUMENT_CHUNK_OVERLAP,
) -> list[str]:
    cleaned_text = text.strip()

    if not cleaned_text:
        return []

    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    chunks: list[str] = []
    start = 0

    while start < len(cleaned_text):
        end = min(
            start + chunk_size,
            len(cleaned_text),
        )

        chunk = cleaned_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned_text):
            break

        start = end - overlap

    return chunks


def aggregate_predictions(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    if not predictions:
        raise PrudenciaAPIError(
            "Aucune prédiction n'a été produite."
        )

    probability_sums: dict[str, float] = defaultdict(float)
    probability_counts: dict[str, int] = defaultdict(int)
    votes: Counter[str] = Counter()
    confidences_by_label: dict[str, list[float]] = defaultdict(list)

    normalized_chunks: list[dict[str, Any]] = []

    for index, result in enumerate(predictions, start=1):
        label = str(
            result.get("prediction")
            or result.get("classification")
            or result.get("label")
            or "inconnu"
        )

        confidence = result.get("confidence")
        probabilities = result.get("probabilities") or {}

        votes[label] += 1

        if isinstance(confidence, (int, float)):
            confidences_by_label[label].append(float(confidence))

        if isinstance(probabilities, dict):
            for class_name, score in probabilities.items():
                if isinstance(score, (int, float)):
                    probability_sums[str(class_name)] += float(score)
                    probability_counts[str(class_name)] += 1

        normalized_chunks.append(
            {
                **result,
                "chunk": result.get("chunk", index),
                "chunk_length": result.get("chunk_length"),
            }
        )

    averaged_probabilities = {
        class_name: (
            probability_sums[class_name]
            / probability_counts[class_name]
        )
        for class_name in probability_sums
        if probability_counts[class_name] > 0
    }

    if averaged_probabilities:
        final_prediction = max(
            averaged_probabilities,
            key=averaged_probabilities.get,
        )
        final_confidence = averaged_probabilities[
            final_prediction
        ]
        aggregation_method = "moyenne_probabilites"
    else:
        final_prediction = votes.most_common(1)[0][0]
        matching_confidences = confidences_by_label.get(
            final_prediction,
            [],
        )
        final_confidence = (
            sum(matching_confidences)
            / len(matching_confidences)
            if matching_confidences
            else None
        )
        aggregation_method = "vote_majoritaire"

    return {
        "success": True,
        "prediction": final_prediction,
        "classification": final_prediction,
        "confidence": final_confidence,
        "probabilities": averaged_probabilities,
        "model_name": FINE_TUNED_MODEL_NAME,
        "aggregation_method": aggregation_method,
        "chunk_count": len(normalized_chunks),
        "votes": dict(votes),
        "chunks": normalized_chunks,
    }


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


def safe_progress_value(
    value: Any,
) -> float:
    if not isinstance(value, (int, float)):
        return 0.0

    numeric_value = float(value)

    if numeric_value > 1:
        numeric_value /= 100

    return min(max(numeric_value, 0.0), 1.0)


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
    diagnostics = report.get("analysis_diagnostics") or {}

    pipeline_info = st.session_state.get(
        "documentary_pipeline",
        {},
    )

    st.divider()
    st.header("📘 Rapport de pré-diagnostic")

    st.subheader(
        str(
            project.get("title")
            or project.get("name")
            or "Projet IA"
        )
    )

    if project.get("description"):
        st.write(str(project["description"]))

    st.divider()
    st.subheader("Informations sur le document")

    doc_col1, doc_col2, doc_col3, doc_col4 = st.columns(4)

    doc_col1.metric(
        "Version",
        report.get("version", "1.0"),
    )

    doc_col2.metric(
        "Pages",
        (
            project.get("page_count")
            or pipeline_info.get("page_count")
            or "—"
        ),
    )

    document_chars = (
        project.get("character_count")
        or pipeline_info.get("character_count")
        or len(str(project.get("description") or ""))
    )

    doc_col3.metric(
        "Caractères",
        f"{int(document_chars):,}".replace(",", " "),
    )

    doc_col4.metric(
        "Références",
        len(legal_references),
    )

    st.divider()

    classification = str(
        ai_act.get("classification")
        or ai_act.get("prediction")
        or report.get("classification")
        or "À déterminer"
    )

    st.subheader("Classification AI Act")
    display_classification(classification)

    raw_predictions = diagnostics.get("raw_predictions") or {}
    deep_learning = raw_predictions.get("deep_learning") or {}

    if not deep_learning:
        candidate = pipeline_info.get("prediction")
        deep_learning = (
            candidate if isinstance(candidate, dict) else {}
        )

    confidence = (
        ai_act.get("confidence")
        if ai_act.get("confidence") is not None
        else deep_learning.get("confidence")
    )

    conf_col1, conf_col2 = st.columns([1, 3])

    with conf_col1:
        st.metric(
            "Confiance",
            format_confidence(confidence),
        )

    with conf_col2:
        justification = ai_act.get("justification")

        if justification:
            st.info(str(justification))
        else:
            st.info(
                "Classification calculée automatiquement par "
                "le moteur d'analyse IA."
            )

    model_name = (
        deep_learning.get("model_name")
        or FINE_TUNED_MODEL_NAME
    )

    probabilities = (
        deep_learning.get("probabilities")
        or {}
    )

    st.divider()
    st.subheader("Modèle d'analyse")

    model_col1, model_col2 = st.columns(2)

    model_col1.metric(
        "Modèle",
        model_name,
    )

    model_col2.metric(
        "Confiance du modèle",
        format_confidence(confidence),
    )

    if probabilities:
        st.divider()
        st.subheader("Probabilités par classe")

        sorted_probabilities = sorted(
            probabilities.items(),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        for label, score in sorted_probabilities:
            st.progress(
                safe_progress_value(score),
                text=(
                    f"{label} : "
                    f"{format_confidence(score)}"
                ),
            )

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

    st.divider()
    st.subheader("Pipeline d'analyse")

    pipeline = [
        ("📄", "Extraction PDF"),
        ("🤖", "JuriBERT"),
        ("📚", "Recherche RAG"),
        ("📘", "Rapport PRUDENCIA"),
    ]

    cols = st.columns(len(pipeline))

    for col, (icon, label) in zip(cols, pipeline):
        with col:
            st.success(icon)
            st.caption(label)

    execution_time = diagnostics.get("execution_time_ms")

    if execution_time:
        st.info(
            f"Temps total d'analyse : **{execution_time} ms**"
        )

    history = report.get("analysis_history") or {}

    if history.get("saved"):
        st.success(
            f"Analyse historisée : {history.get('analysis_id')}"
        )
    else:
        st.warning(
            "Analyse non historisée dans PostgreSQL."
        )

    st.divider()
    st.subheader("Diagnostic technique")

    character_count = int(
        pipeline_info.get(
            "character_count",
            0,
        )
        or 0
    )

    chunk_count = int(
        pipeline_info.get(
            "chunk_count",
            0,
        )
        or 0
    )

    chunk_size = int(
        pipeline_info.get(
            "chunk_size",
            DOCUMENT_CHUNK_SIZE,
        )
        or DOCUMENT_CHUNK_SIZE
    )

    rag_query_chars = min(
        character_count,
        MAX_RAG_QUERY_LENGTH,
    )

    diag1, diag2, diag3, diag4 = st.columns(4)

    diag1.metric(
        "Document",
        f"{character_count:,}".replace(",", " "),
    )

    diag2.metric(
        "Chunks",
        chunk_count,
    )

    diag3.metric(
        "Taille chunk",
        chunk_size,
    )

    diag4.metric(
        "RAG",
        f"{rag_query_chars:,}".replace(",", " "),
    )

    if chunk_count > 1:
        st.success(
            "Le document complet a été découpé et analysé "
            f"en {chunk_count} morceaux."
        )

    if character_count > MAX_RAG_QUERY_LENGTH:
        st.info(
            (
                "Le RAG utilise les premiers "
                f"{MAX_RAG_QUERY_LENGTH:,} caractères comme requête. "
                "La classification JuriBERT utilise tous les chunks."
            ).replace(",", " ")
        )

    prediction_info = pipeline_info.get(
        "prediction",
        {},
    )

    rag_info = pipeline_info.get(
        "rag",
        {},
    )

    rag_results = (
        rag_info.get("results")
        or rag_info.get("references")
        or []
    )

    st.subheader("Résumé du pipeline")

    st.json(
        {
            "Extraction PDF": "OK",
            "Texte extrait": character_count,
            "Chunks analysés": chunk_count,
            "Prédiction": (
                prediction_info.get("prediction")
                if isinstance(prediction_info, dict)
                else None
            ),
            "Confiance": (
                prediction_info.get("confidence")
                if isinstance(prediction_info, dict)
                else None
            ),
            "Méthode d'agrégation": (
                prediction_info.get("aggregation_method")
                if isinstance(prediction_info, dict)
                else None
            ),
            "Références RAG": (
                len(rag_results)
                if isinstance(rag_results, list)
                else 0
            ),
        }
    )

    st.divider()
    st.subheader("Contrôle de cohérence")

    consistency = diagnostics.get("consistency") or {}
    status = consistency.get("status", "not_checked")

    raw_prediction = (
        deep_learning.get("prediction")
        or (
            prediction_info.get("prediction")
            if isinstance(prediction_info, dict)
            else None
        )
        or "Non disponible"
    )

    final_prediction = (
        diagnostics.get("final_prediction")
        or ai_act.get("classification")
        or "Non disponible"
    )

    coh1, coh2 = st.columns(2)

    coh1.metric(
        "Prédiction JuriBERT",
        raw_prediction,
    )

    coh2.metric(
        "Rapport final",
        final_prediction,
    )

    if status == "coherent":
        st.success(
            "✅ Le rapport est cohérent avec la prédiction du modèle."
        )
    elif status == "warning":
        st.warning(
            consistency.get(
                "message",
                "Des différences ont été détectées.",
            )
        )
    elif status == "incoherent":
        st.error(
            consistency.get(
                "message",
                "Le rapport est incohérent avec la prédiction.",
            )
        )
    else:
        normalized_raw = str(raw_prediction).strip().lower()
        normalized_final = str(final_prediction).strip().lower()

        if (
            normalized_raw
            and normalized_final
            and normalized_raw != "non disponible"
            and normalized_final != "non disponible"
            and normalized_raw == normalized_final
        ):
            st.success(
                "✅ Le rapport est cohérent avec la prédiction du modèle."
            )
        else:
            st.info(
                "Le contrôle de cohérence n'a pas encore été "
                "enregistré par l'API."
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

            extraction = extract_pdf(
                uploaded_pdf,
                project_title,
            )
            extracted_text = str(
                extraction.get("text") or ""
            ).strip()

            if not extracted_text:
                raise PrudenciaAPIError(
                    "L'API n'a renvoyé aucun texte extrait."
                )

            chunks = split_document(extracted_text)

            if not chunks:
                raise PrudenciaAPIError(
                    "Le document ne contient aucun texte exploitable."
                )

            status.info(
                f"2/5 — Analyse de {len(chunks)} chunk(s) "
                "avec JuriBERT Fine-Tuné"
            )
            progress.progress(35)

            predictions: list[dict[str, Any]] = []

            for index, chunk in enumerate(chunks, start=1):
                result = post_json(
                    PREDICT_ENDPOINT,
                    {
                        "text": chunk,
                        "model_name": FINE_TUNED_MODEL_NAME,
                    },
                )

                result["chunk"] = index
                result["chunk_length"] = len(chunk)
                predictions.append(result)

            prediction = aggregate_predictions(predictions)

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
                    "id": extraction.get("project_id"),
                    "project_id": extraction.get("project_id"),
                    "title": project_title.strip(),
                    "name": project_title.strip(),
                    "description": extracted_text,
                    "source_filename": uploaded_pdf.name,
                    "source_name": uploaded_pdf.name,
                    "page_count": extraction.get("page_count"),
                    "character_count": extraction.get("character_count"),
                },
                "machine_learning_result": {},
                "deep_learning_result": prediction,
                "rag_result": {
                    **rag_result,
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
                "project_id": extraction.get("project_id"),
                "source_filename": uploaded_pdf.name,
                "page_count": extraction.get("page_count"),
                "character_count": extraction.get(
                    "character_count"
                ),
                "chunk_count": len(chunks),
                "chunk_size": DOCUMENT_CHUNK_SIZE,
                "prediction": prediction,
                "predictions": predictions,
                "rag": rag_result,
                "report_payload": report_payload,
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
        columns = st.columns(4)

        columns[0].metric(
            "Pages",
            pipeline.get("page_count") or 0,
        )

        columns[1].metric(
            "Caractères",
            pipeline.get("character_count") or 0,
        )

        columns[2].metric(
            "Chunks",
            pipeline.get("chunk_count") or 0,
        )

        rag_results = pipeline.get("rag") or {}

        references = (
            rag_results.get("results")
            or rag_results.get("references")
            or []
        )

        columns[3].metric(
            "Références",
            len(references)
            if isinstance(references, list)
            else 0,
        )

        st.json(pipeline)


st.divider()
st.subheader("Historique des analyses documentaires")

try:
    history = get_analysis_history()

    if not history:
        st.info(
            "Aucune analyse documentaire enregistrée."
        )
    else:
        for item in history[:10]:
            created_at = str(
                item.get("created_at") or ""
            )

            project_name = str(
                item.get("project_name")
                or item.get("source_name")
                or "Projet"
            )

            with st.expander(
                f"{created_at} — {project_name}"
            ):
                c1, c2 = st.columns(2)

                c1.metric(
                    "Classification",
                    item.get(
                        "final_prediction",
                        "—",
                    ),
                )

                c2.metric(
                    "Statut",
                    item.get(
                        "status",
                        "—",
                    ),
                )

                consistency = item.get(
                    "consistency",
                    {},
                )

                history_status = consistency.get(
                    "status",
                    "not_checked",
                )

                if history_status == "coherent":
                    st.success(
                        "Cohérence vérifiée"
                    )
                elif history_status == "warning":
                    st.warning(
                        consistency.get(
                            "message",
                            "Attention",
                        )
                    )
                elif history_status == "incoherent":
                    st.error(
                        consistency.get(
                            "message",
                            "Incohérence",
                        )
                    )
                else:
                    st.info(
                        "Contrôle de cohérence non disponible."
                    )

except PrudenciaAPIError as error:
    st.warning(
        f"Historique indisponible : {error}"
    )
