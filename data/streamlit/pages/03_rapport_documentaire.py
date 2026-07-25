from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st
import time

API_URL = os.getenv(
    "PRUDENCIA_API_URL",
    os.getenv("API_URL", "http://api:8000"),
).rstrip("/")


st.set_page_config(
    page_title="Rapport Documentaire",
    page_icon="📚",
    layout="wide",
)


def generate_report(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Appelle l'API de génération du rapport PRUDENCIA.
    """

    try:
        response = requests.post(
            f"{API_URL}/reports/generate",
            json=payload,
            timeout=120,
        )

    except requests.Timeout as error:
        raise RuntimeError(
            "Le délai de génération du rapport a été dépassé."
        ) from error

    except requests.ConnectionError as error:
        raise RuntimeError(
            "Impossible de contacter l'API PRUDENCIA."
        ) from error

    except requests.RequestException as error:
        raise RuntimeError(
            f"Erreur réseau : {error}"
        ) from error

    if response.status_code != 200:
        try:
            error_detail = response.json()
        except ValueError:
            error_detail = response.text

        raise RuntimeError(
            f"Erreur API {response.status_code} : "
            f"{error_detail}"
        )

    return response.json()


def display_classification(
    classification: str,
) -> None:
    """
    Affiche visuellement la classification AI Act.
    """

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
    """
    Affiche la vue client du rapport PRUDENCIA.
    """

    project = report.get("project", {})
    ai_act = report.get("ai_act", {})
    risks = report.get("risks", [])
    recommendations = report.get(
        "recommendations",
        [],
    )
    legal_references = report.get(
        "legal_references",
        [],
    )

    st.divider()
    st.header("Rapport de pré-diagnostic")

    st.subheader(
        project.get(
            "title",
            "Projet IA",
        )
    )

    project_description = project.get(
        "description",
        "",
    )

    if project_description:
        st.write(project_description)

    metadata_columns = st.columns(3)

    metadata_columns[0].metric(
        "Version",
        report.get("version", "1.0"),
    )

    metadata_columns[1].metric(
        "Risques",
        len(risks),
    )

    metadata_columns[2].metric(
        "Recommandations",
        len(recommendations),
    )

    st.caption(
        f"Identifiant d'analyse : "
        f"{report.get('analysis_id', 'Non disponible')}"
    )

    st.divider()

    classification_column, conformity_column = st.columns(2)

    with classification_column:
        st.subheader("Classification AI Act")

        classification = str(
            ai_act.get(
                "classification",
                "À déterminer",
            )
        )

        display_classification(
            classification
        )

        confidence = ai_act.get(
            "confidence"
        )

        if isinstance(
            confidence,
            (int, float),
        ):
            st.metric(
                "Indice de confiance",
                f"{float(confidence):.1%}",
            )

    with conformity_column:
        st.subheader("État de conformité")

        conformity_status = report.get(
            "conformity_status",
            "À déterminer",
        )

        st.info(
            str(conformity_status)
        )

    justification = ai_act.get(
        "justification",
        "",
    )

    if justification:
        st.subheader("Justification")
        st.write(justification)

    st.divider()
    st.subheader("Risques identifiés")

    if not risks:
        st.info(
            "Aucun risque spécifique n'a été identifié."
        )

    else:
        for index, risk in enumerate(
            risks,
            start=1,
        ):
            if not isinstance(risk, dict):
                st.warning(
                    f"{index}. {risk}"
                )
                continue

            category = risk.get(
                "category",
                "Général",
            )

            level = risk.get(
                "level",
                "À évaluer",
            )

            description = risk.get(
                "description",
                "",
            )

            with st.container(
                border=True
            ):
                st.markdown(
                    f"**{index}. {category} — {level}**"
                )
                st.write(description)

    st.divider()
    st.subheader("Recommandations")

    if not recommendations:
        st.info(
            "Aucune recommandation n'a été générée."
        )

    else:
        for recommendation in recommendations:
            if not isinstance(
                recommendation,
                dict,
            ):
                st.write(
                    f"- {recommendation}"
                )
                continue

            priority = recommendation.get(
                "priority",
                "—",
            )

            category = recommendation.get(
                "category",
                "Général",
            )

            action = recommendation.get(
                "action",
                "",
            )

            with st.container(
                border=True
            ):
                st.markdown(
                    f"**Priorité {priority} — {category}**"
                )
                st.write(action)

    st.divider()
    st.subheader("Références réglementaires")

    if not legal_references:
        st.info(
            "Aucune référence réglementaire "
            "n'a été associée."
        )

    else:
        for index, reference in enumerate(
            legal_references,
            start=1,
        ):
            if not isinstance(
                reference,
                dict,
            ):
                st.write(reference)
                continue

            document = reference.get(
                "document",
                "Document réglementaire",
            )

            article = reference.get(
                "article",
            )

            excerpt = reference.get(
                "excerpt",
                "",
            )

            reference_title = (
                f"{index}. {document}"
            )

            if article:
                reference_title += (
                    f" — {article}"
                )

            with st.expander(
                reference_title
            ):
                st.write(excerpt)

    conclusion = report.get(
        "conclusion",
        "",
    )

    if conclusion:
        st.divider()
        st.subheader("Conclusion")
        st.write(conclusion)

    st.divider()

    report_json = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    st.download_button(
        label="⬇️ Télécharger le rapport JSON",
        data=report_json,
        file_name="rapport_prudencia.json",
        mime="application/json",
        use_container_width=True,
    )

    if st.button(
        "🆕 Nouvelle analyse",
        use_container_width=True,
    ):
        st.session_state.pop(
            "prudencia_generated_report",
            None,
        )
        st.rerun()

    with st.expander(
        "Afficher le JSON complet"
    ):
        st.json(report)


st.title("📚 Rapports d'analyse")

tab_doc, tab_ml = st.tabs(
    [
        "📘 Analyse documentaire",
        "📗 Analyse questionnaire",
    ]
)

tab_doc, tab_ml = st.tabs(
    [
        "📘 Analyse documentaire",
        "📗 Analyse questionnaire",
    ]
)

with tab_doc:

    st.success(
        "🟢 Rapport documentaire"
    )

    st.success(
        "🟢 MVP PRUDENCIA - Démonstration"
    )

    st.caption(
        "Génération d'un pré-diagnostic réglementaire "
        "structuré pour un projet d'intelligence artificielle."
    )

    st.info(
        "Cette page de démonstration assemble les informations "
        "du projet, le résultat du questionnaire, l'analyse "
        "textuelle et les références réglementaires."
    )

    with st.form(
        "prudencia_report_form"
    ):
        st.subheader("1. Projet analysé")

        project_title = st.text_input(
            "Nom du projet",
            value="Assistant de recrutement",
        )

        project_description = st.text_area(
            "Description du projet",
            value=(
                "Système d'intelligence artificielle qui analyse "
                "les CV, évalue les candidatures et classe "
                "automatiquement les candidats."
            ),
            height=130,
        )

        st.subheader("2. Résultat du questionnaire")

        classification = st.selectbox(
            "Classification AI Act",
            [
                "Haut risque",
                "Risque limité",
                "Risque minimal",
                "Pratique interdite",
                "À déterminer",
            ],
        )

        confidence_percent = st.slider(
            "Indice de confiance",
            min_value=0,
            max_value=100,
            value=87,
        )

        risk_description = st.text_area(
            "Risque principal identifié",
            value=(
                "Le système intervient dans le domaine de "
                "l'emploi et peut influencer l'accès d'une "
                "personne à une opportunité professionnelle."
            ),
            height=100,
        )

        st.subheader("3. Analyse textuelle")

        justification = st.text_area(
            "Justification",
            value=(
                "Le projet automatise une partie du processus "
                "de sélection des candidats. Il doit donc faire "
                "l'objet d'une surveillance humaine et d'une "
                "documentation renforcée."
            ),
            height=120,
        )

        recommendation = st.text_area(
            "Recommandation principale",
            value=(
                "Mettre en place une supervision humaine, "
                "documenter les critères de classement et "
                "permettre la contestation des décisions."
            ),
            height=100,
        )

        st.subheader("4. Référence réglementaire")

        reference_document = st.text_input(
            "Document",
            value="Règlement européen sur l'intelligence artificielle",
        )

        reference_article = st.text_input(
            "Article ou annexe",
            value="Annexe III",
        )

        reference_excerpt = st.text_area(
            "Extrait réglementaire",
            value=(
                "Les systèmes d'IA destinés au recrutement "
                "ou à la sélection de personnes peuvent relever "
                "de la catégorie des systèmes à haut risque."
            ),
            height=110,
        )

        submitted = st.form_submit_button(
            "🚀 Générer le rapport PRUDENCIA",
            type="primary",
            use_container_width=True,
        )


    if submitted:
        if not project_description.strip():
            st.error(
                "La description du projet est obligatoire."
            )

        else:
            risks: list[dict[str, Any]] = []

            if risk_description.strip():
                risks.append(
                    {
                        "category": "AI Act",
                        "level": (
                            "Élevé"
                            if classification
                            == "Haut risque"
                            else "À évaluer"
                        ),
                        "description": (
                            risk_description.strip()
                        ),
                    }
                )

            recommendations: list[str] = []

            if recommendation.strip():
                recommendations.append(
                    recommendation.strip()
                )

            references: list[dict[str, Any]] = []

            if reference_excerpt.strip():
                references.append(
                    {
                        "text": (
                            reference_excerpt.strip()
                        ),
                        "filename": (
                            reference_document.strip()
                        ),
                        "article": (
                            reference_article.strip()
                            or None
                        ),
                        "score": 0.91,
                    }
                )

            payload = {
                "project": {
                    "title": (
                        project_title.strip()
                        or "Projet IA"
                    ),
                    "description": (
                        project_description.strip()
                    ),
                },
                "machine_learning_result": {
                    "prediction": classification,
                    "confidence": (
                        confidence_percent / 100
                    ),
                    "risks": risks,
                },
                "deep_learning_result": {
                    "justification": (
                        justification.strip()
                    ),
                    "recommendations": (
                        recommendations
                    ),
                },
                "rag_result": {
                    "references": references,
                },
            }

            try:
                status = st.empty()
                progress = st.progress(0)

                status.info("📄 Analyse du projet...")
                progress.progress(15)

                time.sleep(0.4)

                status.info("🤖 Analyse Machine Learning...")
                progress.progress(35)

                time.sleep(0.4)

                status.info("🧠 Analyse Deep Learning...")
                progress.progress(55)

                time.sleep(0.4)

                status.info("📚 Recherche réglementaire (RAG)...")
                progress.progress(75)

                with st.spinner("Consultation des connaissances..."):
                    generated_report = generate_report(payload)

                progress.progress(95)

                time.sleep(0.3)

                status.success("✅ Rapport PRUDENCIA généré")
                progress.progress(100)

                st.session_state[
                    "prudencia_generated_report"
                ] = generated_report

                history = st.session_state.setdefault(
                    "prudencia_history",
                    []
                )

                history.insert(
                    0,
                    generated_report,
                )

                history[:] = history[:10]

                st.success("Rapport généré avec succès.")

            except RuntimeError as error:
                st.error(str(error))


    generated_report = st.session_state.get(
        "prudencia_generated_report"
    )

    history = st.session_state.get(
        "prudencia_history",
        []
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Analyses",
        len(history),
    )

    high_risk = sum(
        1
        for r in history
        if "haut"
        in str(
            r.get("ai_act", {})
            .get("classification", "")
        ).lower()
    )

    c2.metric(
        "Haut risque",
        high_risk,
    )

    c3.metric(
        "Conformité",
        generated_report.get(
            "conformity_status",
            "—",
        ),
    )

    c4.metric(
        "Version",
        generated_report.get(
            "version",
            "1.0",
        ),
    )

    if generated_report:
        display_report(
            generated_report
        )

    history = st.session_state.get(
        "prudencia_history",
        []
    )

    if history:
        st.divider()
        st.header("🕘 Historique des analyses")

        for i, report in enumerate(history, start=1):

            project = report.get("project", {})

            title = project.get(
                "title",
                "Projet IA",
            )

            classification = (
                report.get("ai_act", {})
                .get("classification", "—")
            )

            with st.expander(
                f"{i}. {title} - {classification}"
            ):
                st.write(
                    project.get(
                        "description",
                        ""
                    )
                )

                st.json(report)