from __future__ import annotations

from typing import Any

from app.domain.report import (
    AIActClassification,
    LegalReference,
    Project,
    PrudenciaReport,
    Recommendation,
    Risk,
)


class PrudenciaReportBuilder:
    """
    Construit le rapport métier final de PRUDENCIA.

    Le builder reçoit les résultats déjà produits par les moteurs
    Machine Learning, Deep Learning et RAG, puis les transforme en
    un rapport JSON homogène.

    Il ne modifie pas la décision du modèle : il normalise uniquement
    les libellés afin que la prédiction brute et le rapport final
    restent cohérents.
    """

    def build(
        self,
        *,
        project: dict[str, Any],
        machine_learning_result: dict[str, Any] | None = None,
        deep_learning_result: dict[str, Any] | None = None,
        rag_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Construit et retourne le rapport PRUDENCIA complet."""

        ml_result = machine_learning_result or {}
        dl_result = deep_learning_result or {}
        rag_data = rag_result or {}

        report = PrudenciaReport(
            project=self._build_project(project),
        )

        report.machine_learning_result = ml_result
        report.deep_learning_result = dl_result
        report.rag_result = rag_data

        report.ai_act = self._build_ai_act(
            ml_result=ml_result,
            dl_result=dl_result,
        )

        report.risks = self._build_risks(
            ml_result=ml_result,
            dl_result=dl_result,
        )

        report.recommendations = self._build_recommendations(
            ml_result=ml_result,
            dl_result=dl_result,
        )

        report.legal_references = self._build_legal_references(
            rag_result=rag_data,
        )

        # Conservé uniquement pour compatibilité avec le domaine existant.
        # L'interface Streamlit ne l'affiche plus, car PRUDENCIA produit
        # un pré-diagnostic et non une décision de conformité.
        report.conformity_status = "Non évalué"

        report.conclusion = self._build_conclusion(
            report=report,
        )

        return report.to_dict()

    # ------------------------------------------------------------------
    # Projet
    # ------------------------------------------------------------------

    def _build_project(
        self,
        project: dict[str, Any],
    ) -> Project:
        title = (
            project.get("title")
            or project.get("name")
            or project.get("project_name")
            or "Projet IA"
        )

        description = (
            project.get("description")
            or project.get("project_description")
            or ""
        )

        return Project(
            title=str(title),
            description=str(description),
        )

    # ------------------------------------------------------------------
    # Classification AI Act
    # ------------------------------------------------------------------

    def _build_ai_act(
        self,
        *,
        ml_result: dict[str, Any],
        dl_result: dict[str, Any],
    ) -> AIActClassification:
        """
        Détermine la classification principale.

        La valeur utilisée est exactement celle du moteur disponible,
        simplement normalisée dans un format commun.
        """

        classification = self._extract_prediction(ml_result)

        if classification is None:
            classification = self._extract_prediction(dl_result)

        normalized_classification = self._normalize_classification(
            classification
        )

        confidence = self._extract_confidence(ml_result)

        if confidence is None:
            confidence = self._extract_confidence(dl_result)

        justification = self._first_value(
            dl_result,
            [
                "justification",
                "explanation",
                "summary",
                "analyse",
                "analysis",
            ],
        )

        if justification is None:
            justification = self._first_value(
                ml_result,
                [
                    "justification",
                    "explanation",
                    "summary",
                    "analyse",
                    "analysis",
                ],
            )

        if justification is None:
            justification = self._default_justification(
                normalized_classification
            )

        return AIActClassification(
            classification=normalized_classification,
            confidence=confidence,
            justification=str(justification or ""),
        )

    def _extract_prediction(
        self,
        result: dict[str, Any],
    ) -> Any | None:
        return self._first_value(
            result,
            [
                "prediction",
                "predicted_class",
                "classification",
                "risk_level",
                "label",
                "final_prediction",
            ],
        )

    def _normalize_classification(
        self,
        value: Any,
    ) -> str:
        if value in (None, ""):
            return "À déterminer"

        normalized = (
            str(value)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("é", "e")
            .replace("è", "e")
            .replace("ê", "e")
            .replace("à", "a")
        )

        aliases = {
            "interdit": "interdit",
            "prohibited": "interdit",
            "pratique_interdite": "interdit",
            "haut_risque": "haut_risque",
            "high_risk": "haut_risque",
            "risque_eleve": "haut_risque",
            "limite": "risque_limite",
            "risque_limite": "risque_limite",
            "limited_risk": "risque_limite",
            "minimal": "risque_minimal",
            "risque_minimal": "risque_minimal",
            "minimal_risk": "risque_minimal",
            "hors_champ": "hors_perimetre",
            "hors_perimetre": "hors_perimetre",
            "out_of_scope": "hors_perimetre",
        }

        return aliases.get(normalized, normalized)

    def _default_justification(
        self,
        classification: str,
    ) -> str:
        messages = {
            "interdit": (
                "Le moteur a détecté des caractéristiques pouvant "
                "correspondre à une pratique interdite par l’AI Act."
            ),
            "haut_risque": (
                "Le moteur a détecté des caractéristiques pouvant "
                "correspondre à un système à haut risque."
            ),
            "risque_limite": (
                "Le moteur a détecté des obligations potentielles "
                "de transparence."
            ),
            "risque_minimal": (
                "Le moteur a identifié un niveau de risque faible "
                "au regard des informations analysées."
            ),
            "hors_perimetre": (
                "Le moteur n’a pas identifié de rattachement clair "
                "à une catégorie réglementée."
            ),
        }

        return messages.get(
            classification,
            "La classification doit être confirmée par un expert.",
        )

    # ------------------------------------------------------------------
    # Risques
    # ------------------------------------------------------------------

    def _build_risks(
        self,
        *,
        ml_result: dict[str, Any],
        dl_result: dict[str, Any],
    ) -> list[Risk]:
        risks: list[Risk] = []

        for source_result in (ml_result, dl_result):
            raw_risks = source_result.get("risks", [])

            if isinstance(raw_risks, str):
                raw_risks = [raw_risks]

            if not isinstance(raw_risks, list):
                continue

            for raw_risk in raw_risks:
                risk = self._convert_risk(raw_risk)

                if risk is not None:
                    risks.append(risk)

        return self._deduplicate_risks(risks)

    def _convert_risk(
        self,
        raw_risk: Any,
    ) -> Risk | None:
        if isinstance(raw_risk, str):
            return Risk(
                category="Général",
                level="À évaluer",
                description=raw_risk,
            )

        if not isinstance(raw_risk, dict):
            return None

        description = (
            raw_risk.get("description")
            or raw_risk.get("risk")
            or raw_risk.get("message")
            or raw_risk.get("label")
        )

        if not description:
            return None

        return Risk(
            category=str(
                raw_risk.get("category", "Général")
            ),
            level=str(
                raw_risk.get(
                    "level",
                    raw_risk.get("severity", "À évaluer"),
                )
            ),
            description=str(description),
        )

    def _deduplicate_risks(
        self,
        risks: list[Risk],
    ) -> list[Risk]:
        unique_risks: list[Risk] = []
        seen: set[tuple[str, str, str]] = set()

        for risk in risks:
            key = (
                risk.category.strip().lower(),
                risk.level.strip().lower(),
                risk.description.strip().lower(),
            )

            if key in seen:
                continue

            seen.add(key)
            unique_risks.append(risk)

        return unique_risks

    # ------------------------------------------------------------------
    # Recommandations
    # ------------------------------------------------------------------

    def _build_recommendations(
        self,
        *,
        ml_result: dict[str, Any],
        dl_result: dict[str, Any],
    ) -> list[Recommendation]:
        recommendations: list[Recommendation] = []

        for source_result in (ml_result, dl_result):
            raw_recommendations = source_result.get(
                "recommendations",
                [],
            )

            if isinstance(raw_recommendations, str):
                raw_recommendations = [raw_recommendations]

            if not isinstance(raw_recommendations, list):
                continue

            for index, raw_recommendation in enumerate(
                raw_recommendations,
                start=1,
            ):
                recommendation = self._convert_recommendation(
                    raw_recommendation,
                    default_priority=index,
                )

                if recommendation is not None:
                    recommendations.append(recommendation)

        return self._deduplicate_recommendations(
            recommendations
        )

    def _convert_recommendation(
        self,
        raw_recommendation: Any,
        *,
        default_priority: int,
    ) -> Recommendation | None:
        if isinstance(raw_recommendation, str):
            return Recommendation(
                priority=default_priority,
                category="Général",
                action=raw_recommendation,
            )

        if not isinstance(raw_recommendation, dict):
            return None

        action = (
            raw_recommendation.get("action")
            or raw_recommendation.get("recommendation")
            or raw_recommendation.get("description")
            or raw_recommendation.get("message")
        )

        if not action:
            return None

        raw_priority = raw_recommendation.get(
            "priority",
            default_priority,
        )

        try:
            priority = int(raw_priority)
        except (TypeError, ValueError):
            priority = default_priority

        return Recommendation(
            priority=priority,
            category=str(
                raw_recommendation.get(
                    "category",
                    "Général",
                )
            ),
            action=str(action),
        )

    def _deduplicate_recommendations(
        self,
        recommendations: list[Recommendation],
    ) -> list[Recommendation]:
        unique_recommendations: list[Recommendation] = []
        seen: set[tuple[str, str]] = set()

        for recommendation in recommendations:
            key = (
                recommendation.category.strip().lower(),
                recommendation.action.strip().lower(),
            )

            if key in seen:
                continue

            seen.add(key)
            unique_recommendations.append(recommendation)

        return sorted(
            unique_recommendations,
            key=lambda item: item.priority,
        )

    # ------------------------------------------------------------------
    # Références juridiques RAG
    # ------------------------------------------------------------------

    def _build_legal_references(
        self,
        *,
        rag_result: dict[str, Any],
    ) -> list[LegalReference]:
        raw_references = (
            rag_result.get("references")
            or rag_result.get("results")
            or rag_result.get("documents")
            or rag_result.get("chunks")
            or []
        )

        if not isinstance(raw_references, list):
            return []

        references: list[LegalReference] = []

        for raw_reference in raw_references:
            reference = self._convert_legal_reference(
                raw_reference
            )

            if reference is not None:
                references.append(reference)

        return references

    def _convert_legal_reference(
        self,
        raw_reference: Any,
    ) -> LegalReference | None:
        if isinstance(raw_reference, str):
            return LegalReference(
                source="RAG",
                excerpt=raw_reference,
            )

        if not isinstance(raw_reference, dict):
            return None

        metadata = raw_reference.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        excerpt = (
            raw_reference.get("excerpt")
            or raw_reference.get("text")
            or raw_reference.get("document")
            or raw_reference.get("content")
        )

        if not excerpt:
            return None

        relevance_score = self._extract_relevance_score(
            raw_reference
        )

        return LegalReference(
            source=str(
                raw_reference.get(
                    "source",
                    metadata.get("source", "RAG"),
                )
            ),
            excerpt=str(excerpt),
            document=self._optional_string(
                raw_reference.get(
                    "filename",
                    metadata.get("filename"),
                )
            ),
            article=self._optional_string(
                raw_reference.get(
                    "article",
                    metadata.get("article"),
                )
            ),
            relevance_score=relevance_score,
        )

    # ------------------------------------------------------------------
    # Conclusion
    # ------------------------------------------------------------------

    def _build_conclusion(
        self,
        *,
        report: PrudenciaReport,
    ) -> str:
        classification = report.ai_act.classification
        risk_count = len(report.risks)
        recommendation_count = len(report.recommendations)

        if classification == "À déterminer":
            return (
                "Les informations disponibles ne permettent pas "
                "de déterminer la classification du projet. "
                "Une analyse complémentaire est nécessaire."
            )

        labels = {
            "interdit": "pratique potentiellement interdite",
            "haut_risque": "système potentiellement à haut risque",
            "risque_limite": "système à risque limité",
            "risque_minimal": "système à risque minimal",
            "hors_perimetre": "projet potentiellement hors périmètre",
        }

        readable_classification = labels.get(
            classification,
            classification,
        )

        return (
            f"Le projet est classé comme « {readable_classification} ». "
            f"{risk_count} risque(s) et "
            f"{recommendation_count} recommandation(s) ont été "
            "identifiés. Ce résultat constitue un pré-diagnostic "
            "et doit être confirmé par un professionnel compétent."
        )

    # ------------------------------------------------------------------
    # Fonctions utilitaires
    # ------------------------------------------------------------------

    def _first_value(
        self,
        data: dict[str, Any],
        keys: list[str],
    ) -> Any | None:
        for key in keys:
            value = data.get(key)

            if value not in (None, "", [], {}):
                return value

        return None

    def _extract_confidence(
        self,
        result: dict[str, Any],
    ) -> float | None:
        raw_confidence = self._first_value(
            result,
            [
                "confidence",
                "probability",
                "score",
                "confidence_score",
            ],
        )

        if raw_confidence is None:
            return None

        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            return None

        if confidence > 1:
            confidence /= 100

        return round(
            max(0.0, min(1.0, confidence)),
            4,
        )

    def _extract_relevance_score(
        self,
        result: dict[str, Any],
    ) -> float | None:
        raw_score = self._first_value(
            result,
            [
                "relevance_score",
                "similarity",
                "score",
                "distance",
            ],
        )

        if raw_score is None:
            return None

        try:
            return round(float(raw_score), 4)
        except (TypeError, ValueError):
            return None

    def _optional_string(
        self,
        value: Any,
    ) -> str | None:
        if value in (None, ""):
            return None

        return str(value)
