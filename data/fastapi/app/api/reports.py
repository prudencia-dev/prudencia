from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.analysis_orchestrator import AnalysisOrchestrator


router = APIRouter(prefix="/reports", tags=["Reports"])


class ReportGenerationRequest(BaseModel):
    project: dict[str, Any] = Field(default_factory=dict)
    machine_learning_result: dict[str, Any] = Field(default_factory=dict)
    deep_learning_result: dict[str, Any] = Field(default_factory=dict)
    rag_result: dict[str, Any] = Field(default_factory=dict)


class MachineLearningReportRequest(BaseModel):
    project_name: str = Field(min_length=1, max_length=255)
    project_id: str | None = None
    response_id: str | None = None
    prediction: str
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    features: dict[str, Any] = Field(default_factory=dict)


def normalize_prediction(prediction: str) -> str:
    normalized = (
        prediction.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    if "interdit" in normalized or "prohibited" in normalized:
        return "interdit"

    if "haut_risque" in normalized or "high_risk" in normalized:
        return "haut_risque"

    if "risque_limite" in normalized or "limited_risk" in normalized:
        return "risque_limite"

    if (
        "risque_minimal" in normalized
        or "minimal_risk" in normalized
        or normalized == "minimal"
    ):
        return "risque_minimal"

    if "hors_perimetre" in normalized or "hors_périmètre" in normalized:
        return "hors_perimetre"

    return normalized


def classification_label(risk_level: str) -> str:
    labels = {
        "interdit": "Pratique potentiellement interdite",
        "haut_risque": "Système potentiellement à haut risque",
        "risque_limite": "Système à risque limité",
        "risque_minimal": "Système à risque minimal",
        "hors_perimetre": "Hors périmètre identifié",
    }

    return labels.get(risk_level, "Classification à confirmer")


def friendly_value(name: str, value: Any) -> str:
    mappings = {
        "secteur_grp": {
            "rh_emploi": "Ressources humaines et emploi",
            "sante": "Santé",
            "finance_assurance": "Finance et assurance",
            "education": "Éducation",
            "public_justice_securite": "Secteur public, justice ou sécurité",
            "commerce_marketing": "Commerce et marketing",
            "industrie_energie_infra": "Industrie, énergie ou infrastructures",
            "tech_media_plateforme": "Technologie, média ou plateforme",
            "transport_logistique": "Transport et logistique",
            "services": "Services",
            "autre": "Autre secteur",
        },
        "role": {
            "fournisseur": "Fournisseur du système",
            "déployeur": "Utilisateur du système",
            "deployeur": "Utilisateur du système",
        },
        "donnees_perso": {
            "oui": "Oui",
            "non": "Non",
        },
        "donnees_sensibles": {
            "oui": "Oui",
            "non": "Non",
        },
        "type_ia_norm": {
            "scoring": "Classement et aide à la décision",
        },
    }

    return mappings.get(name, {}).get(str(value), str(value))


def build_analysis_inputs(features: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "label": "Secteur d'activité",
            "value": friendly_value(
                "secteur_grp",
                features.get("secteur_grp", ""),
            ),
        },
        {
            "label": "Rôle de l'organisation",
            "value": friendly_value(
                "role",
                features.get("role", ""),
            ),
        },
        {
            "label": "Données personnelles",
            "value": friendly_value(
                "donnees_perso",
                features.get("donnees_perso", ""),
            ),
        },
        {
            "label": "Données sensibles",
            "value": friendly_value(
                "donnees_sensibles",
                features.get("donnees_sensibles", ""),
            ),
        },
        {
            "label": "Usage principal",
            "value": friendly_value(
                "type_ia_norm",
                features.get("type_ia_norm", ""),
            ),
        },
    ]


def build_interpretation(
    risk_level: str,
    features: dict[str, Any],
) -> str:
    sector = friendly_value(
        "secteur_grp",
        features.get("secteur_grp", ""),
    )

    messages = {
        "interdit": (
            "Certaines caractéristiques du projet peuvent correspondre "
            "à une pratique interdite par l’AI Act. Une vérification "
            "juridique immédiate est nécessaire."
        ),
        "haut_risque": (
            f"Le système est utilisé dans le domaine « {sector} » "
            "pour classer ou assister une décision. Cet usage peut relever "
            "des systèmes à haut risque au sens de l’AI Act."
        ),
        "risque_limite": (
            "Le système semble principalement soumis à des obligations "
            "de transparence envers les personnes concernées."
        ),
        "risque_minimal": (
            "Le niveau de risque identifié est faible. Des mesures de "
            "gouvernance et de sécurité restent néanmoins recommandées."
        ),
        "hors_perimetre": (
            "Les informations fournies ne permettent pas de rattacher "
            "clairement le projet à une catégorie réglementée."
        ),
    }

    return messages.get(
        risk_level,
        "La classification doit être confirmée par un expert.",
    )


def obligations_for(risk_level: str) -> list[str]:
    if risk_level == "interdit":
        return [
            "Suspendre l’usage envisagé jusqu’à validation juridique.",
            "Vérifier si la pratique est interdite par l’AI Act.",
        ]

    if risk_level == "haut_risque":
        return [
            "Mettre en place une gestion des risques.",
            "Documenter le fonctionnement et les performances du système.",
            "Prévoir une supervision humaine effective.",
            "Conserver les journaux et les traces d’audit.",
            "Évaluer la robustesse, la sécurité et les biais.",
        ]

    if risk_level == "risque_limite":
        return [
            "Informer les personnes qu’elles interagissent avec une IA.",
            "Présenter clairement la finalité et les limites du système.",
        ]

    if risk_level == "risque_minimal":
        return [
            "Appliquer de bonnes pratiques de gouvernance.",
            "Surveiller les performances et les incidents.",
        ]

    return [
        "Faire confirmer le périmètre réglementaire du projet.",
    ]


def recommendations_for(
    risk_level: str,
    features: dict[str, Any],
) -> list[str]:
    recommendations = [
        "Faire valider ce pré-diagnostic par un expert juridique.",
        "Conserver les réponses et le rapport dans le dossier du projet.",
    ]

    if features.get("donnees_perso") == "oui":
        recommendations.append(
            "Vérifier les obligations applicables au titre du RGPD."
        )

    if features.get("donnees_sensibles") == "oui":
        recommendations.append(
            "Renforcer la protection des données sensibles."
        )

    if risk_level in {"haut_risque", "interdit"}:
        recommendations.extend(
            [
                "Vérifier les biais éventuels.",
                "Formaliser la supervision humaine.",
                "Tester la robustesse et la cybersécurité.",
            ]
        )

    return recommendations


@router.get("/health")
def reports_health() -> dict[str, Any]:
    return {
        "status": "ready",
        "service": "AnalysisOrchestrator",
        "report_builder": "PrudenciaReportBuilder",
        "schema_version": "1.0",
        "available_reports": [
            "global",
            "documentary",
            "machine_learning",
        ],
    }


@router.post("/generate")
def generate_report(
    request: ReportGenerationRequest,
) -> dict[str, Any]:
    try:
        orchestrator = AnalysisOrchestrator()

        return orchestrator.analyse(
            project=request.project,
            machine_learning_result=request.machine_learning_result,
            deep_learning_result=request.deep_learning_result,
            rag_result=request.rag_result,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur pendant la génération "
                f"du rapport PRUDENCIA : {error}"
            ),
        ) from error


@router.post("/generate-ml")
def generate_machine_learning_report(
    request: MachineLearningReportRequest,
) -> dict[str, Any]:
    try:
        risk_level = normalize_prediction(request.prediction)

        report = {
            "application": "PRUDENCIA",
            "version": "1.0.0-MVP",
            "report_type": "questionnaire",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "project": {
                "id": request.project_id,
                "name": request.project_name,
            },
            "questionnaire_response_id": request.response_id,
            "classification": {
                "code": risk_level,
                "label": classification_label(risk_level),
                "confidence": request.confidence,
                "probabilities": request.probabilities,
            },
            "analysis_inputs": build_analysis_inputs(request.features),
            "summary": build_interpretation(
                risk_level,
                request.features,
            ),
            "obligations": obligations_for(risk_level),
            "recommendations": recommendations_for(
                risk_level,
                request.features,
            ),
            "disclaimer": (
                "Ce rapport constitue un pré-diagnostic. "
                "Il ne remplace pas une analyse juridique."
            ),
        }

        return {
            "success": True,
            "report": report,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur pendant la génération "
                f"du rapport questionnaire : {error}"
            ),
        ) from error