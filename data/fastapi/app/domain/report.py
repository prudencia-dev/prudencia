from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class Project:
    """
    Informations relatives au projet IA analysé.
    """

    title: str
    description: str


@dataclass
class AIActClassification:
    """
    Classification réglementaire du projet selon l'AI Act.
    """

    classification: str = "À déterminer"
    confidence: float | None = None
    justification: str = ""


@dataclass
class Risk:
    """
    Risque identifié pendant l'analyse.
    """

    category: str
    level: str
    description: str


@dataclass
class Recommendation:
    """
    Action recommandée à la suite de l'analyse.
    """

    priority: int
    category: str
    action: str


@dataclass
class LegalReference:
    """
    Référence juridique retrouvée par le RAG.
    """

    source: str
    excerpt: str
    document: str | None = None
    article: str | None = None
    relevance_score: float | None = None


@dataclass
class PrudenciaReport:
    """
    Représentation métier du rapport final PRUDENCIA.

    Cet objet ne lance aucune analyse.
    Il contient uniquement les résultats produits par les
    différents moteurs de l'application.
    """

    project: Project

    analysis_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    analysis_date: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    version: str = "1.0"

    ai_act: AIActClassification = field(
        default_factory=AIActClassification
    )

    risks: list[Risk] = field(
        default_factory=list
    )

    recommendations: list[Recommendation] = field(
        default_factory=list
    )

    legal_references: list[LegalReference] = field(
        default_factory=list
    )

    conformity_status: str = "À déterminer"

    conclusion: str = ""

    deep_learning_result: dict[str, Any] = field(
        default_factory=dict
    )

    rag_result: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Transforme le rapport métier en dictionnaire JSON-compatible.
        """

        return asdict(self)

