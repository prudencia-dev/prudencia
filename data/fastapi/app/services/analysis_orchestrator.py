from __future__ import annotations

from typing import Any

from app.services.report_builder import PrudenciaReportBuilder


class AnalysisOrchestrator:
    """
    Orchestre les différentes analyses de PRUDENCIA.

    Ce composant est le point d'entrée unique de l'analyse.

    Les moteurs Deep Learning et RAG restent indépendants.
    L'orchestrateur coordonne uniquement leur exécution puis
    construit le rapport final.
    """

    def __init__(self) -> None:
        self.report_builder = PrudenciaReportBuilder()

    def analyse(
        self,
        *,
        project: dict[str, Any],
        deep_learning_result: dict[str, Any] | None = None,
        rag_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Lance la construction du rapport PRUDENCIA.

        Pour le MVP, les résultats sont déjà produits par les
        différents moteurs. L'orchestrateur les centralise.
        """

        return self.report_builder.build(
            project=project,
            deep_learning_result=deep_learning_result,
            rag_result=rag_result,
        )
