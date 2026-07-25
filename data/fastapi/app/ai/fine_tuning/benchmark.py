from __future__ import annotations

from typing import Any


class BenchmarkManager:
    """
    Compare plusieurs modèles fine-tunés.
    """

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def add_result(
        self,
        result: dict[str, Any],
    ) -> None:
        """
        Ajoute un résultat provenant du Trainer.
        """

        self.results.append(result)

    def compare(self) -> list[dict[str, Any]]:
        """
        Retourne les modèles triés par F1-score décroissant.
        """

        return sorted(
            self.results,
            key=lambda x: x["metrics"]["f1"],
            reverse=True,
        )

    def get_best_model(self) -> dict[str, Any]:
        """
        Retourne le meilleur modèle.
        """

        if not self.results:
            raise RuntimeError(
                "Aucun résultat disponible."
            )

        return self.compare()[0]

    def clear(self) -> None:
        """
        Réinitialise le benchmark.
        """

        self.results.clear()