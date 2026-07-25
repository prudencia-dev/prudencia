from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ModelSelectionService:
    """
    Gère les modèles sélectionnés par PRUDENCIA.

    Deux moteurs sont configurables :

    - RAG :
      modèle utilisé pour produire les embeddings ;

    - Deep Learning :
      modèle utilisé pour analyser le champ libre.

    Le Machine Learning n'est pas configurable :
    Random Forest reste le modèle officiel du projet.
    """

    CONFIG_PATH = (
        Path("models")
        / "configuration"
        / "active_models.json"
    )

    # Modèles disponibles pour le RAG.
    #RAG_MODELS: dict[str, dict[str, str]] = {
    #    "camembert": {
    #        "id": "camembert",
    #        "name": "CamemBERT",
    #        "hf_id": "almanach/camembert-base",
    #        "description": (
    #            "Modèle français actuellement utilisé "
    #            "pour les embeddings du MVP."
    #        ),
    #    },
    #    "camembertv2": {
    #        "id": "camembertv2",
    #        "name": "CamemBERTv2",
    #        "hf_id": "almanach/camembertv2-base",
    #        "description": (
    #            "Version plus récente de CamemBERT."
    #        ),
    #    },
    #}
    RAG_MODELS: dict[str, dict[str, str]] = {
        "bge_m3": {
            "id": "bge_m3",
            "name": "BGE-M3",
            "hf_id": "BAAI/bge-m3",
            "description": (
                "Modèle multilingue spécialisé dans la création "
                "d'embeddings pour la recherche sémantique."
            ),
        },
    }

    # Modèles disponibles pour l'analyse textuelle.
    DEEP_LEARNING_MODELS: dict[str, dict[str, str]] = {
        "camembert": {
            "id": "camembert",
            "name": "CamemBERT",
            "hf_id": "almanach/camembert-base",
            "description": (
                "Transformer généraliste pré-entraîné "
                "sur du texte français."
            ),
        },
        "camembertv2": {
            "id": "camembertv2",
            "name": "CamemBERTv2",
            "hf_id": "almanach/camembertv2-base",
            "description": (
                "Version plus récente du modèle CamemBERT."
            ),
        },
        "juribert": {
            "id": "juribert",
            "name": "JuriBERT",
            "hf_id": "dascim/juribert-base",
            "description": (
                "Transformer spécialisé dans le langage juridique."
            ),
        },
    }

    DEFAULT_SELECTION = {
        #"rag_model": "camembert",
        "rag_model": "bge_m3",
        "deep_learning_model": "juribert",
    }

    def __init__(self) -> None:
        """
        Crée automatiquement le fichier de configuration
        s'il n'existe pas encore.
        """

        self.CONFIG_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.CONFIG_PATH.exists():
            self._write_configuration(
                self.DEFAULT_SELECTION.copy()
            )

    # ------------------------------------------------------------------
    # Lecture
    # ------------------------------------------------------------------

    def get_configuration(self) -> dict[str, Any]:
        """
        Retourne les modèles actuellement sélectionnés.
        """

        configuration = self._read_configuration()

        rag_model_id = configuration.get(
            "rag_model",
            self.DEFAULT_SELECTION["rag_model"],
        )

        deep_learning_model_id = configuration.get(
            "deep_learning_model",
            self.DEFAULT_SELECTION[
                "deep_learning_model"
            ],
        )

        # Sécurité en cas de fichier invalide ou ancien.
        if rag_model_id not in self.RAG_MODELS:
            rag_model_id = self.DEFAULT_SELECTION[
                "rag_model"
            ]

        if (
            deep_learning_model_id
            not in self.DEEP_LEARNING_MODELS
        ):
            deep_learning_model_id = (
                self.DEFAULT_SELECTION[
                    "deep_learning_model"
                ]
            )

        return {
            "rag": self.RAG_MODELS[
                rag_model_id
            ],
            "deep_learning": (
                self.DEEP_LEARNING_MODELS[
                    deep_learning_model_id
                ]
            ),
            "machine_learning": {
                "id": "random_forest",
                "name": "Random Forest",
                "description": (
                    "Modèle Machine Learning officiel "
                    "de PRUDENCIA."
                ),
                "selectable": False,
            },
        }

    def list_available_models(self) -> dict[str, Any]:
        """
        Retourne la liste des modèles sélectionnables.
        """

        return {
            "rag": list(
                self.RAG_MODELS.values()
            ),
            "deep_learning": list(
                self.DEEP_LEARNING_MODELS.values()
            ),
            "machine_learning": [
                {
                    "id": "random_forest",
                    "name": "Random Forest",
                    "selectable": False,
                }
            ],
        }

    # ------------------------------------------------------------------
    # Modification
    # ------------------------------------------------------------------

    def update_configuration(
        self,
        *,
        rag_model: str,
        deep_learning_model: str,
    ) -> dict[str, Any]:
        """
        Enregistre les modèles sélectionnés.

        Une erreur est levée lorsqu'un identifiant inconnu
        est fourni.
        """

        if rag_model not in self.RAG_MODELS:
            raise ValueError(
                f"Modèle RAG inconnu : {rag_model}"
            )

        if (
            deep_learning_model
            not in self.DEEP_LEARNING_MODELS
        ):
            raise ValueError(
                "Modèle Deep Learning inconnu : "
                f"{deep_learning_model}"
            )

        configuration = {
            "rag_model": rag_model,
            "deep_learning_model": (
                deep_learning_model
            ),
        }

        self._write_configuration(
            configuration
        )

        return self.get_configuration()

    # ------------------------------------------------------------------
    # Accès direct aux modèles actifs
    # ------------------------------------------------------------------

    def get_active_rag_model(
        self,
    ) -> dict[str, str]:
        """
        Retourne le modèle actif pour le RAG.
        """

        return self.get_configuration()["rag"]

    def get_active_deep_learning_model(
        self,
    ) -> dict[str, str]:
        """
        Retourne le modèle actif pour l'analyse textuelle.
        """

        return self.get_configuration()[
            "deep_learning"
        ]

    # ------------------------------------------------------------------
    # Fichier JSON interne
    # ------------------------------------------------------------------

    def _read_configuration(
        self,
    ) -> dict[str, Any]:
        """
        Lit le fichier de configuration interne.
        """

        try:
            with self.CONFIG_PATH.open(
                "r",
                encoding="utf-8",
            ) as configuration_file:
                data = json.load(
                    configuration_file
                )

            if isinstance(data, dict):
                return data

        except (
            OSError,
            json.JSONDecodeError,
        ):
            pass

        return self.DEFAULT_SELECTION.copy()

    def _write_configuration(
        self,
        configuration: dict[str, Any],
    ) -> None:
        """
        Enregistre la configuration dans un fichier JSON.
        """

        with self.CONFIG_PATH.open(
            "w",
            encoding="utf-8",
        ) as configuration_file:
            json.dump(
                configuration,
                configuration_file,
                ensure_ascii=False,
                indent=2,
            )