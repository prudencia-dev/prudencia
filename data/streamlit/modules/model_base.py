from __future__ import annotations

from typing import Any

import streamlit as st

from services.api import PrudenciaAPI


def _get_status(endpoint: str) -> tuple[bool, dict[str, Any]]:
    """
    Interroge un endpoint FastAPI.
    """

    try:
        result = PrudenciaAPI.get(endpoint)

        if isinstance(result, dict):
            return True, result

        return True, {"response": result}

    except Exception as error:
        return False, {"error": str(error)}


def render_base_model() -> None:
    """
    Interface du moteur NLP principal.
    """

    st.subheader("Modèle de base")

    st.info(
        "Le moteur NLP est utilisé pour générer les embeddings "
        "et alimenter le moteur documentaire (RAG)."
    )

    available, status = _get_status("/ai/health")

    model_name = (
        status.get("model")
        or status.get("model_name")
        or "Non disponible"
    )

    dimensions = status.get(
        "dimensions",
        "—",
    )

    col_model, col_status, col_dimensions = st.columns(3)

    col_model.metric(
        "Modèle actif",
        model_name,
    )

    col_status.metric(
        "Statut",
        "🟢 Disponible"
        if available
        else "🔴 Indisponible",
    )

    col_dimensions.metric(
        "Dimensions",
        dimensions,
    )

    st.divider()

    st.write("### Test du moteur")

    text = st.text_area(
        "Texte à analyser",
        placeholder=(
            "Décrivez un projet d'intelligence artificielle..."
        ),
        height=140,
        key="base_model_text",
    )

    if st.button(
        "Calculer l'embedding",
        type="primary",
        use_container_width=True,
        key="base_model_embedding",
        disabled=not available,
    ):

        if not text.strip():
            st.warning(
                "Veuillez saisir un texte."
            )

        else:

            try:

                with st.spinner(
                    "Calcul de l'embedding..."
                ):

                    result = PrudenciaAPI.post(
                        "/ai/embedding",
                        json={
                            "description": text,
                        },
                    )

                st.success(
                    "Embedding calculé."
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Dimensions",
                    result.get(
                        "dimensions",
                        0,
                    ),
                )

                col2.metric(
                    "Modèle",
                    result.get(
                        "model",
                        model_name,
                    ),
                )

                with st.expander(
                    "Résultat technique"
                ):
                    st.json(result)

            except Exception as error:

                st.error(error)

    st.divider()

    st.write("### Description")

    st.write(
        """
Le moteur NLP est chargé de :

- comprendre les textes ;

- générer les embeddings ;

- alimenter ChromaDB ;

- permettre la recherche sémantique du RAG.

Le modèle utilisé est interchangeable.
L'interface de PRUDENCIA ne dépend jamais
du modèle concret.
"""
    )

    with st.expander(
        "Informations techniques"
    ):
        st.json(status)