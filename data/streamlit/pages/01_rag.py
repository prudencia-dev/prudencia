from __future__ import annotations

import os
from typing import Any

import streamlit as st

from services.api import PrudenciaAPI


COLLECTION_NAME = "prudencia_legal_documents"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
RAG_INDEX_TIMEOUT_SECONDS = int(
    os.getenv("RAG_INDEX_TIMEOUT_SECONDS", "1800")
)


st.set_page_config(
    page_title="Base vectorielle",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Base vectorielle")
st.caption(
    "Importation, indexation et recherche dans le corpus "
    "documentaire de PRUDENCIA."
)

# ==========================================================
# FONCTIONS
# ==========================================================

def load_rag_stats() -> dict[str, Any]:
    """
    Charge les statistiques générales du RAG.
    """

    return PrudenciaAPI.get("/rag/stats")


def load_rag_documents() -> list[dict[str, Any]]:
    """
    Charge les documents enregistrés dans PostgreSQL.
    """

    result = PrudenciaAPI.get("/rag/documents")
    return result.get("documents", [])


def format_file_size(size_bytes: int | None) -> str:
    """
    Convertit une taille en octets vers un format lisible.
    """

    if not size_bytes:
        return "Non disponible"

    size = float(size_bytes)

    for unit in ["octets", "Ko", "Mo", "Go"]:
        if size < 1024:
            return f"{size:.1f} {unit}"

        size /= 1024

    return f"{size:.1f} To"


def display_indexing_result(result: dict[str, Any]) -> None:
    """
    Affiche le rapport produit après une indexation RAG.
    """

    st.success(
        result.get(
            "message",
            "Document correctement indexé dans le RAG.",
        )
    )

    col_pages, col_chunks, col_embeddings, col_duration = st.columns(4)

    col_pages.metric(
        "Pages",
        result.get("pages", 0),
    )

    col_chunks.metric(
        "Chunks",
        result.get("chunks", 0),
    )

    col_embeddings.metric(
        "Embeddings",
        result.get("embeddings", 0),
    )

    col_duration.metric(
        "Durée",
        f"{result.get('duration_seconds', 0)} s",
    )

    st.write(
        f"**Document :** {result.get('document', 'Inconnu')}"
    )

    st.write(
        f"**Collection :** "
        f"`{result.get('collection', COLLECTION_NAME)}`"
    )

    st.write(
        f"**Modèle d'embedding :** "
        f"`{result.get('embedding_model', 'Non disponible')}`"
    )

    st.write(
        f"**Dimension de l'embedding :** "
        f"{result.get('embedding_dimension', 0)}"
    )

    with st.expander("Voir le rapport technique complet"):
        st.json(result)


# -------------------------------------------------------------------
# Dashboard RAG
# -------------------------------------------------------------------

try:
    stats = load_rag_stats()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "📄 Documents",
        stats.get("document_count", 0),
    )

    c2.metric(
        "🧩 Chunks",
        stats.get("chunk_count", 0),
    )

    c3.metric(
        "🧠 Embeddings",
        stats.get("embedding_count", 0),
    )

    c4.metric(
        "📚 Collection",
        stats.get(
            "collection",
            COLLECTION_NAME,
        ),
    )

except Exception:
    st.warning(
        "Impossible de charger les statistiques du RAG."
    )

# ==========================================================
# ONGLETS
# ==========================================================

documents_tab, collection_tab, search_tab, administration_tab = st.tabs(
    [
        "📄 Documents",
        "📊 Collection",
        "🔍 Recherche",
        "⚙️ Administration",
    ]
)

# ==========================================================
# DOCUMENTS
# ==========================================================

with documents_tab:
    st.subheader("Importer et indexer un document")

    st.info(
        "Le PDF sera automatiquement enregistré, extrait, "
        "découpé en chunks, transformé en embeddings puis "
        "indexé dans ChromaDB."
    )

    uploaded_file = st.file_uploader(
        "Sélectionner un fichier PDF",
        type=["pdf"],
        key="rag_pdf_uploader",
    )

    col_chunk_size, col_chunk_overlap = st.columns(2)

    with col_chunk_size:
        chunk_size = st.number_input(
            "Taille des chunks",
            min_value=100,
            max_value=5000,
            value=DEFAULT_CHUNK_SIZE,
            step=100,
            help=(
                "Nombre maximal de caractères contenus "
                "dans chaque chunk."
            ),
        )

    with col_chunk_overlap:
        chunk_overlap = st.number_input(
            "Chevauchement",
            min_value=0,
            max_value=2000,
            value=DEFAULT_CHUNK_OVERLAP,
            step=50,
            help=(
                "Nombre de caractères repris entre deux chunks."
            ),
        )

    parameters_are_valid = chunk_overlap < chunk_size

    if not parameters_are_valid:
        st.error(
            "Le chevauchement doit être strictement inférieur "
            "à la taille des chunks."
        )

    index_button = st.button(
        "Indexer le document",
        type="primary",
        disabled=(
            uploaded_file is None
            or not parameters_are_valid
        ),
        use_container_width=True,
    )

    if index_button and uploaded_file is not None:
        progress_bar = st.progress(0)
        progress_message = st.empty()

        try:
            progress_message.info("1/5 — Enregistrement du PDF")
            progress_bar.progress(10)

            progress_message.info(
                "2/5 — Extraction du texte et création des chunks"
            )
            progress_bar.progress(25)

            progress_message.info(
                "3/5 — Génération des embeddings"
            )
            progress_bar.progress(45)

            with st.spinner(
                "Indexation complète du document en cours..."
            ):
                result = PrudenciaAPI.post(
                    "/rag/index",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    data={
                        "chunk_size": str(chunk_size),
                        "chunk_overlap": str(chunk_overlap),
                        "collection_name": COLLECTION_NAME,
                    },
                    timeout=RAG_INDEX_TIMEOUT_SECONDS,
                )

            progress_message.info(
                "4/5 — Enregistrement dans ChromaDB"
            )
            progress_bar.progress(80)

            progress_message.info(
                "5/5 — Mise à jour de PostgreSQL"
            )
            progress_bar.progress(95)

            progress_bar.progress(100)
            progress_message.success(
                "Indexation terminée."
            )

            display_indexing_result(result)

        except Exception as error:
            progress_message.empty()
            progress_bar.empty()

            st.error(
                "Impossible d'indexer le document : "
                f"{error}"
            )

    st.divider()
    st.subheader("Documents enregistrés")

    col_refresh, col_information = st.columns(
        [1, 4]
    )

    with col_refresh:
        if st.button(
            "Actualiser",
            key="refresh_rag_documents",
            use_container_width=True,
        ):
            st.rerun()

    with col_information:
        st.caption(
            "La liste provient de la table "
            "`prudencia.documents`."
        )

    try:
        documents = load_rag_documents()

        if not documents:
            st.info(
                "Aucun document n'est actuellement enregistré."
            )

        else:
            for document in documents:
                filename = (
                    document.get("original_filename")
                    or document.get("stored_filename")
                    or "Document sans nom"
                )

                document_id = document.get(
                    "id",
                    "Non disponible",
                )

                extraction_status = document.get(
                    "extraction_status",
                    "Non disponible",
                )

                page_count = document.get(
                    "page_count",
                    0,
                )

                created_at = document.get(
                    "created_at",
                    "Non disponible",
                )

                with st.expander(filename):
                    col_id, col_status = st.columns(2)

                    with col_id:
                        st.write(
                            f"**Identifiant :** `{document_id}`"
                        )

                    with col_status:
                        st.write(
                            f"**Extraction :** `{extraction_status}`"
                        )

                    col_pages, col_size, col_date = st.columns(3)

                    col_pages.metric(
                        "Pages",
                        page_count or 0,
                    )

                    col_size.metric(
                        "Taille",
                        format_file_size(
                            document.get("file_size_bytes")
                        ),
                    )

                    col_date.write(
                        f"**Ajouté le :**  \n{created_at}"
                    )

                    st.write(
                        f"**Chemin :** "
                        f"`{document.get('file_path', 'Non disponible')}`"
                    )

                    st.divider()
                    if st.button(
                        "🗑️ Supprimer ce document",
                        key=f"delete_{document_id}",
                        use_container_width=True,
                    ):

                        try:

                            result = PrudenciaAPI.delete(
                                f"/rag/document/{document_id}"
                            )

                            st.success(
                                result.get(
                                    "message",
                                    "Document supprimé."
                                )
                            )

                            st.info(
                                f"{result.get('deleted_chunks',0)} chunks supprimés\n\n"
                                f"{result.get('deleted_embeddings',0)} embeddings supprimés"
                            )

                            st.rerun()

                        except Exception as error:

                            st.error(str(error))

    except Exception as error:
        st.error(
            "Impossible de charger les documents : "
            f"{error}"
        )


# ==========================================================
# COLLECTION
# ==========================================================

with collection_tab:
    st.subheader("État de la collection RAG")

    col_refresh_stats, col_stats_caption = st.columns(
        [1, 4]
    )

    with col_refresh_stats:
        if st.button(
            "Actualiser",
            key="refresh_rag_stats",
            use_container_width=True,
        ):
            st.rerun()

    with col_stats_caption:
        st.caption(
            "Les statistiques sont calculées depuis ChromaDB."
        )

    try:
        stats = load_rag_stats()

        col_documents, col_chunks, col_embeddings = st.columns(3)

        col_documents.metric(
            "Documents",
            stats.get("document_count", 0),
        )

        col_chunks.metric(
            "Chunks",
            stats.get("chunk_count", 0),
        )

        col_embeddings.metric(
            "Embeddings",
            stats.get("embedding_count", 0),
        )

        st.divider()

        col_collection, col_service, col_status = st.columns(3)

        with col_collection:
            st.write("**Collection active**")
            st.code(
                stats.get(
                    "collection",
                    COLLECTION_NAME,
                )
            )

        with col_service:
            st.write("**Service vectoriel**")
            st.code(
                stats.get(
                    "service",
                    "chromadb",
                )
            )

        with col_status:
            st.write("**Statut**")

            status = stats.get(
                "status",
                "inconnu",
            )

            if status == "ok":
                st.success("Opérationnel")
            elif status == "empty":
                st.warning("Collection vide")
            else:
                st.error(status)

        st.caption(
            "Dans PRUDENCIA, un chunk indexé correspond "
            "à un embedding stocké dans ChromaDB."
        )

    except Exception as error:
        st.error(
            "Impossible de récupérer les statistiques RAG : "
            f"{error}"
        )


# ==========================================================
# RECHERCHE
# ==========================================================

with search_tab:
    st.subheader("Rechercher dans le corpus juridique")

    query = st.text_area(
        "Question ou description du projet IA",
        placeholder=(
            "Exemple : Quelles obligations concernent "
            "un système d'IA utilisé pour le recrutement ?"
        ),
        height=150,
    )

    result_limit = st.slider(
        "Nombre de résultats",
        min_value=1,
        max_value=20,
        value=5,
    )

    search_button = st.button(
        "Lancer la recherche",
        type="primary",
        use_container_width=True,
    )

    if search_button:
        if not query.strip():
            st.warning(
                "Veuillez saisir une question ou une description."
            )

        else:
            try:
                with st.spinner(
                    "Recherche sémantique en cours..."
                ):
                    result = PrudenciaAPI.post(
                        "/rag/search",
                        json={
                            "query": query,
                            "collection_name": COLLECTION_NAME,
                            "limit": result_limit,
                        },
                    )
                    st.session_state["rag_result"] = result
                    st.session_state["rag_query"] = query

                    results = result.get("results", [])

                    st.info(
                        f"🔎 Requête : {query}"
                    )

                    st.caption(
                        f"{len(results)} résultat(s) trouvé(s)"
                    )

                    st.caption(
                        f"Question : {query}"
                    )

                if not results:
                    st.info(
                        "Aucun résultat pertinent n'a été trouvé."
                    )

                else:
                    st.success(
                        f"{len(results)} résultat(s) trouvé(s)."
                    )

                    for position, chunk in enumerate(
                        results,
                        start=1,
                    ):
                        metadata = chunk.get(
                            "metadata",
                            {},
                        )

                        similarity = chunk.get(
                            "similarity",
                            0,
                        )

                        score = similarity * 100

                        if score >= 90:
                            badge = "🟢"

                        elif score >= 75:
                            badge = "🟡"

                        else:
                            badge = "🟠"

                        title = (
                            f"{badge} Résultat {position} "
                            f"({score:.1f} %)"
                        )

                        with st.expander(
                            title,
                            expanded=position == 1,
                        ):
                            st.write(
                                chunk.get(
                                    "text",
                                    "",
                                )
                            )

                            st.divider()

                            col_document, col_chunk, col_model = st.columns(3)

                            col_document.write(
                                "**Document**  \n"
                                f"{metadata.get('filename', 'Inconnu')}"
                            )

                            col_chunk.write(
                                "**Chunk**  \n"
                                f"{metadata.get('chunk_index', 'N/A')}"
                            )

                            col_model.write(
                                "**Modèle**  \n"
                                f"{metadata.get('embedding_model', 'N/A')}"
                            )

                            st.caption(
                                f"Identifiant ChromaDB : "
                                f"{chunk.get('id', 'Non disponible')}"
                            )

            except Exception as error:
                st.error(
                    "Erreur pendant la recherche RAG : "
                    f"{error}"
                )


# ==========================================================
# ADMINISTRATION
# ==========================================================

with administration_tab:
    st.subheader("Administration du RAG")

    st.warning(
        "Les opérations ci-dessous sont irréversibles."
    )

    st.markdown("### Réinitialisation complète")

    st.write(
        "Cette opération supprimera :"
    )

    st.markdown(
        """
- tous les documents importés ;
- tous les chunks PostgreSQL ;
- tous les embeddings ChromaDB ;
- tous les fichiers PDF du dossier `/uploads`.
"""
    )

    confirmation = st.checkbox(
        "Je confirme vouloir supprimer l'intégralité de la base vectorielle.",
        key="confirm_reset_rag",
    )

    if st.button(
        "🗑️ Réinitialiser la base vectorielle",
        type="primary",
        disabled=not confirmation,
        use_container_width=True,
    ):

        try:
            with st.spinner(
                "Réinitialisation de la base vectorielle..."
            ):
                result = PrudenciaAPI.post(
                    "/rag/reset",
                )

            st.success(
                result.get(
                    "message",
                    "Base vectorielle réinitialisée."
                )
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Documents supprimés",
                result.get(
                    "deleted_documents",
                    0,
                ),
            )

            col2.metric(
                "Chunks supprimés",
                result.get(
                    "deleted_chunks",
                    0,
                ),
            )

            col3.metric(
                "PDF supprimés",
                result.get(
                    "deleted_files",
                    0,
                ),
            )

            st.info(
                "Actualisation de la page..."
            )

            st.rerun()

        except Exception as error:
            st.error(
                f"Erreur : {error}"
            )
