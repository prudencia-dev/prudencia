from __future__ import annotations

import os
from typing import Any

import chromadb
from app.ai.embedding_config import RAG_EMBEDDING_MODEL

CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
DEFAULT_COLLECTION = os.getenv(
    "CHROMA_COLLECTION",
    "prudencia_legal_documents",
)


def get_chroma_client():
    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
    )


def check_chromadb() -> dict:
    client = get_chroma_client()

    return {
        "status": "ok",
        "service": "chromadb",
        "host": CHROMA_HOST,
        "port": CHROMA_PORT,
        "heartbeat": client.heartbeat(),
    }


def list_collections() -> list[dict]:
    client = get_chroma_client()
    collections = client.list_collections()

    return [
        {
            "name": collection.name,
            "count": collection.count(),
        }
        for collection in collections
    ]


def get_or_create_collection(
    collection_name: str = DEFAULT_COLLECTION,
    *,
    embedding_dimension: int | None = None,
):
    client = get_chroma_client()

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": RAG_EMBEDDING_MODEL,
            "embedding_normalized": True,
            **(
                {"embedding_dimension": embedding_dimension}
                if embedding_dimension is not None
                else {}
            ),
        },
    )

    metadata = collection.metadata or {}
    stored_model = metadata.get("embedding_model")
    stored_dimension = metadata.get("embedding_dimension")

    collection_count = collection.count()
    if collection_count and stored_model is None:
        raise ValueError(
            "La collection existante ne déclare pas son modèle d'embedding. "
            "Réinitialisez-la avant de l'utiliser avec BGE-M3."
        )
    if stored_model not in {None, RAG_EMBEDDING_MODEL}:
        raise ValueError(
            f"La collection utilise {stored_model}, mais l'application utilise "
            f"{RAG_EMBEDDING_MODEL}."
        )
    if (
        embedding_dimension is not None
        and stored_dimension is not None
        and int(stored_dimension) != embedding_dimension
    ):
        raise ValueError(
            f"Dimension d'embedding incompatible : {stored_dimension} stockée, "
            f"{embedding_dimension} reçue."
        )

    if not collection_count and (
        stored_model is None
        or (embedding_dimension is not None and stored_dimension is None)
    ):
        collection.modify(
            metadata={
                **metadata,
                "hnsw:space": "cosine",
                "embedding_model": RAG_EMBEDDING_MODEL,
                "embedding_normalized": True,
                **(
                    {"embedding_dimension": embedding_dimension}
                    if embedding_dimension is not None
                    else {}
                ),
            }
        )

    return collection


def add_chunk(
    collection_name: str,
    chunk_id: str,
    text: str,
    embedding: list[float],
    metadata: dict[str, Any],
) -> dict:
    collection = get_or_create_collection(
        collection_name,
        embedding_dimension=len(embedding),
    )

    collection.upsert(
        ids=[chunk_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata],
    )

    return {
        "status": "success",
        "collection": collection_name,
        "chunk_id": chunk_id,
    }


def add_chunks(
    collection_name: str,
    ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> dict:
    if not ids:
        raise ValueError("Aucun chunk à indexer.")

    lengths = {len(ids), len(texts), len(embeddings), len(metadatas)}
    if len(lengths) != 1:
        raise ValueError(
            "Les identifiants, textes, embeddings et métadonnées doivent "
            "avoir la même longueur."
        )

    dimensions = {len(embedding) for embedding in embeddings}
    if len(dimensions) != 1:
        raise ValueError("Tous les embeddings doivent avoir la même dimension.")
    collection = get_or_create_collection(
        collection_name,
        embedding_dimension=dimensions.pop(),
    )

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return {
        "status": "success",
        "collection": collection_name,
        "chunks_indexed": len(ids),
        "collection_count": collection.count(),
    }


def search_chunks(
    query_embedding: list[float],
    collection_name: str = DEFAULT_COLLECTION,
    limit: int = 5,
) -> dict:
    collection = get_or_create_collection(
        collection_name,
        embedding_dimension=len(query_embedding),
    )

    if collection.count() == 0:
        return {
            "collection": collection_name,
            "count": 0,
            "results": [],
        }

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(limit, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    results = []

    for index, chunk_id in enumerate(ids):
        distance = distances[index]

        results.append(
            {
                "id": chunk_id,
                "text": documents[index],
                "metadata": metadatas[index],
                "distance": distance,
                "similarity": max(0.0, 1.0 - distance),
            }
        )

    return {
        "collection": collection_name,
        "count": len(results),
        "results": results,
    }


def list_chunks(
    collection_name: str = DEFAULT_COLLECTION,
    limit: int = 20,
) -> dict:
    collection = get_or_create_collection(collection_name)

    result = collection.get(
        limit=limit,
        include=[
            "documents",
            "metadatas",
        ],
    )

    chunks = []

    for index, chunk_id in enumerate(result["ids"]):
        chunks.append(
            {
                "id": chunk_id,
                "text": result["documents"][index],
                "metadata": result["metadatas"][index],
            }
        )

    return {
        "collection": collection_name,
        "count": collection.count(),
        "chunks": chunks,
    }

def get_rag_stats(
    collection_name: str = DEFAULT_COLLECTION,
) -> dict[str, Any]:
    """
    Retourne l'état général de la collection RAG.

    Un chunk stocké dans ChromaDB correspond à un embedding.
    """

    client = get_chroma_client()

    try:
        collection = client.get_collection(
            name=collection_name,
        )
    except Exception:
        return {
            "status": "empty",
            "service": "chromadb",
            "collection": collection_name,
            "document_count": 0,
            "chunk_count": 0,
            "embedding_count": 0,
        }

    collection_count = collection.count()

    document_names: set[str] = set()

    if collection_count > 0:
        result = collection.get(
            include=["metadatas"],
        )

        for metadata in result.get("metadatas", []):
            if not metadata:
                continue

            document_name = (
                metadata.get("filename")
                or metadata.get("document_name")
                or metadata.get("source")
                or metadata.get("original_filename")
            )

            if document_name:
                document_names.add(str(document_name))

    return {
        "status": "ok",
        "service": "chromadb",
        "collection": collection_name,
        "document_count": len(document_names),
        "chunk_count": collection_count,
        "embedding_count": collection_count,
    }


def delete_collection(
    collection_name: str = DEFAULT_COLLECTION,
) -> bool:
    """
    Supprime complètement une collection ChromaDB.
    """

    client = get_chroma_client()

    try:
        client.delete_collection(
            name=collection_name,
        )
        return True

    except Exception:
        return False

def reset_collection(
    collection_name: str = DEFAULT_COLLECTION,
) -> dict[str, Any]:
    """
    Supprime complètement une collection ChromaDB.

    Elle sera recréée automatiquement lors de la prochaine
    indexation avec la dimension du nouveau modèle d'embeddings.
    """

    deleted = delete_collection(collection_name)

    return {
        "status": "success",
        "collection": collection_name,
        "deleted": deleted,
    }

def delete_document_embeddings(
    document_id: str,
    collection_name: str = DEFAULT_COLLECTION,
) -> dict[str, Any]:
    """
    Supprime de ChromaDB tous les embeddings associés
    à un document précis.
    """

    if not document_id.strip():
        raise ValueError(
            "L'identifiant du document est obligatoire."
        )

    client = get_chroma_client()

    try:
        collection = client.get_collection(
            name=collection_name,
        )

    except Exception:
        return {
            "status": "success",
            "collection": collection_name,
            "document_id": document_id,
            "deleted_embeddings": 0,
            "message": (
                "La collection ChromaDB n'existe pas "
                "ou est déjà vide."
            ),
        }

    matching_chunks = collection.get(
        where={
            "document_id": document_id,
        },
        include=[],
    )

    chunk_ids = matching_chunks.get(
        "ids",
        [],
    )

    deleted_embeddings = len(chunk_ids)

    if chunk_ids:
        collection.delete(
            ids=chunk_ids,
        )

    return {
        "status": "success",
        "collection": collection_name,
        "document_id": document_id,
        "deleted_embeddings": deleted_embeddings,
        "message": (
            f"{deleted_embeddings} embedding(s) "
            "supprimé(s) de ChromaDB."
        ),
    }
