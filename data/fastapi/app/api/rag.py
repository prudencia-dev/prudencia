from __future__ import annotations

from typing import Any

from app.ai.rag import DEFAULT_COLLECTION, get_rag_stats
from app.api.error_responses import raise_api_error
from app.services.document_service import list_documents
from app.services.rag_service import (
    delete_document,
    index_pdf_document,
    reset_rag_database,
)
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.get("/stats")
def rag_stats() -> dict[str, Any]:
    """Retourne les statistiques générales du RAG."""

    try:
        return get_rag_stats()
    except Exception as error:
        raise_api_error(
            operation="rag.stats",
            error=error,
            detail="Impossible de récupérer les statistiques RAG.",
        )


@router.get("/documents")
def rag_documents() -> dict[str, Any]:
    """Retourne les documents enregistrés dans PostgreSQL."""

    try:
        return {"documents": list_documents()}
    except Exception as error:
        raise_api_error(
            operation="rag.documents",
            error=error,
            detail="Impossible de récupérer les documents.",
        )


@router.post("/index")
def rag_index_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=200),
    collection_name: str = Form(default=DEFAULT_COLLECTION),
) -> dict[str, Any]:
    """Importe et indexe un PDF dans le RAG."""

    try:
        return index_pdf_document(
            upload_file=file,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            collection_name=collection_name,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except Exception as error:
        raise_api_error(
            operation="rag.index",
            error=error,
            detail="Impossible d'indexer le document.",
        )


@router.post("/reset")
def rag_reset() -> dict[str, Any]:
    """Réinitialise les données documentaires du RAG."""

    try:
        return reset_rag_database(collection_name=DEFAULT_COLLECTION)
    except Exception as error:
        raise_api_error(
            operation="rag.reset",
            error=error,
            detail="Impossible de réinitialiser le RAG.",
        )


@router.delete("/document/{document_id}")
def rag_delete_document(document_id: str) -> dict[str, Any]:
    """Supprime un document, ses chunks, embeddings et fichier PDF."""

    try:
        return delete_document(document_id=document_id)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except Exception as error:
        raise_api_error(
            operation="rag.delete_document",
            error=error,
            detail="Impossible de supprimer le document.",
        )
