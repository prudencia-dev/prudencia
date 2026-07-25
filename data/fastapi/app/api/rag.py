from __future__ import annotations

from typing import Any

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.ai.rag import (
    DEFAULT_COLLECTION,
    get_rag_stats,
)
from app.services.document_service import list_documents
from app.services.rag_service import (
    index_pdf_document,
    reset_rag_database,
    delete_document,
)


router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


@router.get("/stats")
def rag_stats() -> dict[str, Any]:
    """
    Retourne les statistiques générales du RAG.
    """

    try:
        return get_rag_stats()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur lors de la récupération "
                f"des statistiques RAG : {error}"
            ),
        ) from error


@router.get("/documents")
def rag_documents() -> dict[str, Any]:
    """
    Retourne les documents enregistrés dans PostgreSQL.
    """

    try:
        return {
            "documents": list_documents(),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur lors de la récupération "
                f"des documents : {error}"
            ),
        ) from error


@router.post("/index")
def rag_index_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=200),
    collection_name: str = Form(
        default=DEFAULT_COLLECTION,
    ),
) -> dict[str, Any]:
    """
    Importe et indexe un PDF dans le RAG.
    """

    try:
        return index_pdf_document(
            upload_file=file,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            collection_name=collection_name,
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
                "Erreur pendant l'indexation "
                f"du document : {error}"
            ),
        ) from error


@router.post("/reset")
def rag_reset() -> dict[str, Any]:
    """
    Réinitialise complètement la base documentaire RAG.

    Cette opération supprime :
    - les embeddings ChromaDB ;
    - les chunks PostgreSQL ;
    - les documents PostgreSQL ;
    - les fichiers PDF importés.
    """

    try:
        return reset_rag_database(
            collection_name=DEFAULT_COLLECTION,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur pendant la réinitialisation "
                f"du RAG : {error}"
            ),
        ) from error
    
@router.delete("/document/{document_id}")
def rag_delete_document(
    document_id: str,
) -> dict[str, Any]:
    """
    Supprime un document ainsi que ses chunks,
    ses embeddings et son fichier PDF.
    """

    try:
        return delete_document(
            document_id=document_id,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur pendant la suppression "
                f"du document : {error}"
            ),
        ) from error