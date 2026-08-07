
from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import UUID

from app.ai.bge_m3 import get_embeddings
from app.ai.embedding_config import RAG_EMBEDDING_MODEL
from app.ai.rag import (
    DEFAULT_COLLECTION,
    add_chunks,
    delete_document_embeddings,
    reset_collection,
)
from app.database import get_connection
from app.services.chunking_service import save_chunks, split_text
from app.services.document_service import save_uploaded_file
from app.services.pdf_service import extract_pdf_text
from app.services.upload_security import confined_path
from fastapi import UploadFile

DEFAULT_EMBEDDING_MODEL = RAG_EMBEDDING_MODEL
UPLOAD_DIR = Path(
    os.getenv("UPLOAD_DIR", "/uploads")
)

def validate_indexing_parameters(
    upload_file: UploadFile,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """
    Vérifie les paramètres reçus avant de démarrer l'indexation.
    """

    if not upload_file.filename:
        raise ValueError("Le fichier ne possède pas de nom.")

    if not upload_file.filename.lower().endswith(".pdf"):
        raise ValueError("Seuls les fichiers PDF sont acceptés.")

    if chunk_size < 100:
        raise ValueError(
            "La taille des chunks doit être supérieure ou égale à 100."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "Le chevauchement des chunks ne peut pas être négatif."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "Le chevauchement doit être strictement inférieur "
            "à la taille des chunks."
        )


def update_document_extraction(
    document_id: str,
    page_count: int,
    text_extracted: bool,
) -> None:
    """
    Met à jour les informations d'extraction du document.
    """

    extraction_status = (
        "completed"
        if text_extracted
        else "failed"
    )

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE prudencia.documents
                SET
                    extraction_status = %s,
                    text_extracted = %s,
                    page_count = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s;
                """,
                (
                    extraction_status,
                    text_extracted,
                    page_count,
                    document_id,
                ),
            )

        connection.commit()


def update_chunk_indexing_metadata(
    document_id: str,
    collection_name: str,
    chunk_ids: list[str],
    embeddings: list[list[float]],
    embedding_model: str,
) -> None:
    """
    Ajoute aux chunks PostgreSQL les identifiants ChromaDB
    et les informations relatives aux embeddings.
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            for chunk_index, chunk_id in enumerate(chunk_ids):
                embedding_dimension = len(embeddings[chunk_index])

                cursor.execute(
                    """
                    UPDATE prudencia.document_chunks
                    SET
                        chroma_collection = %s,
                        chroma_document_id = %s,
                        token_count = %s,
                        embedding_model = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE document_id = %s
                      AND chunk_index = %s;
                    """,
                    (
                        collection_name,
                        chunk_id,
                        embedding_dimension,
                        embedding_model,
                        document_id,
                        chunk_index,
                    ),
                )

        connection.commit()


def build_chunk_metadata(
    document_id: str,
    filename: str,
    chunks: list[str],
    collection_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_dimension: int,
) -> list[dict[str, Any]]:
    """
    Construit les métadonnées envoyées à ChromaDB.
    """

    metadata_list: list[dict[str, Any]] = []

    for chunk_index, _chunk in enumerate(chunks):
        metadata_list.append(
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": chunk_index,
                "collection_name": collection_name,
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
                "embedding_normalized": True,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "source_type": "pdf",
            }
        )

    return metadata_list


def index_pdf_document(
    upload_file: UploadFile,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    """
    Exécute l'intégralité du pipeline RAG :

    1. sauvegarde du PDF ;
    2. extraction du texte ;
    3. création des chunks ;
    4. génération des embeddings ;
    5. indexation dans ChromaDB ;
    6. sauvegarde des informations dans PostgreSQL.
    """

    validate_indexing_parameters(
        upload_file=upload_file,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if embedding_model != DEFAULT_EMBEDDING_MODEL:
        raise ValueError(
            f"Le modèle d'embedding doit être {DEFAULT_EMBEDDING_MODEL}."
        )

    start_time = perf_counter()

    # ------------------------------------------------------
    # 1. Sauvegarde du PDF
    # ------------------------------------------------------

    saved_document = save_uploaded_file(upload_file)

    document_id = saved_document["document_id"]
    filename = saved_document["filename"]
    stored_filename = saved_document["stored_filename"]

    # Vérifie que PostgreSQL a retourné un UUID valide.
    UUID(document_id)

    # ------------------------------------------------------
    # 2. Extraction du texte
    # ------------------------------------------------------

    extraction_result = extract_pdf_text(stored_filename)

    text = extraction_result.get("text", "")
    page_count = extraction_result.get("page_count", 0)
    text_extracted = extraction_result.get(
        "text_extracted",
        False,
    )

    update_document_extraction(
        document_id=document_id,
        page_count=page_count,
        text_extracted=text_extracted,
    )

    if not text_extracted or not text.strip():
        raise ValueError(
            "Aucun texte exploitable n'a été extrait du PDF. "
            "Le document est peut-être scanné."
        )

    # ------------------------------------------------------
    # 3. Création des chunks
    # ------------------------------------------------------

    chunks = split_text(
        text=text,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError(
            "Aucun chunk n'a pu être créé à partir du document."
        )

    save_chunks(
        document_id=document_id,
        chunks=chunks,
    )

    # ------------------------------------------------------
    # 4. Génération des embeddings BGE-M3
    # ------------------------------------------------------

    embeddings = get_embeddings(chunks)
    embedding_dimension = len(embeddings[0])

    # Un identifiant ChromaDB unique par chunk.
    chunk_ids = [
        f"{document_id}_{chunk_index}"
        for chunk_index in range(len(chunks))
    ]

    metadata_list = build_chunk_metadata(
        document_id=document_id,
        filename=filename,
        chunks=chunks,
        collection_name=collection_name,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_dimension=embedding_dimension,
    )

    # ------------------------------------------------------
    # 5. Indexation ChromaDB
    # ------------------------------------------------------

    chroma_result = add_chunks(
        collection_name=collection_name,
        ids=chunk_ids,
        texts=chunks,
        embeddings=embeddings,
        metadatas=metadata_list,
    )

    # ------------------------------------------------------
    # 6. Mise à jour des chunks dans PostgreSQL
    # ------------------------------------------------------

    update_chunk_indexing_metadata(
        document_id=document_id,
        collection_name=collection_name,
        chunk_ids=chunk_ids,
        embeddings=embeddings,
        embedding_model=embedding_model,
    )

    duration_seconds = round(
        perf_counter() - start_time,
        2,
    )

    # ------------------------------------------------------
    # 7. Rapport final
    # ------------------------------------------------------

    return {
        "status": "success",
        "message": "Document correctement indexé dans le RAG.",
        "document_id": document_id,
        "document": filename,
        "stored_filename": stored_filename,
        "collection": collection_name,
        "embedding_model": embedding_model,
        "pages": page_count,
        "characters": extraction_result.get(
            "characters",
            0,
        ),
        "chunks": len(chunks),
        "embeddings": len(embeddings),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_dimension": (
            embedding_dimension
        ),
        "collection_count": chroma_result.get(
            "collection_count",
            0,
        ),
        "duration_seconds": duration_seconds,
    }

def reset_rag_database(
    collection_name: str = DEFAULT_COLLECTION,
) -> dict[str, Any]:
    """
    Réinitialise complètement la base vectorielle RAG.

    Cette opération supprime :
    1. la collection ChromaDB ;
    2. les chunks PostgreSQL ;
    3. les documents PostgreSQL ;
    4. les fichiers PDF importés.
    """

    start_time = perf_counter()

    # ------------------------------------------------------
    # 1. Comptage des données PostgreSQL avant suppression
    # ------------------------------------------------------

    deleted_chunks = 0
    deleted_documents = 0

    with get_connection() as connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM prudencia.document_chunks;
                    """
                )
                deleted_chunks = cursor.fetchone()[0]

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM prudencia.documents;
                    """
                )
                deleted_documents = cursor.fetchone()[0]

                # Les chunks doivent être supprimés avant
                # les documents à cause de la clé étrangère.
                cursor.execute(
                    """
                    DELETE FROM prudencia.document_chunks;
                    """
                )

                cursor.execute(
                    """
                    DELETE FROM prudencia.documents;
                    """
                )

            connection.commit()

        except Exception:
            connection.rollback()
            raise

    # ------------------------------------------------------
    # 2. Réinitialisation de la collection ChromaDB
    # ------------------------------------------------------

    chroma_result = reset_collection(
        collection_name=collection_name,
    )

    # ------------------------------------------------------
    # 3. Suppression des fichiers PDF
    # ------------------------------------------------------

    deleted_files = 0
    failed_files: list[str] = []

    if UPLOAD_DIR.exists():
        for file_path in UPLOAD_DIR.iterdir():
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() != ".pdf":
                continue

            try:
                file_path.unlink()
                deleted_files += 1

            except OSError:
                failed_files.append(file_path.name)

    duration_seconds = round(
        perf_counter() - start_time,
        2,
    )

    # ------------------------------------------------------
    # 4. Rapport final
    # ------------------------------------------------------

    return {
        "status": "success",
        "message": "Base vectorielle RAG réinitialisée.",
        "collection": collection_name,
        "collection_reset": True,
        "collection_existed": chroma_result.get(
            "deleted",
            False,
        ),
        "deleted_chunks": deleted_chunks,
        "deleted_documents": deleted_documents,
        "deleted_files": deleted_files,
        "failed_files": failed_files,
        "duration_seconds": duration_seconds,
    }

def delete_document(
    document_id: str,
    collection_name: str = DEFAULT_COLLECTION,
) -> dict[str, Any]:
    """
    Supprime un document ainsi que toutes les données associées.
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    file_path,
                    original_filename
                FROM prudencia.documents
                WHERE id = %s;
                """,
                (document_id,),
            )

            document = cursor.fetchone()

            if document is None:
                raise ValueError(
                    "Document introuvable."
                )

            file_path, filename = document

            embedding_result = delete_document_embeddings(
                document_id=document_id,
                collection_name=collection_name,
            )

            cursor.execute(
                """
                DELETE FROM prudencia.document_chunks
                WHERE document_id = %s;
                """,
                (document_id,),
            )

            deleted_chunks = cursor.rowcount

            cursor.execute(
                """
                DELETE FROM prudencia.documents
                WHERE id = %s;
                """,
                (document_id,),
            )

            connection.commit()

    deleted_file = False

    if file_path:
        safe_file_path = confined_path(
            UPLOAD_DIR,
            Path(file_path).name,
        )
        if safe_file_path.is_file():
            safe_file_path.unlink()
            deleted_file = True

    return {
        "status": "success",
        "message": "Document supprimé.",
        "document": filename,
        "deleted_chunks": deleted_chunks,
        "deleted_embeddings": embedding_result.get(
            "deleted_embeddings",
            0,
        ),
        "deleted_file": deleted_file,
    }
