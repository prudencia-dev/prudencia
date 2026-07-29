from __future__ import annotations
from typing import Any
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from app.ai.camembert import (
    get_camembert_status,
    get_embedding as get_camembert_embedding,
)

from app.ai.bge_m3 import (
    get_embedding as get_rag_embedding,
)
from app.ai.rag import (
    DEFAULT_COLLECTION,
    add_chunk,
    add_chunks,
    check_chromadb,
    list_chunks,
    list_collections,
    search_chunks,
)
from app.database import init_database, save_analysis
from app.services.chunking_service import (
    create_chunks,
    split_text,
)
from app.services.document_service import (
    get_document_by_filename,
    list_documents,
    save_uploaded_file,
)
from app.services.pdf_service import extract_pdf_text
from app.api.model_registry import router as model_registry_router
from app.api.fine_tuning import router as fine_tuning_router
from app.api.rag import router as rag_router
from app.api.training_history import (
    router as training_history_router,
)
from app.api.ml import router as ml_router
from app.api.reports import router as reports_router
from app.api.document_analysis import (
    router as document_analysis_router,
)
from app.api.questionnaire_ml import (
    router as questionnaire_ml_router,
)


app = FastAPI(
    title="PRUDENCIA API",
    version="0.2.0",
)
app.include_router(model_registry_router)
app.include_router(fine_tuning_router)
app.include_router(rag_router)
app.include_router(training_history_router)
app.include_router(ml_router)
app.include_router(reports_router)
app.include_router(document_analysis_router)
app.include_router(questionnaire_ml_router)


class AnalyseRequest(BaseModel):
    description: str


class ChunkRequest(BaseModel):
    collection_name: str = DEFAULT_COLLECTION
    chunk_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagSearchRequest(BaseModel):
    query: str
    collection_name: str = DEFAULT_COLLECTION
    limit: int = Field(default=5, ge=1, le=20)


@app.on_event("startup")
def startup_event():
    init_database()


@app.get("/")
def root():
    return {
        "message": "Bienvenue sur PRUDENCIA",
        "version": "0.2.0",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyse")
def analyse(request: AnalyseRequest):
    analysis_id = save_analysis(request.description)

    return {
        "status": "success",
        "message": "Analyse enregistrée",
        "analysis_id": analysis_id,
        "description": request.description,
    }


# ---------------------------------------------------------
# CAMEMBERT
# ---------------------------------------------------------

@app.get("/ai/health")
def ai_health():
    return get_camembert_status()


@app.post("/ai/embedding")
def create_embedding(request: AnalyseRequest):
    try:
        embedding = get_camembert_embedding(request.description)

        return {
            "status": "success",
            "model": "camembert-base",
            "dimensions": len(embedding),
            "embedding_preview": embedding[:10],
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


# ---------------------------------------------------------
# CHROMADB / RAG
# ---------------------------------------------------------

@app.get("/rag/health")
def rag_health():
    return check_chromadb()


@app.get("/rag/collections")
def rag_collections():
    return {
        "collections": list_collections(),
    }


@app.get("/rag/collections/{collection_name}/chunks")
def rag_collection_chunks(
    collection_name: str,
    limit: int = 20,
):
    return list_chunks(
        collection_name=collection_name,
        limit=limit,
    )


@app.post("/rag/chunks")
def rag_add_chunk(request: ChunkRequest):
    try:
        embedding = get_camembert_embedding(
            request.text
        )

        metadata = {
            **request.metadata,
            "source_type": request.metadata.get(
                "source_type",
                "manual",
            ),
        }

        return add_chunk(
            collection_name=request.collection_name,
            chunk_id=request.chunk_id,
            text=request.text,
            embedding=embedding,
            metadata=metadata,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.post("/rag/search")
def rag_search(request: RagSearchRequest):
    try:
        query_embedding = get_rag_embedding(request.query)

        return {
            "status": "success",
            "query": request.query,
            **search_chunks(
                query_embedding=query_embedding,
                collection_name=request.collection_name,
                limit=request.limit,
            ),
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la recherche RAG : {error}",
        ) from error


# ---------------------------------------------------------
# DOCUMENTS
# ---------------------------------------------------------

@app.get("/documents")
def get_documents():
    return {
        "documents": list_documents(),
    }


@app.get("/documents/{filename}")
def get_document(filename: str):
    document = get_document_by_filename(filename)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document introuvable.",
        )

    return document


@app.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Seuls les fichiers PDF sont acceptés.",
        )

    return save_uploaded_file(file)


@app.get("/documents/{filename}/extract")
def extract_document_text(filename: str):
    try:
        extraction = extract_pdf_text(filename)

        return {
            "status": "success",
            **extraction,
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Impossible d'extraire le PDF : {error}",
        ) from error


@app.post("/documents/{filename}/chunks")
def generate_document_chunks(filename: str):
    try:
        document = get_document_by_filename(filename)

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document introuvable.",
            )

        extraction = extract_pdf_text(filename)
        text = extraction.get("text", "")

        if not text.strip():
            raise ValueError(
                "Le PDF ne contient pas de texte exploitable."
            )

        result = create_chunks(
            str(document["id"]),
            text,
        )

        return {
            "status": "success",
            "filename": filename,
            "document_id": str(document["id"]),
            **result,
        }

    except HTTPException:
        raise

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Impossible de créer les chunks : {error}",
        ) from error


@app.post("/documents/{filename}/index")
def index_document_in_chromadb(
    filename: str,
    collection_name: str = DEFAULT_COLLECTION,
):
    try:
        document = get_document_by_filename(filename)

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document introuvable.",
            )

        extraction = extract_pdf_text(filename)
        text = extraction.get("text", "")

        if not text.strip():
            raise ValueError(
                "Le PDF ne contient pas de texte exploitable."
            )

        chunks = split_text(
            text=text,
            chunk_size=1000,
            overlap=200,
        )

        if not chunks:
            raise ValueError(
                "Aucun chunk n'a pu être créé."
            )

        document_id = str(document["id"])

        ids = []
        embeddings = []
        metadatas = []

        for index, chunk in enumerate(chunks):
            chunk_id = f"{document_id}-{index}"

            ids.append(chunk_id)
            embeddings.append(
                get_rag_embedding(chunk)
            )
            metadatas.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": index,
                    "source_type": "pdf",
                    "embedding_model": "camembert-base",
                }
            )

        chroma_result = add_chunks(
            collection_name=collection_name,
            ids=ids,
            texts=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return {
            "status": "success",
            "message": "Document indexé dans ChromaDB.",
            "filename": filename,
            "document_id": document_id,
            **chroma_result,
        }

    except HTTPException:
        raise

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Impossible d'indexer le document : {error}",
        ) from error