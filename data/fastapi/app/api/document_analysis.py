from __future__ import annotations

import fitz
from app.services.upload_security import (
    MAX_PDF_UPLOAD_BYTES,
    UploadTooLargeError,
    read_limited_upload,
    sanitized_filename,
)
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(
    prefix="/document-analysis",
    tags=["Document Analysis"],
)


@router.post("/extract")
async def extract_project_pdf(
    file: UploadFile = File(...),
) -> dict:
    """
    Extrait le texte d'un PDF décrivant un projet IA.

    Ce document client n'est ni découpé en chunks,
    ni indexé dans ChromaDB.
    """

    try:
        safe_filename = sanitized_filename(file.filename, ".pdf")
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    try:
        pdf_bytes = await read_limited_upload(
            file,
            max_bytes=MAX_PDF_UPLOAD_BYTES,
            require_pdf_signature=True,
        )
    except UploadTooLargeError as error:
        raise HTTPException(
            status_code=413,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    try:
        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf",
        )
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="Le fichier PDF est invalide ou illisible.",
        ) from error

    pages: list[str] = []

    try:
        for page in document:
            text = page.get_text("text")
            cleaned_text = " ".join(text.split())

            if cleaned_text:
                pages.append(cleaned_text)

        extracted_text = "\n\n".join(pages).strip()

        if not extracted_text:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Aucun texte exploitable n'a été trouvé. "
                    "Le PDF est peut-être constitué uniquement d'images."
                ),
            )

        return {
            "success": True,
            "filename": safe_filename,
            "page_count": document.page_count,
            "character_count": len(extracted_text),
            "text": extracted_text,
        }

    finally:
        document.close()
