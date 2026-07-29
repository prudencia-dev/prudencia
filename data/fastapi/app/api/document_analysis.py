from __future__ import annotations

import fitz
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

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Le fichier doit être un PDF.",
        )

    pdf_bytes = await file.read()

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="Le fichier PDF est vide.",
        )

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
            "filename": file.filename,
            "page_count": document.page_count,
            "character_count": len(extracted_text),
            "text": extracted_text,
        }

    finally:
        document.close()
