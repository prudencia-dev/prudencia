from __future__ import annotations

import fitz
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from psycopg.rows import dict_row

from app.database import get_connection


router = APIRouter(
    prefix="/document-analysis",
    tags=["Document Analysis"],
)


def get_or_create_project(
    project_name: str,
    description: str | None = None,
) -> str:
    """
    Retourne l'identifiant d'un projet existant portant le même nom,
    ou crée un nouveau projet documentaire.
    """

    clean_name = project_name.strip()

    if not clean_name:
        raise HTTPException(
            status_code=422,
            detail="Le nom du projet est obligatoire.",
        )

    try:
        with get_connection() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT id
                    FROM prudencia.projects
                    WHERE LOWER(name) = LOWER(%s)
                    ORDER BY created_at DESC
                    LIMIT 1;
                    """,
                    (clean_name,),
                )

                existing = cursor.fetchone()

                if existing is not None:
                    return str(existing["id"])

                cursor.execute(
                    """
                    INSERT INTO prudencia.projects (
                        name,
                        description,
                        status
                    )
                    VALUES (%s, %s, 'ready')
                    RETURNING id;
                    """,
                    (
                        clean_name,
                        description.strip()
                        if description
                        else None,
                    ),
                )

                created = cursor.fetchone()

            connection.commit()

        if created is None:
            raise RuntimeError(
                "La création du projet n'a retourné aucun identifiant."
            )

        return str(created["id"])

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur pendant la création ou la récupération "
                f"du projet documentaire : {error}"
            ),
        ) from error


@router.post("/extract")
async def extract_project_pdf(
    file: UploadFile = File(...),
    project_name: str = Form(...),
) -> dict:
    """
    Crée ou récupère le projet, puis extrait le texte du PDF.

    Le document client n'est ni découpé en chunks,
    ni indexé dans ChromaDB à cette étape.
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

        project_id = get_or_create_project(
            project_name=project_name,
            description=(
                f"Projet documentaire importé depuis le fichier "
                f"{file.filename or 'document.pdf'}."
            ),
        )

        return {
            "success": True,
            "project_id": project_id,
            "project_name": project_name.strip(),
            "filename": file.filename,
            "page_count": document.page_count,
            "character_count": len(extracted_text),
            "text": extracted_text,
        }

    finally:
        document.close()
