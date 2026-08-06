from pathlib import Path

import pymupdf

UPLOAD_DIR = Path("/uploads")


def get_safe_pdf_path(filename: str) -> Path:
    """
    Retourne le chemin sécurisé d'un PDF déjà téléversé.
    Empêche d'accéder à un fichier situé hors du dossier /uploads.
    """
    safe_filename = Path(filename).name
    file_path = UPLOAD_DIR / safe_filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Le document '{safe_filename}' est introuvable."
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Le document demandé n'est pas un fichier PDF.")

    return file_path


def extract_pdf_text(filename: str) -> dict:
    """
    Extrait le texte d'un PDF, page par page.
    """
    file_path = get_safe_pdf_path(filename)

    pages = []
    full_text_parts = []

    with pymupdf.open(file_path) as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text", sort=True).strip()

            pages.append(
                {
                    "page_number": page_number,
                    "characters": len(page_text),
                    "text": page_text,
                }
            )

            if page_text:
                full_text_parts.append(page_text)

        metadata = document.metadata or {}

    full_text = "\n\n".join(full_text_parts)

    return {
        "filename": file_path.name,
        "page_count": len(pages),
        "characters": len(full_text),
        "text_extracted": bool(full_text.strip()),
        "possible_scanned_pdf": not bool(full_text.strip()),
        "metadata": {
            "title": metadata.get("title") or "",
            "author": metadata.get("author") or "",
            "subject": metadata.get("subject") or "",
        },
        "pages": pages,
        "text": full_text,
    }
