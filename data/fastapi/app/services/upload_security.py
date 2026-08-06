"""Validation et sauvegarde sécurisées des fichiers téléversés."""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from fastapi import UploadFile

PDF_SIGNATURE = b"%PDF-"
DEFAULT_MAX_PDF_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_CSV_BYTES = 10 * 1024 * 1024
COPY_CHUNK_SIZE = 1024 * 1024

MAX_PDF_UPLOAD_BYTES = int(
    os.getenv("MAX_PDF_UPLOAD_BYTES", str(DEFAULT_MAX_PDF_BYTES))
)
MAX_CSV_UPLOAD_BYTES = int(
    os.getenv("MAX_CSV_UPLOAD_BYTES", str(DEFAULT_MAX_CSV_BYTES))
)


class UploadTooLargeError(ValueError):
    """Signale qu'un upload dépasse la limite configurée."""


def sanitized_filename(filename: str | None, expected_suffix: str) -> str:
    """Retourne un nom client sans composante de chemin et valide son extension."""
    if not filename:
        raise ValueError("Le fichier ne possède pas de nom.")

    safe_name = Path(filename).name.strip()
    if not safe_name or safe_name in {".", ".."}:
        raise ValueError("Le nom du fichier est invalide.")
    if Path(safe_name).suffix.lower() != expected_suffix.lower():
        raise ValueError(f"Le fichier doit avoir l'extension {expected_suffix}.")
    return safe_name


def build_stored_filename(original_filename: str, suffix: str) -> str:
    """Génère un nom interne non prédictible sans réutiliser le nom client."""
    del original_filename
    return f"{uuid4().hex}{suffix.lower()}"


def confined_path(directory: Path, filename: str) -> Path:
    """Construit un chemin dont la résolution reste strictement dans directory."""
    root = directory.resolve()
    candidate = (root / Path(filename).name).resolve()
    if candidate.parent != root:
        raise ValueError("Le chemin du fichier est invalide.")
    return candidate


def _validate_pdf_signature(prefix: bytes) -> None:
    if not prefix.startswith(PDF_SIGNATURE):
        raise ValueError("Le contenu du fichier ne correspond pas à un PDF valide.")


def copy_limited_upload(
    upload_file: UploadFile,
    destination: Path,
    *,
    max_bytes: int,
    require_pdf_signature: bool = False,
) -> int:
    """Copie un upload par blocs et interrompt l'écriture au-delà de max_bytes."""
    if max_bytes <= 0:
        raise ValueError("La taille maximale autorisée doit être positive.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    prefix = b""

    try:
        with destination.open("xb") as output:
            while chunk := upload_file.file.read(COPY_CHUNK_SIZE):
                if not prefix:
                    prefix = chunk[: len(PDF_SIGNATURE)]
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise UploadTooLargeError(
                        f"Le fichier dépasse la taille maximale autorisée "
                        f"de {max_bytes // (1024 * 1024)} Mo."
                    )
                output.write(chunk)

        if total_bytes == 0:
            raise ValueError("Le fichier téléversé est vide.")
        if require_pdf_signature:
            _validate_pdf_signature(prefix)
    except Exception:
        if destination.exists():
            destination.unlink()
        raise

    return total_bytes


def read_limited_file(
    file_object: BinaryIO,
    *,
    max_bytes: int,
    require_pdf_signature: bool = False,
) -> bytes:
    """Lit au plus max_bytes depuis un flux synchrone sans charger un gros fichier."""
    content = file_object.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise UploadTooLargeError(
            f"Le fichier dépasse la taille maximale autorisée "
            f"de {max_bytes // (1024 * 1024)} Mo."
        )
    if not content:
        raise ValueError("Le fichier téléversé est vide.")
    if require_pdf_signature:
        _validate_pdf_signature(content[: len(PDF_SIGNATURE)])
    return content


async def read_limited_upload(
    upload_file: UploadFile,
    *,
    max_bytes: int,
    require_pdf_signature: bool = False,
) -> bytes:
    """Lit un UploadFile asynchrone avec une limite stricte."""
    content = await upload_file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise UploadTooLargeError(
            f"Le fichier dépasse la taille maximale autorisée "
            f"de {max_bytes // (1024 * 1024)} Mo."
        )
    if not content:
        raise ValueError("Le fichier téléversé est vide.")
    if require_pdf_signature:
        _validate_pdf_signature(content[: len(PDF_SIGNATURE)])
    return content
