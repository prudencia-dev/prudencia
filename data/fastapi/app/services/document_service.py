import hashlib
import shutil
from pathlib import Path

from app.database import get_connection

UPLOAD_DIR = Path("/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(8192), b""):
            sha256.update(block)

    return sha256.hexdigest()


def save_uploaded_file(upload_file):
    destination = UPLOAD_DIR / upload_file.filename

    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    file_size = destination.stat().st_size
    checksum = calculate_sha256(destination)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO prudencia.documents (
                    original_filename,
                    stored_filename,
                    file_path,
                    mime_type,
                    file_size_bytes,
                    checksum_sha256
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    upload_file.filename,
                    upload_file.filename,
                    str(destination),
                    upload_file.content_type,
                    file_size,
                    checksum,
                ),
            )

            document_id = cur.fetchone()[0]

        conn.commit()

    return {
        "status": "success",
        "document_id": str(document_id),
        "filename": upload_file.filename,
        "path": str(destination),
        "size": file_size,
    }


def list_documents():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    original_filename,
                    stored_filename,
                    file_path,
                    document_type,
                    file_size_bytes,
                    extraction_status,
                    text_extracted,
                    page_count,
                    created_at
                FROM prudencia.documents
                ORDER BY created_at DESC;
                """
            )

            rows = cur.fetchall()

    return [
        {
            "id": str(row[0]),
            "original_filename": row[1],
            "stored_filename": row[2],
            "file_path": row[3],
            "document_type": row[4],
            "file_size_bytes": row[5],
            "extraction_status": row[6],
            "text_extracted": row[7],
            "page_count": row[8],
            "created_at": row[9],
        }
        for row in rows
    ]


def get_document_by_filename(filename: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    original_filename,
                    stored_filename,
                    file_path
                FROM prudencia.documents
                WHERE original_filename = %s
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                (filename,),
            )

            row = cur.fetchone()

    if row is None:
        return None

    return {
        "id": str(row[0]),
        "original_filename": row[1],
        "stored_filename": row[2],
        "file_path": row[3],
    }