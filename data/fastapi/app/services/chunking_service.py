from typing import List

from app.database import get_connection


def split_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[str]:

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def count_words(text: str) -> int:
    return len(text.split()) if text else 0


def save_chunks(document_id: str, chunks: List[str]):

    with get_connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                DELETE FROM prudencia.document_chunks
                WHERE document_id = %s;
                """,
                (document_id,),
            )

            for index, chunk in enumerate(chunks):
                cur.execute(
                    """
                    INSERT INTO prudencia.document_chunks (
                        document_id,
                        chunk_index,
                        content,
                        word_count
                    )
                    VALUES (%s, %s, %s, %s);
                    """,
                    (
                        document_id,
                        index,
                        chunk,
                        count_words(chunk),
                    ),
                )

        conn.commit()


def create_chunks(document_id: str, text: str):
    chunks = split_text(text)

    save_chunks(document_id, chunks)

    return {
        "chunks_created": len(chunks)
    }