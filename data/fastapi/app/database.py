import os

import psycopg

DB_NAME = os.getenv("POSTGRES_DB", "prudencia")
DB_USER = os.getenv("POSTGRES_USER", "prudencia")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")


def get_connection():
    return psycopg.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


def init_database():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prudencia.analyses (
                    id SERIAL PRIMARY KEY,
                    description TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()


def save_analysis(description: str) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prudencia.analyses "
                "(description) VALUES (%s) RETURNING id;",
                (description,)
            )
            analysis_id = cur.fetchone()[0]
        conn.commit()
        return analysis_id
