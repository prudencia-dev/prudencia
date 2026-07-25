from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from app.database import get_connection


class BaseRepository:
    """
    Classe de base de tous les repositories PostgreSQL.

    Elle centralise :
    - l'ouverture des connexions ;
    - les transactions ;
    - l'exécution des requêtes SQL.

    Les repositories métiers héritent de cette classe.
    """

    @contextmanager
    def connection(self) -> Generator[Any, None, None]:
        """
        Retourne une connexion PostgreSQL.
        """

        connection = get_connection()

        try:
            yield connection
            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    def execute(
        self,
        query: str,
        parameters: tuple[Any, ...] = (),
    ) -> None:
        """
        Exécute une requête sans retour.
        """

        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    parameters,
                )

    def fetch_one(
        self,
        query: str,
        parameters: tuple[Any, ...] = (),
    ) -> Any:
        """
        Retourne une seule ligne.
        """

        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    parameters,
                )

                return cursor.fetchone()

    def fetch_all(
        self,
        query: str,
        parameters: tuple[Any, ...] = (),
    ) -> list[Any]:
        """
        Retourne plusieurs lignes.
        """

        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    parameters,
                )

                return cursor.fetchall()

    def execute_returning(
        self,
        query: str,
        parameters: tuple[Any, ...] = (),
    ) -> Any:
        """
        Exécute une requête SQL avec RETURNING.
        """

        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    parameters,
                )

                return cursor.fetchone()