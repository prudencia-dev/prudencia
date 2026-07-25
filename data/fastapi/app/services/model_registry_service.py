import os
from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row


class ModelRegistryService:
    """Gestion du registre des modèles PRUDENCIA."""

    @staticmethod
    def _get_connection() -> psycopg.Connection:
        return psycopg.connect(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            row_factory=dict_row,
        )

    @classmethod
    def list_models(cls) -> list[dict[str, Any]]:
        query = """
            SELECT
                id,
                name,
                version,
                task_type,
                base_model,
                model_path,
                status,
                is_active,
                accuracy,
                precision_score,
                recall_score,
                f1_score,
                validation_loss,
                training_duration_seconds,
                model_size_bytes,
                created_at,
                activated_at
            FROM prudencia.trained_models
            ORDER BY created_at DESC;
        """

        with cls._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchall()

    @classmethod
    def get_model(cls, model_id: UUID) -> dict[str, Any] | None:
        query = """
            SELECT *
            FROM prudencia.trained_models
            WHERE id = %s;
        """

        with cls._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (model_id,))
                return cursor.fetchone()

    @classmethod
    def activate_model(cls, model_id: UUID) -> dict[str, Any] | None:
        with cls._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, task_type, status
                    FROM prudencia.trained_models
                    WHERE id = %s
                    FOR UPDATE;
                    """,
                    (model_id,),
                )

                model = cursor.fetchone()

                if model is None:
                    return None

                if model["status"] != "available":
                    raise ValueError(
                        "Seul un modèle disponible peut être activé."
                    )

                cursor.execute(
                    """
                    UPDATE prudencia.trained_models
                    SET
                        is_active = FALSE,
                        activated_at = NULL
                    WHERE task_type = %s
                      AND is_active = TRUE;
                    """,
                    (model["task_type"],),
                )

                cursor.execute(
                    """
                    UPDATE prudencia.trained_models
                    SET
                        is_active = TRUE,
                        activated_at = NOW()
                    WHERE id = %s
                    RETURNING *;
                    """,
                    (model_id,),
                )

                result = cursor.fetchone()
                connection.commit()

                return result

    @classmethod
    def archive_model(cls, model_id: UUID) -> dict[str, Any] | None:
        query = """
            UPDATE prudencia.trained_models
            SET
                status = 'archived',
                is_active = FALSE,
                activated_at = NULL
            WHERE id = %s
            RETURNING *;
        """

        with cls._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (model_id,))
                result = cursor.fetchone()
                connection.commit()

                return result

    @classmethod
    def list_training_runs(cls) -> list[dict[str, Any]]:
        query = """
            SELECT
                id,
                model_id,
                run_name,
                task_type,
                base_model,
                dataset_name,
                dataset_version,
                total_examples,
                training_examples,
                validation_examples,
                number_of_classes,
                epochs,
                batch_size,
                learning_rate,
                validation_split,
                device,
                status,
                accuracy,
                precision_score,
                recall_score,
                f1_score,
                training_loss,
                validation_loss,
                training_duration_seconds,
                model_output_path,
                error_message,
                started_at,
                completed_at,
                created_at
            FROM prudencia.model_training_runs
            ORDER BY created_at DESC;
        """

        with cls._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchall()