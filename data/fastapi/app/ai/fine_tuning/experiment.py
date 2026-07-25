import json
import os
from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row


class ExperimentManager:
    """Mémorise les entraînements dans PostgreSQL."""

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
    def create_run(
        cls,
        *,
        run_name: str,
        base_model: str,
        dataset_name: str,
        text_column: str,
        label_column: str,
        total_examples: int,
        number_of_classes: int,
        classes: list[str],
        epochs: int,
        batch_size: int,
        learning_rate: float,
        validation_split: float,
        device: str,
        dataset_version: str | None = None,
        dataset_checksum_sha256: str | None = None,
    ) -> dict[str, Any]:
        query = """
            INSERT INTO prudencia.model_training_runs (
                run_name,
                task_type,
                base_model,
                dataset_name,
                dataset_version,
                dataset_checksum_sha256,
                text_column,
                label_column,
                total_examples,
                number_of_classes,
                classes,
                epochs,
                batch_size,
                learning_rate,
                validation_split,
                device,
                status,
                created_at
            )
            VALUES (
                %s,
                'ai_act_classification',
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s::jsonb,
                %s,
                %s,
                %s,
                %s,
                %s,
                'pending',
                NOW()
            )
            RETURNING *;
        """

        values = (
            run_name,
            base_model,
            dataset_name,
            dataset_version,
            dataset_checksum_sha256,
            text_column,
            label_column,
            total_examples,
            number_of_classes,
            json.dumps(classes),
            epochs,
            batch_size,
            learning_rate,
            validation_split,
            device,
        )

        with cls._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, values)
                run = cursor.fetchone()
                connection.commit()

        return run

    @classmethod
    def mark_running(
        cls,
        run_id: UUID,
        *,
        training_examples: int,
        validation_examples: int,
    ) -> dict[str, Any] | None:
        query = """
            UPDATE prudencia.model_training_runs
            SET
                status = 'running',
                training_examples = %s,
                validation_examples = %s,
                started_at = NOW()
            WHERE id = %s
            RETURNING *;
        """

        with cls._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        training_examples,
                        validation_examples,
                        run_id,
                    ),
                )
                run = cursor.fetchone()
                connection.commit()

        return run

    @classmethod
    def mark_completed(
        cls,
        run_id: UUID,
        *,
        accuracy: float,
        precision_score: float,
        recall_score: float,
        f1_score: float,
        training_loss: float | None,
        validation_loss: float | None,
        training_duration_seconds: int,
        model_output_path: str,
        logs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        query = """
            UPDATE prudencia.model_training_runs
            SET
                status = 'completed',
                accuracy = %s,
                precision_score = %s,
                recall_score = %s,
                f1_score = %s,
                training_loss = %s,
                validation_loss = %s,
                training_duration_seconds = %s,
                model_output_path = %s,
                logs = %s::jsonb,
                completed_at = NOW()
            WHERE id = %s
            RETURNING *;
        """

        with cls._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        accuracy,
                        precision_score,
                        recall_score,
                        f1_score,
                        training_loss,
                        validation_loss,
                        training_duration_seconds,
                        model_output_path,
                        json.dumps(logs or []),
                        run_id,
                    ),
                )
                run = cursor.fetchone()
                connection.commit()

        return run

    @classmethod
    def mark_failed(
        cls,
        run_id: UUID,
        *,
        error_message: str,
        logs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        query = """
            UPDATE prudencia.model_training_runs
            SET
                status = 'failed',
                error_message = %s,
                logs = %s::jsonb,
                completed_at = NOW()
            WHERE id = %s
            RETURNING *;
        """

        with cls._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        error_message,
                        json.dumps(logs or []),
                        run_id,
                    ),
                )
                run = cursor.fetchone()
                connection.commit()

        return run

    @classmethod
    def get_run(
        cls,
        run_id: UUID,
    ) -> dict[str, Any] | None:
        query = """
            SELECT *
            FROM prudencia.model_training_runs
            WHERE id = %s;
        """

        with cls._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (run_id,))
                return cursor.fetchone()

    @classmethod
    def list_runs(
        cls,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT *
            FROM prudencia.model_training_runs
            ORDER BY created_at DESC
            LIMIT %s;
        """

        with cls._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (limit,))
                return cursor.fetchall()