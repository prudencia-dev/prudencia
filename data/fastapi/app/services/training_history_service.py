from __future__ import annotations

import json
from typing import Any

from app.database import get_connection


def set_active_model(
    *,
    model_type: str,
    model_name: str,
    model_version: str,
) -> None:
    """
    Active une version et désactive les anciennes
    versions du même modèle.
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE prudencia.model_executions
                SET is_active = FALSE
                WHERE model_type = %s
                  AND model_name = %s;
                """,
                (
                    model_type,
                    model_name,
                ),
            )

            cursor.execute(
                """
                UPDATE prudencia.model_executions
                SET is_active = TRUE
                WHERE id = (
                    SELECT id
                    FROM prudencia.model_executions
                    WHERE model_type = %s
                    AND model_name = %s
                    AND model_version = %s
                    ORDER BY executed_at DESC
                    LIMIT 1
                );
                """,
                (
                    model_type,
                    model_name,
                    model_version,
                ),
            )

        connection.commit()

def save_training_execution(
    *,
    model_type: str,
    model_name: str,
    model_version: str,
    task_name: str,
    dataset_name: str,
    dataset_rows: int,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    execution_time_ms: int,
    success: bool,
    error_message: str | None = None,
    execution_type: str = "training",
) -> dict[str, Any]:
    """
    Enregistre un entraînement dans PostgreSQL.
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO prudencia.model_executions
                (
                    model_type,
                    model_name,
                    model_version,
                    task_name,
                    execution_type,
                    dataset_name,
                    dataset_rows,
                    input_data,
                    output_data,
                    execution_time_ms,
                    success,
                    error_message
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                RETURNING id, executed_at;
                """,
                (
                    model_type,
                    model_name,
                    model_version,
                    task_name,
                    execution_type,
                    dataset_name,
                    dataset_rows,
                    json.dumps(input_data),
                    json.dumps(output_data),
                    execution_time_ms,
                    success,
                    error_message,
                ),
            )
            execution_id, executed_at = cursor.fetchone()

        connection.commit()

        set_active_model(
            model_type=model_type,
            model_name=model_name,
            model_version=model_version,
        )

        return {
            "run_id": str(execution_id),
            "executed_at": executed_at.isoformat(),
        }

def get_training_history() -> list[dict]:
    """
    Retourne l'historique des entraînements.
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    model_type,
                    model_name,
                    model_version,
                    task_name,
                    execution_type,
                    dataset_name,
                    dataset_rows,
                    input_data,
                    output_data,
                    execution_time_ms,
                    is_active,
                    success,
                    executed_at
                FROM prudencia.model_executions
                ORDER BY executed_at DESC;
                """
            )

            columns = [
                column.name
                for column in cursor.description
            ]

            return [
                dict(zip(columns, row))
                for row in cursor.fetchall()
            ]
        
def save_model_reset(
    *,
    model_type: str,
    model_name: str,
    model_version: str,
    reason: str = "Retour au modèle de base",
) -> None:
    """
    Enregistre une réinitialisation de modèle.
    """

    save_training_execution(
        model_type=model_type,
        model_name=model_name,
        model_version=model_version,
        task_name="reset",
        execution_type="reset",
        dataset_name="",
        dataset_rows=0,
        input_data={
            "reason": reason,
        },
        output_data={},
        execution_time_ms=0,
        success=True,
    )
