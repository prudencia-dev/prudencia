from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb

from app.database import get_connection


class AnalysisHistoryService:
    """
    Service d'historisation des analyses PRUDENCIA.

    Ce service enregistre :

    - les analyses documentaires ;
    - les analyses issues du questionnaire ;
    - les exécutions Machine Learning ;
    - les exécutions Deep Learning ;
    - les résultats RAG ;
    - les prédictions brutes ;
    - les rapports finaux ;
    - les éventuelles incohérences entre les résultats.

    Les données sont enregistrées dans :

    - prudencia.analyses ;
    - prudencia.model_executions.
    """

    VALID_MODEL_TYPES = {
        "machine_learning",
        "deep_learning",
        "embedding",
        "rag",
        "rule_engine",
    }

    VALID_ANALYSIS_STATUSES = {
        "pending",
        "running",
        "completed",
        "failed",
        "cancelled",
    }

    def start_analysis(
        self,
        *,
        project_id: str | UUID,
        analysis_type: str,
        questionnaire_response_id: str | UUID | None = None,
        requested_by: str | UUID | None = None,
        input_data: dict[str, Any] | None = None,
        source_name: str | None = None,
    ) -> str:
        """
        Crée une analyse avec le statut ``running``.

        Le type d'analyse et les données d'entrée sont conservés dès le début
        dans ``result_json`` afin de pouvoir diagnostiquer une erreur survenue
        avant la génération du rapport.
        """

        normalized_project_id = self._to_uuid(project_id, "project_id")
        normalized_response_id = self._to_optional_uuid(
            questionnaire_response_id,
            "questionnaire_response_id",
        )
        normalized_requested_by = self._to_optional_uuid(
            requested_by,
            "requested_by",
        )

        initial_result = {
            "schema_version": "1.0",
            "analysis_type": self._normalize_analysis_type(analysis_type),
            "source_name": source_name,
            "input_data": input_data or {},
            "history": {
                "started_at": self._utc_now_iso(),
            },
        }

        query = """
            INSERT INTO prudencia.analyses (
                project_id,
                questionnaire_response_id,
                requested_by,
                status,
                result_json,
                started_at
            )
            VALUES (
                %s,
                %s,
                %s,
                'running',
                %s,
                CURRENT_TIMESTAMP
            )
            RETURNING id;
        """

        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        normalized_project_id,
                        normalized_response_id,
                        normalized_requested_by,
                        Jsonb(initial_result),
                    ),
                )
                row = cursor.fetchone()

            connection.commit()

        if row is None:
            raise RuntimeError(
                "La création de l'analyse n'a retourné aucun identifiant."
            )

        return str(row[0])

    def complete_analysis(
        self,
        *,
        analysis_id: str | UUID,
        report: dict[str, Any],
        final_prediction: str | None = None,
        confidence: float | None = None,
        overall_risk_score: float | None = None,
        summary: str | None = None,
        raw_predictions: dict[str, Any] | None = None,
        consistency: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Termine une analyse avec le statut ``completed``.

        Le rapport complet, les prédictions brutes et le contrôle de cohérence
        sont conservés dans ``result_json``.
        """

        normalized_analysis_id = self._to_uuid(
            analysis_id,
            "analysis_id",
        )

        normalized_confidence = self._normalize_confidence(confidence)
        normalized_risk_score = self._normalize_risk_score(
            overall_risk_score
        )

        current_result = self.get_result_json(normalized_analysis_id)

        completed_at = self._utc_now_iso()

        result_json = {
            **current_result,
            "final_prediction": final_prediction,
            "confidence": normalized_confidence,
            "raw_predictions": raw_predictions or {},
            "consistency": consistency or {
                "status": "not_checked",
                "message": (
                    "Le contrôle de cohérence n'a pas encore été exécuté."
                ),
            },
            "metadata": metadata or {},
            "report": report,
            "history": {
                **current_result.get("history", {}),
                "completed_at": completed_at,
            },
        }

        query = """
            UPDATE prudencia.analyses
            SET
                status = 'completed',
                overall_risk_score = %s,
                confidence_score = %s,
                summary = %s,
                result_json = %s,
                error_message = NULL,
                completed_at = CURRENT_TIMESTAMP
            WHERE id = %s;
        """

        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        normalized_risk_score,
                        normalized_confidence,
                        summary,
                        Jsonb(result_json),
                        normalized_analysis_id,
                    ),
                )

                if cursor.rowcount == 0:
                    raise ValueError(
                        "Impossible de terminer l'analyse : "
                        f"identifiant introuvable {analysis_id}."
                    )

            connection.commit()

    def fail_analysis(
        self,
        *,
        analysis_id: str | UUID,
        error_message: str,
        error_context: dict[str, Any] | None = None,
    ) -> None:
        """
        Termine une analyse avec le statut ``failed``.
        """

        normalized_analysis_id = self._to_uuid(
            analysis_id,
            "analysis_id",
        )

        current_result = self.get_result_json(normalized_analysis_id)

        result_json = {
            **current_result,
            "error": {
                "message": error_message,
                "context": error_context or {},
                "occurred_at": self._utc_now_iso(),
            },
            "history": {
                **current_result.get("history", {}),
                "completed_at": self._utc_now_iso(),
            },
        }

        query = """
            UPDATE prudencia.analyses
            SET
                status = 'failed',
                result_json = %s,
                error_message = %s,
                completed_at = CURRENT_TIMESTAMP
            WHERE id = %s;
        """

        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        Jsonb(result_json),
                        error_message,
                        normalized_analysis_id,
                    ),
                )

                if cursor.rowcount == 0:
                    raise ValueError(
                        "Impossible de marquer l'analyse en erreur : "
                        f"identifiant introuvable {analysis_id}."
                    )

            connection.commit()

    def save_model_execution(
        self,
        *,
        analysis_id: str | UUID | None,
        model_type: str,
        model_name: str,
        input_data: dict[str, Any] | None,
        output_data: dict[str, Any] | None,
        execution_time_ms: int | None = None,
        model_version: str | None = None,
        task_name: str | None = None,
        success: bool = True,
        error_message: str | None = None,
        execution_type: str = "inference",
        dataset_name: str | None = None,
        dataset_rows: int | None = None,
        is_active: bool = False,
    ) -> str:
        """
        Enregistre une exécution de modèle.

        Pour les rapports, ``execution_type`` doit normalement être
        ``inference``.
        """

        normalized_model_type = model_type.strip().lower()

        if normalized_model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                "Type de modèle invalide : "
                f"{model_type}. Valeurs autorisées : "
                f"{sorted(self.VALID_MODEL_TYPES)}."
            )

        if not model_name or not model_name.strip():
            raise ValueError("Le nom du modèle est obligatoire.")

        normalized_analysis_id = self._to_optional_uuid(
            analysis_id,
            "analysis_id",
        )

        normalized_execution_time = None

        if execution_time_ms is not None:
            normalized_execution_time = max(
                0,
                int(execution_time_ms),
            )

        query = """
            INSERT INTO prudencia.model_executions (
                analysis_id,
                model_type,
                model_name,
                model_version,
                task_name,
                input_data,
                output_data,
                execution_time_ms,
                success,
                error_message,
                dataset_rows,
                dataset_name,
                execution_type,
                is_active
            )
            VALUES (
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
                %s,
                %s,
                %s
            )
            RETURNING id;
        """

        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        normalized_analysis_id,
                        normalized_model_type,
                        model_name.strip(),
                        model_version,
                        task_name,
                        Jsonb(input_data or {}),
                        Jsonb(output_data or {}),
                        normalized_execution_time,
                        success,
                        error_message,
                        dataset_rows,
                        dataset_name,
                        execution_type,
                        is_active,
                    ),
                )
                row = cursor.fetchone()

            connection.commit()

        if row is None:
            raise RuntimeError(
                "L'enregistrement de l'exécution du modèle "
                "n'a retourné aucun identifiant."
            )

        return str(row[0])

    def get_analysis(
        self,
        analysis_id: str | UUID,
    ) -> dict[str, Any] | None:
        """
        Retourne une analyse et ses exécutions de modèles.
        """

        normalized_analysis_id = self._to_uuid(
            analysis_id,
            "analysis_id",
        )

        analysis_query = """
            SELECT
                a.id,
                a.project_id,
                a.questionnaire_response_id,
                a.requested_by,
                a.status,
                a.overall_risk_score,
                a.confidence_score,
                a.summary,
                a.result_json,
                a.error_message,
                a.started_at,
                a.completed_at,
                a.created_at,
                p.name AS project_name
            FROM prudencia.analyses AS a
            INNER JOIN prudencia.projects AS p
                ON p.id = a.project_id
            WHERE a.id = %s;
        """

        executions_query = """
            SELECT
                id,
                model_type,
                model_name,
                model_version,
                task_name,
                input_data,
                output_data,
                execution_time_ms,
                success,
                error_message,
                executed_at,
                execution_type
            FROM prudencia.model_executions
            WHERE analysis_id = %s
            ORDER BY executed_at ASC;
        """

        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    analysis_query,
                    (normalized_analysis_id,),
                )
                analysis_row = cursor.fetchone()

                if analysis_row is None:
                    return None

                cursor.execute(
                    executions_query,
                    (normalized_analysis_id,),
                )
                execution_rows = cursor.fetchall()

        return {
            "id": str(analysis_row[0]),
            "project_id": str(analysis_row[1]),
            "questionnaire_response_id": (
                str(analysis_row[2])
                if analysis_row[2] is not None
                else None
            ),
            "requested_by": (
                str(analysis_row[3])
                if analysis_row[3] is not None
                else None
            ),
            "status": analysis_row[4],
            "overall_risk_score": self._to_float(analysis_row[5]),
            "confidence_score": self._to_float(analysis_row[6]),
            "summary": analysis_row[7],
            "result_json": analysis_row[8] or {},
            "error_message": analysis_row[9],
            "started_at": self._datetime_to_iso(analysis_row[10]),
            "completed_at": self._datetime_to_iso(analysis_row[11]),
            "created_at": self._datetime_to_iso(analysis_row[12]),
            "project_name": analysis_row[13],
            "model_executions": [
                {
                    "id": str(row[0]),
                    "model_type": row[1],
                    "model_name": row[2],
                    "model_version": row[3],
                    "task_name": row[4],
                    "input_data": row[5] or {},
                    "output_data": row[6] or {},
                    "execution_time_ms": row[7],
                    "success": row[8],
                    "error_message": row[9],
                    "executed_at": self._datetime_to_iso(row[10]),
                    "execution_type": row[11],
                }
                for row in execution_rows
            ],
        }

    def list_analyses(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        analysis_type: str | None = None,
        status: str | None = None,
        project_id: str | UUID | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retourne l'historique des analyses.

        Le type d'analyse est lu depuis ``result_json.analysis_type``.
        """

        normalized_limit = min(max(int(limit), 1), 500)
        normalized_offset = max(int(offset), 0)

        conditions: list[str] = []
        parameters: list[Any] = []

        if analysis_type:
            conditions.append(
                "a.result_json ->> 'analysis_type' = %s"
            )
            parameters.append(
                self._normalize_analysis_type(analysis_type)
            )

        if status:
            normalized_status = status.strip().lower()

            if normalized_status not in self.VALID_ANALYSIS_STATUSES:
                raise ValueError(
                    f"Statut d'analyse invalide : {status}."
                )

            conditions.append("a.status = %s")
            parameters.append(normalized_status)

        if project_id:
            conditions.append("a.project_id = %s")
            parameters.append(
                self._to_uuid(project_id, "project_id")
            )

        where_clause = ""

        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT
                a.id,
                a.project_id,
                p.name AS project_name,
                a.questionnaire_response_id,
                a.status,
                a.overall_risk_score,
                a.confidence_score,
                a.summary,
                a.result_json,
                a.error_message,
                a.started_at,
                a.completed_at,
                a.created_at
            FROM prudencia.analyses AS a
            INNER JOIN prudencia.projects AS p
                ON p.id = a.project_id
            {where_clause}
            ORDER BY a.created_at DESC
            LIMIT %s
            OFFSET %s;
        """

        parameters.extend(
            [
                normalized_limit,
                normalized_offset,
            ]
        )

        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(parameters))
                rows = cursor.fetchall()

        analyses: list[dict[str, Any]] = []

        for row in rows:
            result_json = row[8] or {}

            analyses.append(
                {
                    "id": str(row[0]),
                    "project_id": str(row[1]),
                    "project_name": row[2],
                    "questionnaire_response_id": (
                        str(row[3])
                        if row[3] is not None
                        else None
                    ),
                    "analysis_type": result_json.get(
                        "analysis_type",
                        "non_renseigne",
                    ),
                    "source_name": result_json.get("source_name"),
                    "status": row[4],
                    "overall_risk_score": self._to_float(row[5]),
                    "confidence_score": self._to_float(row[6]),
                    "summary": row[7],
                    "final_prediction": result_json.get(
                        "final_prediction"
                    ),
                    "consistency": result_json.get(
                        "consistency",
                        {
                            "status": "not_checked",
                            "message": (
                                "Contrôle de cohérence non disponible."
                            ),
                        },
                    ),
                    "error_message": row[9],
                    "started_at": self._datetime_to_iso(row[10]),
                    "completed_at": self._datetime_to_iso(row[11]),
                    "created_at": self._datetime_to_iso(row[12]),
                }
            )

        return analyses

    def get_result_json(
        self,
        analysis_id: str | UUID,
    ) -> dict[str, Any]:
        """
        Retourne uniquement ``result_json`` pour une analyse.
        """

        normalized_analysis_id = self._to_uuid(
            analysis_id,
            "analysis_id",
        )

        query = """
            SELECT result_json
            FROM prudencia.analyses
            WHERE id = %s;
        """

        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (normalized_analysis_id,),
                )
                row = cursor.fetchone()

        if row is None:
            raise ValueError(
                f"Analyse introuvable : {analysis_id}."
            )

        return row[0] or {}

    @staticmethod
    def measure_execution_time_ms(start_time: float) -> int:
        """
        Calcule une durée en millisecondes à partir de ``time.perf_counter``.
        """

        return max(
            0,
            int((time.perf_counter() - start_time) * 1000),
        )

    @staticmethod
    def _normalize_analysis_type(analysis_type: str) -> str:
        normalized = (
            analysis_type.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        aliases = {
            "document": "documentaire",
            "documentary": "documentaire",
            "pdf": "documentaire",
            "texte": "documentaire",
            "text": "documentaire",
            "questionnaire_ml": "questionnaire",
            "machine_learning": "questionnaire",
            "ml": "questionnaire",
        }

        return aliases.get(normalized, normalized)

    @staticmethod
    def _normalize_confidence(
        confidence: float | None,
    ) -> float | None:
        if confidence is None:
            return None

        value = float(confidence)

        if value > 1.0 and value <= 100.0:
            value = value / 100.0

        return min(max(value, 0.0), 1.0)

    @staticmethod
    def _normalize_risk_score(
        risk_score: float | None,
    ) -> float | None:
        if risk_score is None:
            return None

        return min(max(float(risk_score), 0.0), 100.0)

    @staticmethod
    def _to_uuid(
        value: str | UUID,
        field_name: str,
    ) -> UUID:
        if isinstance(value, UUID):
            return value

        try:
            return UUID(str(value))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{field_name} doit être un UUID valide."
            ) from error

    @classmethod
    def _to_optional_uuid(
        cls,
        value: str | UUID | None,
        field_name: str,
    ) -> UUID | None:
        if value in {None, ""}:
            return None

        return cls._to_uuid(value, field_name)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None

        return float(value)

    @staticmethod
    def _datetime_to_iso(
        value: datetime | None,
    ) -> str | None:
        if value is None:
            return None

        return value.isoformat()

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def serialize_for_debug(value: Any) -> str:
        """
        Sérialise une valeur pour les journaux de diagnostic.
        """

        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        )