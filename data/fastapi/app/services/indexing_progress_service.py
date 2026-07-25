from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4


class IndexingProgressService:
    """
    Stocke en mémoire l'état des indexations RAG.

    Ce service permet à Streamlit de consulter régulièrement
    l'avancement d'une indexation lancée par l'API.

    Pour le MVP, le stockage en mémoire est suffisant.
    Les tâches sont perdues lorsque le conteneur API redémarre.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def create_job(self, filename: str) -> dict[str, Any]:
        """
        Crée une nouvelle tâche d'indexation.
        """

        job_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()

        job = {
            "job_id": job_id,
            "filename": filename,
            "status": "pending",
            "stage": "Préparation de l'indexation",
            "progress": 0,
            "current_chunk": 0,
            "total_chunks": 0,
            "message": "Indexation en attente de démarrage.",
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }

        with self._lock:
            self._jobs[job_id] = job

        return deepcopy(job)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """
        Retourne une copie d'une tâche.
        """

        with self._lock:
            job = self._jobs.get(job_id)

            if job is None:
                return None

            return deepcopy(job)

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: int | float | None = None,
        current_chunk: int | None = None,
        total_chunks: int | None = None,
        message: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """
        Met à jour l'état d'une tâche.
        """

        with self._lock:
            job = self._jobs.get(job_id)

            if job is None:
                raise ValueError(
                    f"Tâche d'indexation introuvable : {job_id}"
                )

            if status is not None:
                job["status"] = status

            if stage is not None:
                job["stage"] = stage

            if progress is not None:
                job["progress"] = max(
                    0,
                    min(100, round(float(progress), 2)),
                )

            if current_chunk is not None:
                job["current_chunk"] = current_chunk

            if total_chunks is not None:
                job["total_chunks"] = total_chunks

            if message is not None:
                job["message"] = message

            if result is not None:
                job["result"] = result

            if error is not None:
                job["error"] = error

            job["updated_at"] = (
                datetime.now(timezone.utc).isoformat()
            )

            return deepcopy(job)

    def mark_running(
        self,
        job_id: str,
        *,
        stage: str,
        progress: int | float,
        message: str,
    ) -> dict[str, Any]:
        """
        Indique qu'une tâche est en cours.
        """

        return self.update_job(
            job_id,
            status="running",
            stage=stage,
            progress=progress,
            message=message,
        )

    def mark_completed(
        self,
        job_id: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Indique qu'une tâche est terminée avec succès.
        """

        return self.update_job(
            job_id,
            status="completed",
            stage="Indexation terminée",
            progress=100,
            message="Le document a été correctement indexé.",
            result=result,
            error=None,
        )

    def mark_failed(
        self,
        job_id: str,
        error: str,
    ) -> dict[str, Any]:
        """
        Indique qu'une tâche a échoué.
        """

        return self.update_job(
            job_id,
            status="failed",
            stage="Échec de l'indexation",
            message="Une erreur est survenue pendant l'indexation.",
            error=error,
        )


indexing_progress_service = IndexingProgressService()