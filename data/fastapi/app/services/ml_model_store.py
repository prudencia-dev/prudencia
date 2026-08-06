"""Stockage versionné et cache du modèle Machine Learning actif."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

import joblib


class MachineLearningModelStore:
    """Publie des modèles immuables et charge atomiquement la version active."""

    def __init__(self, root: Path = Path("models") / "machine_learning") -> None:
        self.root = root
        self.versions_dir = root / "versions"
        self.active_pointer = root / "active_model.json"
        self.legacy_model_path = root / "random_forest.joblib"
        self._lock = RLock()
        self._cached_model: Any = None
        self._cached_path: Path | None = None
        self._cached_mtime_ns: int | None = None

    def _new_version(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{timestamp}-{uuid4().hex[:8]}"

    def _confined_model_path(self, relative_path: str) -> Path:
        root = self.root.resolve()
        candidate = (root / relative_path).resolve()
        if root not in candidate.parents:
            raise RuntimeError("Le chemin du modèle actif est invalide.")
        return candidate

    def _read_pointer(self) -> dict[str, Any] | None:
        if not self.active_pointer.is_file():
            return None
        try:
            pointer = json.loads(self.active_pointer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Le pointeur du modèle actif est illisible.") from error
        if not isinstance(pointer, dict):
            raise RuntimeError("Le pointeur du modèle actif est invalide.")
        return pointer

    def save_and_activate(
        self,
        model: Any,
        *,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Sauvegarde une nouvelle version puis bascule le pointeur actif."""
        with self._lock:
            version = self._new_version()
            version_dir = self.versions_dir / version
            version_dir.mkdir(parents=True, exist_ok=False)
            model_path = version_dir / "model.joblib"
            temporary_model_path = version_dir / ".model.joblib.tmp"
            metadata_path = version_dir / "metadata.json"
            temporary_pointer = self.root / f".active-{uuid4().hex}.tmp"

            try:
                joblib.dump(model, temporary_model_path)
                os.replace(temporary_model_path, model_path)

                artifact = {
                    "version": version,
                    "model_path": str(model_path.relative_to(self.root)),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": metadata,
                }
                metadata_path.write_text(
                    json.dumps(artifact, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                self.root.mkdir(parents=True, exist_ok=True)
                temporary_pointer.write_text(
                    json.dumps(artifact, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(temporary_pointer, self.active_pointer)
            except Exception:
                temporary_model_path.unlink(missing_ok=True)
                temporary_pointer.unlink(missing_ok=True)
                raise

            self._cached_model = model
            self._cached_path = model_path.resolve()
            self._cached_mtime_ns = model_path.stat().st_mtime_ns
            return artifact

    def get_active_artifact(self) -> dict[str, Any] | None:
        """Retourne les métadonnées actives, avec support de l'ancien fichier."""
        pointer = self._read_pointer()
        if pointer is not None:
            model_path = pointer.get("model_path")
            version = pointer.get("version")
            if not isinstance(model_path, str) or not isinstance(version, str):
                raise RuntimeError("Le pointeur du modèle actif est incomplet.")
            resolved_path = self._confined_model_path(model_path)
            if not resolved_path.is_file():
                raise RuntimeError("Le fichier du modèle actif est introuvable.")
            return {**pointer, "resolved_model_path": str(resolved_path)}

        if self.legacy_model_path.is_file():
            return {
                "version": "legacy-v1.1.0",
                "model_path": self.legacy_model_path.name,
                "resolved_model_path": str(self.legacy_model_path.resolve()),
                "metadata": {"legacy": True},
            }
        return None

    def load_active(self) -> tuple[Any, dict[str, Any]]:
        """Charge le modèle actif une seule fois tant que son fichier ne change pas."""
        with self._lock:
            artifact = self.get_active_artifact()
            if artifact is None:
                raise RuntimeError("Le modèle Machine Learning n'est pas entraîné.")

            model_path = Path(artifact["resolved_model_path"])
            mtime_ns = model_path.stat().st_mtime_ns
            if (
                self._cached_model is None
                or self._cached_path != model_path
                or self._cached_mtime_ns != mtime_ns
            ):
                self._cached_model = joblib.load(model_path)
                self._cached_path = model_path
                self._cached_mtime_ns = mtime_ns
            return self._cached_model, artifact

    def deactivate(self) -> dict[str, Any]:
        """Désactive le modèle sans supprimer les versions historisées."""
        with self._lock:
            pointer_deleted = self.active_pointer.is_file()
            self.active_pointer.unlink(missing_ok=True)
            legacy_deleted = self.legacy_model_path.is_file()
            self.legacy_model_path.unlink(missing_ok=True)
            self._cached_model = None
            self._cached_path = None
            self._cached_mtime_ns = None
            return {
                "model_deleted": pointer_deleted or legacy_deleted,
                "versions_preserved": self.versions_dir.is_dir(),
            }


ML_MODEL_STORE = MachineLearningModelStore()
