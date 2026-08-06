import json
from pathlib import Path

import pytest
from app.services import ml_model_store
from app.services.ml_model_store import MachineLearningModelStore


def test_model_versions_are_immutable_and_latest_is_active(tmp_path: Path) -> None:
    store = MachineLearningModelStore(tmp_path / "models")

    first = store.save_and_activate({"model": 1}, metadata={"score": 0.7})
    second = store.save_and_activate({"model": 2}, metadata={"score": 0.9})

    assert first["version"] != second["version"]
    assert (store.root / first["model_path"]).is_file()
    assert (store.root / second["model_path"]).is_file()
    model, active = store.load_active()
    assert model == {"model": 2}
    assert active["version"] == second["version"]


def test_active_model_is_loaded_only_once(tmp_path: Path, monkeypatch) -> None:
    store = MachineLearningModelStore(tmp_path / "models")
    store.save_and_activate({"model": 1}, metadata={})
    store._cached_model = None
    calls = 0
    original_load = ml_model_store.joblib.load

    def counted_load(path):
        nonlocal calls
        calls += 1
        return original_load(path)

    monkeypatch.setattr(ml_model_store.joblib, "load", counted_load)

    assert store.load_active()[0] == {"model": 1}
    assert store.load_active()[0] == {"model": 1}
    assert calls == 1


def test_deactivation_preserves_versioned_files(tmp_path: Path) -> None:
    store = MachineLearningModelStore(tmp_path / "models")
    artifact = store.save_and_activate({"model": 1}, metadata={})
    model_path = store.root / artifact["model_path"]

    result = store.deactivate()

    assert result == {"model_deleted": True, "versions_preserved": True}
    assert model_path.is_file()
    assert store.get_active_artifact() is None


def test_active_pointer_cannot_escape_model_directory(tmp_path: Path) -> None:
    store = MachineLearningModelStore(tmp_path / "models")
    store.root.mkdir(parents=True)
    store.active_pointer.write_text(
        json.dumps({"version": "bad", "model_path": "../../outside.joblib"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="chemin du modèle actif"):
        store.get_active_artifact()
