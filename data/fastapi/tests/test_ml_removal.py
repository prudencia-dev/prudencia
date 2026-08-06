from pathlib import Path

from app.services.model_selection_service import ModelSelectionService
from app.services.report_builder import PrudenciaReportBuilder

PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_machine_learning_modules_are_removed() -> None:
    removed_paths = (
        "app/api/ml.py",
        "app/api/questionnaire_ml.py",
        "app/ai/machine_learning/trainer.py",
    )

    assert all(not (PROJECT_DIR / path).exists() for path in removed_paths)

    main_source = (PROJECT_DIR / "app/main.py").read_text(encoding="utf-8")
    report_source = (PROJECT_DIR / "app/api/reports.py").read_text(
        encoding="utf-8"
    )

    assert "ml_router" not in main_source
    assert "questionnaire_ml_router" not in main_source
    assert '"/generate-ml"' not in report_source


def test_model_catalog_only_contains_rag_and_deep_learning(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ModelSelectionService,
        "CONFIG_PATH",
        tmp_path / "active_models.json",
    )

    service = ModelSelectionService()

    assert set(service.get_configuration()) == {"rag", "deep_learning"}
    assert set(service.list_available_models()) == {"rag", "deep_learning"}


def test_report_contains_only_deep_learning_and_rag_results() -> None:
    report = PrudenciaReportBuilder().build(
        project={"title": "Projet test", "description": "Description"},
        deep_learning_result={
            "prediction": "haut_risque",
            "confidence": 0.91,
        },
        rag_result={"references": []},
    )

    assert "machine_learning_result" not in report
    assert report["deep_learning_result"]["prediction"] == "haut_risque"
    assert report["ai_act"]["classification"] == "haut_risque"
