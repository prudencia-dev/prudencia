from pathlib import Path

import pandas as pd
from app.ai.machine_learning import trainer as trainer_module
from app.ai.machine_learning.trainer import MachineLearningTrainer
from app.services.ml_model_store import MachineLearningModelStore


def test_training_publishes_version_and_prediction_uses_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataframe = pd.DataFrame(
        {
            "score": list(range(20)),
            "sector": ["health", "education"] * 10,
            "risk": ["high", "limited"] * 10,
        }
    )
    dataframe.to_csv(dataset_path, index=False)
    monkeypatch.setattr(trainer_module, "save_training_execution", lambda **_: None)
    monkeypatch.setattr(trainer_module, "save_model_reset", lambda **_: None)

    store = MachineLearningModelStore(tmp_path / "models")
    trainer = MachineLearningTrainer(
        n_estimators=10,
        model_store=store,
    )

    training = trainer.train(
        csv_path=str(dataset_path),
        feature_columns=["score", "sector"],
        target_column="risk",
    )
    prediction = trainer.predict(
        pd.DataFrame([{"score": 10, "sector": "health"}])
    )

    assert training["model_version"] == prediction["model_version"]
    assert Path(training["model_path"]).is_file()
    assert prediction["predictions"][0] in {"high", "limited"}

    reset = trainer.reset_model()
    assert reset["model_deleted"]
    assert reset["versions_preserved"]
    assert store.get_active_artifact() is None
