from pathlib import Path

from app.ai.fine_tuning.dataset import DatasetManager


def test_training_demo_dataset_is_trainable() -> None:
    dataset_path = Path(__file__).parents[1] / "datasets" / "training_demo.csv"
    manager = DatasetManager()

    dataframe = manager.load_csv(dataset_path)
    report = manager.validate(text_column="texte", label_column="label")

    assert len(dataframe) >= manager.MINIMUM_EXAMPLES
    assert report.is_trainable
    assert report.info.number_of_classes == 2
