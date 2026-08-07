import pytest
from app.ai.fine_tuning.benchmark import BenchmarkManager


def test_benchmark_orders_models_by_f1() -> None:
    benchmark = BenchmarkManager()
    benchmark.add_result({"model_name": "camembert", "metrics": {"f1": 0.91}})
    benchmark.add_result({"model_name": "juribert", "metrics": {"f1": 0.88}})

    assert [result["model_name"] for result in benchmark.compare()] == [
        "camembert",
        "juribert",
    ]
    assert benchmark.get_best_model()["model_name"] == "camembert"


def test_benchmark_rejects_empty_results() -> None:
    with pytest.raises(RuntimeError, match="Aucun résultat"):
        BenchmarkManager().get_best_model()
