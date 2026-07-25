from pprint import pprint

from app.ai.fine_tuning.benchmark import BenchmarkManager

benchmark = BenchmarkManager()

benchmark.add_result(
    {
        "model_name": "camembert",
        "metrics": {
            "accuracy": 0.91,
            "precision": 0.90,
            "recall": 0.92,
            "f1": 0.91,
        },
        "training_time": 120,
    }
)

benchmark.add_result(
    {
        "model_name": "juribert",
        "metrics": {
            "accuracy": 0.88,
            "precision": 0.87,
            "recall": 0.89,
            "f1": 0.88,
        },
        "training_time": 160,
    }
)

print("\nClassement :")
pprint(benchmark.compare())

print("\nMeilleur modèle :")
pprint(benchmark.get_best_model())