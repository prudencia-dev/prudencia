from pprint import pprint

from app.ai.fine_tuning.trainer import FineTuningTrainer


print("=== PRUDENCIA - TEST PIPELINE FINE-TUNING ===")

trainer = FineTuningTrainer()

print("Préparation du dataset...")

preparation = trainer.prepare_training(
    csv_path="/app/datasets/training_demo.csv",
    text_column="texte",
    label_column="label",
)

print("Dataset prêt :")
pprint(preparation.to_dict())

print("\nLancement du Fine-Tuning...")

result = trainer.train(
    model_name="camembert",
    epochs=1,
    batch_size=2,
    learning_rate=2e-5,
)

print("\nRésultat :")
pprint(result)

print("\n✅ PIPELINE COMPLET OK")