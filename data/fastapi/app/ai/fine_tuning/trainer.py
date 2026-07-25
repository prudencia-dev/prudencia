import time
import logging
import shutil
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import ClassLabel, Dataset
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.services.training_history_service import (
    save_model_reset,
)
from app.services.training_history_service import (
    save_training_execution,
)
from app.config import (
    AVAILABLE_MODELS,
    PRETRAINED_DIR,
    FINE_TUNED_DIR,
)
from app.ai.fine_tuning.dataset import (
    DatasetManager,
    DatasetQualityReport,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

@dataclass
class TrainingPreparation:
    status: str
    dataset_path: str
    text_column: str
    label_column: str
    total_examples: int
    number_of_classes: int
    classes: list[str]
    quality_score: int
    quality_level: str
    is_trainable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FineTuningTrainer:
    """
    Orchestrateur du pipeline de Fine-Tuning.

    Pour l'instant, cette classe :
    - charge le CSV ;
    - valide le dataset ;
    - nettoie les données ;
    - prépare le dataset d'entraînement.

    L'entraînement Hugging Face/PyTorch sera ajouté ensuite.
    """

    def __init__(self) -> None:
        self.dataset_manager = DatasetManager()
        self.quality_report: DatasetQualityReport | None = None
        self.training_dataframe: pd.DataFrame | None = None
        self.preparation: TrainingPreparation | None = None
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        self.hf_dataset = None
        self.tokenized_dataset = None
        self.train_dataset = None
        self.eval_dataset = None
        self.training_args = None
        self.trainer = None

    def prepare_training(
        self,
        csv_path: str,
        text_column: str,
        label_column: str,
    ) -> TrainingPreparation:
        path = Path(csv_path)

        self.dataset_manager.load_csv(str(path))

        self.quality_report = self.dataset_manager.validate(
            text_column=text_column,
            label_column=label_column,
        )

        self.training_dataframe = (
            self.dataset_manager.build_training_dataframe(
                text_column=text_column,
                label_column=label_column,
            )
        )

        self.preparation = TrainingPreparation(
            status=(
                "ready"
                if self.quality_report.is_trainable
                else "blocked"
            ),
            dataset_path=str(path),
            text_column=text_column,
            label_column=label_column,
            total_examples=len(self.training_dataframe),
            number_of_classes=(
                self.quality_report.info.number_of_classes
            ),
            classes=self.quality_report.info.classes,
            quality_score=self.quality_report.quality_score,
            quality_level=self.quality_report.quality_level,
            is_trainable=self.quality_report.is_trainable,
        )

        return self.preparation

    def get_quality_report(self) -> dict[str, Any]:
        if self.quality_report is None:
            raise RuntimeError(
                "Aucun dataset n'a encore été préparé."
            )

        return self.quality_report.to_dict()

    def get_training_dataframe(self) -> pd.DataFrame:
        if self.training_dataframe is None:
            raise RuntimeError(
                "Aucun dataset d'entraînement n'est disponible."
            )

        return self.training_dataframe.copy()

    def get_summary(self) -> dict[str, Any]:
        if self.preparation is None:
            raise RuntimeError(
                "Aucune préparation d'entraînement disponible."
            )

        return {
            "preparation": self.preparation.to_dict(),
            "quality_report": self.get_quality_report(),
        }

    def train(
        self,
        model_name: str,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
    ) -> dict[str, Any]:
        """
        Lance le pipeline complet de Fine-Tuning.

        Le dataset doit avoir été préparé avec prepare_training()
        avant l'appel de cette méthode.
        """

        if self.preparation is None:
            raise RuntimeError(
                "Le dataset doit être préparé avant l'entraînement."
            )

        if not self.preparation.is_trainable:
            raise ValueError(
                "Le dataset n'est pas exploitable pour l'entraînement."
            )

        self._load_model(model_name)
        self._load_tokenizer()
        self._create_hf_dataset()
        self._tokenize_dataset()
        self._split_dataset()

        self._create_training_arguments(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        self._create_trainer()

        metrics = self._train_model()
        model_path = self._save_model()

        result = {
            "success": True,
            "model_name": model_name,
            "base_model": {
                "selected_id": model_name,
                "display_name": AVAILABLE_MODELS[model_name]["name"],
                "huggingface_id": AVAILABLE_MODELS[model_name]["hf_id"],
                "architecture": self.model.__class__.__name__,
            },
            "model_path": model_path,
            "training_examples": len(self.train_dataset),
            "validation_examples": len(self.eval_dataset),
            "metrics": {
                "loss": metrics.get("eval_loss"),
                "accuracy": metrics.get("eval_accuracy"),
                "precision": metrics.get("eval_precision"),
                "recall": metrics.get("eval_recall"),
                "f1": metrics.get("eval_f1"),
            },
            "training_time": metrics.get("training_time"),
            "parameters": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
        }

        return result

    def _load_model(self, model_name: str) -> None:
        """
        Charge un modèle pré-entraîné.

        Si le modèle n'existe pas localement, il est téléchargé
        depuis Hugging Face puis sauvegardé dans models/pretrained.
        """
        
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Modèle inconnu : {model_name}")

        model_config = AVAILABLE_MODELS[model_name]

        pretrained_path = PRETRAINED_DIR / model_config["folder"]
        fine_tuned_path = FINE_TUNED_DIR / model_config["folder"]

        if pretrained_path.exists():
            self.logger.info(
                "Chargement du modèle local : %s",
                model_name,
            )

        else:
            self.logger.info(
                "Téléchargement du modèle %s depuis Hugging Face...",
                model_name,
            )

            pretrained_path.mkdir(parents=True, exist_ok=True)

            model = AutoModelForSequenceClassification.from_pretrained(
                model_config["hf_id"],
                num_labels=2,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                model_config["hf_id"],
            )

            model.save_pretrained(pretrained_path)
            tokenizer.save_pretrained(pretrained_path)

            self.logger.info("Modèle sauvegardé.")

        if fine_tuned_path.exists():
            shutil.rmtree(fine_tuned_path)

        shutil.copytree(pretrained_path, fine_tuned_path)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            fine_tuned_path,
        )

        self.current_model_name = model_name

    def _load_tokenizer(self) -> None:
        """
        Charge le tokenizer correspondant au modèle.
        """

        if self.current_model_name is None:
            raise RuntimeError(
                "Aucun modèle n'est chargé."
            )

        model_config = AVAILABLE_MODELS[self.current_model_name]

        fine_tuned_path = (
            FINE_TUNED_DIR / model_config["folder"]
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            fine_tuned_path,
        )

        self.logger.info(
            "Tokenizer chargé."
        )

    def _create_hf_dataset(self) -> None:
        """
        Convertit le DataFrame pandas en Dataset Hugging Face
        et transforme les labels en classes numériques.
        """

        if self.training_dataframe is None:
            raise RuntimeError(
                "Aucun DataFrame d'entraînement disponible."
            )

        dataframe = self.training_dataframe.copy()

        # Les labels textuels sont convertis en identifiants numériques.
        dataframe["label"] = (
            dataframe["label"]
            .astype(str)
            .str.strip()
        )

        class_names = sorted(
            dataframe["label"].unique().tolist()
        )

        label_to_id = {
            class_name: class_id
            for class_id, class_name in enumerate(class_names)
        }

        dataframe["label"] = dataframe["label"].map(
            label_to_id
        )

        if dataframe["label"].isna().any():
            raise ValueError(
                "Certains labels n'ont pas pu être convertis."
            )

        dataframe["label"] = dataframe["label"].astype(int)

        self.label_to_id = label_to_id
        self.id_to_label = {
            class_id: class_name
            for class_name, class_id in label_to_id.items()
        }

        num_labels = len(class_names)

        if self.current_model_name is None:
            raise RuntimeError(
                "Aucun modèle n'est chargé."
            )

        model_config = AVAILABLE_MODELS[
            self.current_model_name
        ]

        fine_tuned_path = (
            FINE_TUNED_DIR
            / model_config["folder"]
        )

        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                fine_tuned_path,
                num_labels=num_labels,
                label2id=self.label_to_id,
                id2label=self.id_to_label,
                ignore_mismatched_sizes=True,
            )
        )

        self.logger.info(
            "Tête de classification configurée pour %d classes.",
            num_labels,
        )

        self.hf_dataset = Dataset.from_pandas(
            dataframe,
            preserve_index=False,
        )

        self.hf_dataset = self.hf_dataset.cast_column(
            "label",
            ClassLabel(names=class_names),
        )

        self.logger.info(
            "Dataset créé : %d exemples, %d classes.",
            len(self.hf_dataset),
            len(class_names),
        )

    def _tokenize_dataset(self) -> None:
        """
        Tokenise le dataset Hugging Face.
        """

        if self.hf_dataset is None:
            raise RuntimeError(
                "Le Dataset Hugging Face n'est pas disponible."
            )

        if self.tokenizer is None:
            raise RuntimeError(
                "Le tokenizer n'est pas chargé."
            )

        text_column = "text"

        if text_column not in self.hf_dataset.column_names:
            raise RuntimeError(
                f"La colonne '{text_column}' est absente du Dataset. "
                f"Colonnes disponibles : {self.hf_dataset.column_names}"
            )

        def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
            return self.tokenizer(
                batch[text_column],
                truncation=True,
                padding="max_length",
                max_length=512,
            )

        self.tokenized_dataset = self.hf_dataset.map(
            tokenize,
            batched=True,
        )

        required_columns = {
            "input_ids",
            "attention_mask",
            "label",
        }

        missing_columns = (
            required_columns
            - set(self.tokenized_dataset.column_names)
        )

        if missing_columns:
            raise RuntimeError(
                "Colonnes manquantes après tokenisation : "
                f"{sorted(missing_columns)}"
            )

        self.logger.info(
            "Dataset tokenisé. Colonnes disponibles : %s",
            self.tokenized_dataset.column_names,
        )


    def _split_dataset(self) -> None:
        """
        Sépare le dataset tokenisé entre entraînement et validation.
        """

        if self.tokenized_dataset is None:
            raise RuntimeError(
                "Le dataset tokenisé n'est pas disponible."
            )

        if self.training_dataframe is None:
            raise RuntimeError(
                "Le DataFrame d'entraînement n'est pas disponible."
            )

        label_counts = (
            self.training_dataframe["label"]
            .value_counts()
        )

        can_stratify = (
            len(label_counts) > 1
            and label_counts.min() >= 2
        )

        if can_stratify:
            try:
                self.dataset_split = (
                    self.tokenized_dataset.train_test_split(
                        test_size=0.2,
                        seed=42,
                        stratify_by_column="label",
                    )
                )

            except ValueError as error:
                self.logger.warning(
                    "Découpage stratifié impossible (%s). "
                    "Utilisation d'un découpage simple.",
                    error,
                )

                self.dataset_split = (
                    self.tokenized_dataset.train_test_split(
                        test_size=0.2,
                        seed=42,
                    )
                )

        else:
            self.logger.warning(
                "Découpage non stratifié : au moins une classe "
                "contient moins de 2 exemples."
            )

            self.dataset_split = (
                self.tokenized_dataset.train_test_split(
                    test_size=0.2,
                    seed=42,
                )
            )

        self.train_dataset = self.dataset_split["train"]
        self.eval_dataset = self.dataset_split["test"]

        self.logger.info(
            "Dataset séparé : %d entraînement, %d validation.",
            len(self.train_dataset),
            len(self.eval_dataset),
        )

        self.logger.info(
            "Colonnes d'entraînement : %s",
            self.train_dataset.column_names,
        )

    def _compute_metrics(self, eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        )

        accuracy = accuracy_score(labels, predictions)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def _create_training_arguments(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> None:
        """
        Crée les paramètres d'entraînement Hugging Face.
        """

        self.training_args = TrainingArguments(
            output_dir="/app/models/checkpoints",

            num_train_epochs=epochs,

            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,

            learning_rate=learning_rate,

            eval_strategy="epoch",
            save_strategy="epoch",

            logging_strategy="epoch",

            load_best_model_at_end=True,

            metric_for_best_model="f1",

            report_to="none",
        )

    def _create_trainer(self) -> None:
        """
        Construit le Trainer Hugging Face.
        """

        if self.model is None:
            raise RuntimeError("Le modèle n'est pas chargé.")

        if self.training_args is None:
            raise RuntimeError(
                "Les paramètres d'entraînement ne sont pas créés."
            )

        if self.train_dataset is None or self.eval_dataset is None:
            raise RuntimeError(
                "Les datasets train et validation ne sont pas disponibles."
            )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            compute_metrics=self._compute_metrics,
        )

        self.logger.info("Trainer créé.")

    def _train_model(self) -> dict[str, float]:
        """
        Lance le Fine-Tuning.
        """

        if self.trainer is None:
            raise RuntimeError(
                "Le Trainer n'est pas initialisé."
            )

        self.logger.info("Début de l'entraînement...")

        start = time.time()

        self.trainer.train()

        metrics = self.trainer.evaluate()

        duration = time.time() - start

        metrics["training_time"] = round(duration, 2)

        self.logger.info(
            "Entraînement terminé en %.2f secondes.",
            duration,
        )

        return metrics
    
    def _save_model(self) -> str:
        """
        Sauvegarde le modèle fine-tuné.
        """

        if self.current_model_name is None:
            raise RuntimeError(
                "Aucun modèle chargé."
            )

        model_config = AVAILABLE_MODELS[self.current_model_name]

        model_path = (
            FINE_TUNED_DIR /
            model_config["folder"]
        )

        self.trainer.save_model(model_path)

        self.logger.info(
            "Modèle sauvegardé : %s",
            model_path,
        )

        return str(model_path)
    
    def reset_model(
        self,
        model_name: str,
    ) -> dict[str, Any]:
        """
        Supprime le modèle fine-tuné et restaure
        la copie du modèle pré-entraîné.
        """

        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Modèle inconnu : {model_name}"
            )

        model_config = AVAILABLE_MODELS[model_name]

        pretrained_path = (
            PRETRAINED_DIR / model_config["folder"]
        )

        fine_tuned_path = (
            FINE_TUNED_DIR / model_config["folder"]
        )

        if not pretrained_path.exists():
            raise FileNotFoundError(
                "Le modèle pré-entraîné n'existe pas localement."
            )

        if fine_tuned_path.exists():
            shutil.rmtree(fine_tuned_path)

        shutil.copytree(
            pretrained_path,
            fine_tuned_path,
        )

        save_model_reset(
            model_type="deep_learning",
            model_name=model_name,
            model_version="v1.0.0",
            reason="Retour au modèle pré-entraîné de référence",
        )

        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        self.trainer = None

        return {
            "success": True,
            "message": "Modèle Fine-Tuning réinitialisé.",
            "model_name": model_name,
            "model_version": "v1.0.0",
            "model_path": str(fine_tuned_path),
        }