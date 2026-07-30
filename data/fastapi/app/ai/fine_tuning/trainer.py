from __future__ import annotations

import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import ClassLabel, Dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from app.ai.fine_tuning.dataset import (
    DatasetManager,
    DatasetQualityReport,
)
from app.config import (
    AVAILABLE_MODELS,
    FINE_TUNED_DIR,
    PRETRAINED_DIR,
)
from app.services.training_history_service import save_model_reset


@dataclass
class TrainingPreparation:
    """Résumé de la préparation du dataset avant l'entraînement."""

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


class WeightedLossTrainer(Trainer):
    """
    Variante du Trainer Hugging Face utilisant des poids de classes.

    Pourquoi ?
    -----------
    Lorsqu'une classe est beaucoup plus présente que les autres, un modèle peut
    obtenir une accuracy correcte en privilégiant cette classe majoritaire.
    Les poids de classes augmentent le coût des erreurs commises sur les classes
    moins représentées.
    """

    def __init__(
        self,
        *args: Any,
        class_weights: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **_: Any,
    ) -> Any:
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if labels is None or logits is None:
            loss = outputs.get("loss")
        else:
            weights = self.class_weights

            if weights is not None:
                weights = weights.to(logits.device)

            loss_function = torch.nn.CrossEntropyLoss(
                weight=weights,
            )
            loss = loss_function(
                logits.view(-1, model.config.num_labels),
                labels.view(-1),
            )

        return (loss, outputs) if return_outputs else loss


class FineTuningTrainer:
    """
    Orchestre le pipeline complet de Fine-Tuning de PRUDENCIA.

    Étapes principales :
    1. validation et nettoyage du CSV ;
    2. chargement du modèle et de son tokenizer ;
    3. conversion des labels textuels en identifiants numériques ;
    4. tokenisation des descriptions ;
    5. séparation entraînement / validation ;
    6. entraînement avec les hyperparamètres choisis dans Streamlit ;
    7. évaluation détaillée et sauvegarde du meilleur modèle.
    """

    def __init__(self) -> None:
        self.dataset_manager = DatasetManager()
        self.quality_report: DatasetQualityReport | None = None
        self.training_dataframe: pd.DataFrame | None = None
        self.preparation: TrainingPreparation | None = None
        self.logger = logging.getLogger(__name__)

        self.model: Any = None
        self.tokenizer: Any = None
        self.current_model_name: str | None = None
        self.hf_dataset: Dataset | None = None
        self.tokenized_dataset: Dataset | None = None
        self.train_dataset: Dataset | None = None
        self.eval_dataset: Dataset | None = None
        self.training_args: TrainingArguments | None = None
        self.trainer: Trainer | None = None

        self.label_to_id: dict[str, int] = {}
        self.id_to_label: dict[int, str] = {}

        # Valeurs mémorisées pour que toutes les étapes du pipeline utilisent
        # exactement la configuration choisie depuis Streamlit.
        self.seed = 42
        self.max_length = 256
        self.early_stopping_patience = 3
        self.use_class_weights = True
        self.class_weights: torch.Tensor | None = None
        self.token_length_statistics: dict[str, Any] = {}

    def prepare_training(
        self,
        csv_path: str,
        text_column: str,
        label_column: str,
    ) -> TrainingPreparation:
        """Charge, contrôle et prépare le dataset CSV."""

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
        epochs: int = 10,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        seed: int = 42,
        max_length: int = 256,
        gradient_accumulation_steps: int = 2,
        early_stopping_patience: int = 3,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.10,
        metric_for_best_model: str = "macro_f1",
        use_class_weights: bool = True,
    ) -> dict[str, Any]:
        """
        Lance le pipeline complet de Fine-Tuning.

        Les paramètres proviennent de l'interface Streamlit. Le dataset doit
        avoir été préparé avec ``prepare_training`` avant cet appel.
        """

        if self.preparation is None:
            raise RuntimeError(
                "Le dataset doit être préparé avant l'entraînement."
            )

        if not self.preparation.is_trainable:
            raise ValueError(
                "Le dataset n'est pas exploitable pour l'entraînement."
            )

        # Une graine fixe rend les expériences comparables : même découpage,
        # même initialisation de la tête de classification et même mélange.
        set_seed(seed)
        self.seed = seed
        self.max_length = max_length
        self.early_stopping_patience = early_stopping_patience
        self.use_class_weights = use_class_weights

        self._load_model(model_name)
        self._load_tokenizer()
        self._create_hf_dataset()
        self._analyze_token_lengths()
        self._tokenize_dataset()
        self._split_dataset()
        self._prepare_class_weights()

        self._create_training_arguments(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            gradient_accumulation_steps=(
                gradient_accumulation_steps
            ),
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            metric_for_best_model=metric_for_best_model,
        )

        self._create_trainer()

        metrics = self._train_model()
        detailed_metrics = self._build_detailed_evaluation()
        metrics.update(detailed_metrics)
        model_path = self._save_model()

        effective_batch_size = (
            batch_size * gradient_accumulation_steps
        )

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
                "precision": metrics.get(
                    "eval_precision_weighted"
                ),
                "recall": metrics.get("eval_recall_weighted"),
                "f1": metrics.get("eval_f1_weighted"),
                "precision_weighted": metrics.get(
                    "eval_precision_weighted"
                ),
                "recall_weighted": metrics.get(
                    "eval_recall_weighted"
                ),
                "f1_weighted": metrics.get(
                    "eval_f1_weighted"
                ),
                "precision_macro": metrics.get(
                    "eval_precision_macro"
                ),
                "recall_macro": metrics.get("eval_recall_macro"),
                "macro_f1": metrics.get("eval_macro_f1"),
                "per_class": metrics.get("per_class", {}),
                "confusion_matrix": metrics.get(
                    "confusion_matrix",
                    [],
                ),
                "class_names": metrics.get("class_names", []),
            },
            "training_time": metrics.get("training_time"),
            "training_state": {
                "epochs_requested": epochs,
                "epochs_completed": metrics.get(
                    "epochs_completed"
                ),
                "best_metric": metrics.get("best_metric"),
                "best_checkpoint": metrics.get("best_checkpoint"),
                "early_stopping_enabled": (
                    early_stopping_patience > 0
                ),
            },
            "token_length_statistics": self.token_length_statistics,
            "parameters": {
                "epochs": epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": (
                    gradient_accumulation_steps
                ),
                "effective_batch_size": effective_batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "max_length": max_length,
                "early_stopping_patience": (
                    early_stopping_patience
                ),
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "metric_for_best_model": metric_for_best_model,
                "use_class_weights": use_class_weights,
            },
        }

        return result

    def _load_model(self, model_name: str) -> None:
        """
        Charge une copie locale du modèle pré-entraîné.

        Une copie de travail est créée dans ``fine_tuned`` afin que le modèle
        pré-entraîné de référence reste intact pour les réinitialisations.
        """

        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Modèle inconnu : {model_name}")

        model_config = AVAILABLE_MODELS[model_name]
        pretrained_path = PRETRAINED_DIR / model_config["folder"]
        fine_tuned_path = FINE_TUNED_DIR / model_config["folder"]

        if not pretrained_path.exists():
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
        else:
            self.logger.info(
                "Chargement du modèle local : %s",
                model_name,
            )

        if fine_tuned_path.exists():
            shutil.rmtree(fine_tuned_path)

        shutil.copytree(pretrained_path, fine_tuned_path)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            fine_tuned_path,
        )
        self.current_model_name = model_name

    def _load_tokenizer(self) -> None:
        """Charge le tokenizer correspondant au modèle sélectionné."""

        if self.current_model_name is None:
            raise RuntimeError("Aucun modèle n'est chargé.")

        model_config = AVAILABLE_MODELS[self.current_model_name]
        fine_tuned_path = FINE_TUNED_DIR / model_config["folder"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            fine_tuned_path,
        )

    def _create_hf_dataset(self) -> None:
        """
        Convertit le DataFrame en Dataset Hugging Face.

        Les catégories textuelles sont transformées en identifiants numériques
        compris entre 0 et ``nombre_de_classes - 1``.
        """

        if self.training_dataframe is None:
            raise RuntimeError(
                "Aucun DataFrame d'entraînement disponible."
            )

        dataframe = self.training_dataframe.copy()
        dataframe["label"] = (
            dataframe["label"].astype(str).str.strip()
        )

        class_names = sorted(
            dataframe["label"].unique().tolist()
        )
        self.label_to_id = {
            class_name: class_id
            for class_id, class_name in enumerate(class_names)
        }
        self.id_to_label = {
            class_id: class_name
            for class_name, class_id in self.label_to_id.items()
        }

        dataframe["label"] = dataframe["label"].map(
            self.label_to_id
        )

        if dataframe["label"].isna().any():
            raise ValueError(
                "Certains labels n'ont pas pu être convertis."
            )

        dataframe["label"] = dataframe["label"].astype(int)

        if self.current_model_name is None:
            raise RuntimeError("Aucun modèle n'est chargé.")

        model_config = AVAILABLE_MODELS[self.current_model_name]
        fine_tuned_path = FINE_TUNED_DIR / model_config["folder"]

        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                fine_tuned_path,
                num_labels=len(class_names),
                label2id=self.label_to_id,
                id2label=self.id_to_label,
                ignore_mismatched_sizes=True,
            )
        )

        self.hf_dataset = Dataset.from_pandas(
            dataframe,
            preserve_index=False,
        )
        self.hf_dataset = self.hf_dataset.cast_column(
            "label",
            ClassLabel(names=class_names),
        )

    def _analyze_token_lengths(self) -> None:
        """
        Mesure la longueur réelle des textes avant troncature.

        Cette information aide à choisir ``max_length``. Une valeur trop grande
        ralentit fortement le serveur ; une valeur trop petite coupe une partie
        du texte et peut supprimer l'information juridique décisive.
        """

        if self.hf_dataset is None or self.tokenizer is None:
            return

        lengths: list[int] = []

        for text in self.hf_dataset["text"]:
            encoded = self.tokenizer(
                str(text),
                truncation=False,
                add_special_tokens=True,
            )
            lengths.append(len(encoded["input_ids"]))

        if not lengths:
            self.token_length_statistics = {}
            return

        values = np.asarray(lengths)
        self.token_length_statistics = {
            "minimum": int(values.min()),
            "median": int(np.median(values)),
            "p95": int(np.percentile(values, 95)),
            "maximum": int(values.max()),
            "texts_truncated": int(
                (values > self.max_length).sum()
            ),
            "total_texts": int(len(values)),
            "max_length_used": self.max_length,
        }

        self.logger.info(
            "Tokens : min=%d, médiane=%d, p95=%d, max=%d, "
            "tronqués=%d/%d, max_length=%d",
            self.token_length_statistics["minimum"],
            self.token_length_statistics["median"],
            self.token_length_statistics["p95"],
            self.token_length_statistics["maximum"],
            self.token_length_statistics["texts_truncated"],
            self.token_length_statistics["total_texts"],
            self.max_length,
        )

    def _tokenize_dataset(self) -> None:
        """Tokenise les textes avec la longueur choisie dans Streamlit."""

        if self.hf_dataset is None:
            raise RuntimeError(
                "Le Dataset Hugging Face n'est pas disponible."
            )

        if self.tokenizer is None:
            raise RuntimeError("Le tokenizer n'est pas chargé.")

        if "text" not in self.hf_dataset.column_names:
            raise RuntimeError(
                "La colonne 'text' est absente du Dataset."
            )

        def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
            return self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )

        self.tokenized_dataset = self.hf_dataset.map(
            tokenize,
            batched=True,
        )

    def _split_dataset(self) -> None:
        """Crée un split validation de 20 %, stratifié si possible."""

        if self.tokenized_dataset is None:
            raise RuntimeError(
                "Le dataset tokenisé n'est pas disponible."
            )

        if self.training_dataframe is None:
            raise RuntimeError(
                "Le DataFrame d'entraînement n'est pas disponible."
            )

        label_counts = self.training_dataframe["label"].value_counts()
        can_stratify = (
            len(label_counts) > 1
            and label_counts.min() >= 2
        )

        split_options: dict[str, Any] = {
            "test_size": 0.2,
            "seed": self.seed,
        }

        if can_stratify:
            split_options["stratify_by_column"] = "label"

        try:
            self.dataset_split = (
                self.tokenized_dataset.train_test_split(
                    **split_options,
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
                    seed=self.seed,
                )
            )

        self.train_dataset = self.dataset_split["train"]
        self.eval_dataset = self.dataset_split["test"]

    def _prepare_class_weights(self) -> None:
        """Calcule les poids inverses des classes du jeu d'entraînement."""

        if not self.use_class_weights or self.train_dataset is None:
            self.class_weights = None
            return

        labels = np.asarray(self.train_dataset["label"], dtype=int)
        counts = np.bincount(
            labels,
            minlength=len(self.id_to_label),
        )

        if np.any(counts == 0):
            self.logger.warning(
                "Poids de classes désactivés : une classe est absente "
                "du jeu d'entraînement."
            )
            self.class_weights = None
            return

        total = counts.sum()
        number_of_classes = len(counts)
        weights = total / (number_of_classes * counts)
        self.class_weights = torch.tensor(
            weights,
            dtype=torch.float32,
        )

    def _compute_metrics(self, eval_pred: Any) -> dict[str, float]:
        """
        Calcule deux lectures complémentaires des performances.

        - ``weighted`` reflète la performance globale en tenant compte du
          nombre d'exemples par classe ;
        - ``macro`` donne le même poids à toutes les classes et révèle plus
          facilement une classe rare mal reconnue.
        """

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        precision_w, recall_w, f1_w, _ = (
            precision_recall_fscore_support(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        )
        precision_m, recall_m, f1_m, _ = (
            precision_recall_fscore_support(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        return {
            "accuracy": float(
                accuracy_score(labels, predictions)
            ),
            "precision_weighted": float(precision_w),
            "recall_weighted": float(recall_w),
            "f1_weighted": float(f1_w),
            "precision_macro": float(precision_m),
            "recall_macro": float(recall_m),
            "macro_f1": float(f1_m),
        }

    def _create_training_arguments(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        seed: int,
        gradient_accumulation_steps: int,
        weight_decay: float,
        warmup_ratio: float,
        metric_for_best_model: str,
    ) -> None:
        """Transforme les réglages Streamlit en TrainingArguments."""

        greater_is_better = metric_for_best_model != "loss"

        self.training_args = TrainingArguments(
            output_dir="/app/models/checkpoints",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=(
                gradient_accumulation_steps
            ),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            seed=seed,
            data_seed=seed,
            save_total_limit=2,
            report_to="none",
        )

    def _create_trainer(self) -> None:
        """Crée le Trainer, avec early stopping et poids de classes."""

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

        callbacks: list[Any] = []

        if self.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=(
                        self.early_stopping_patience
                    )
                )
            )

        trainer_class = (
            WeightedLossTrainer
            if self.class_weights is not None
            else Trainer
        )

        trainer_kwargs: dict[str, Any] = {
            "model": self.model,
            "args": self.training_args,
            "train_dataset": self.train_dataset,
            "eval_dataset": self.eval_dataset,
            "processing_class": self.tokenizer,
            "compute_metrics": self._compute_metrics,
            "callbacks": callbacks,
        }

        if trainer_class is WeightedLossTrainer:
            trainer_kwargs["class_weights"] = self.class_weights

        self.trainer = trainer_class(**trainer_kwargs)

    def _train_model(self) -> dict[str, Any]:
        """Lance l'entraînement puis évalue le meilleur checkpoint."""

        if self.trainer is None:
            raise RuntimeError("Le Trainer n'est pas initialisé.")

        start = time.time()
        self.trainer.train()
        metrics = self.trainer.evaluate()
        duration = time.time() - start

        metrics["training_time"] = round(duration, 2)
        metrics["epochs_completed"] = (
            round(float(self.trainer.state.epoch), 2)
            if self.trainer.state.epoch is not None
            else None
        )
        metrics["best_metric"] = self.trainer.state.best_metric
        metrics["best_checkpoint"] = (
            self.trainer.state.best_model_checkpoint
        )

        return metrics

    def _build_detailed_evaluation(self) -> dict[str, Any]:
        """Produit les métriques par classe et la matrice de confusion."""

        if self.trainer is None or self.eval_dataset is None:
            return {}

        predictions_output = self.trainer.predict(self.eval_dataset)
        labels = predictions_output.label_ids
        predictions = np.argmax(
            predictions_output.predictions,
            axis=-1,
        )

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                labels,
                predictions,
                labels=list(range(len(self.id_to_label))),
                zero_division=0,
            )
        )

        per_class: dict[str, dict[str, Any]] = {}

        for class_id in range(len(self.id_to_label)):
            class_name = self.id_to_label[class_id]
            per_class[class_name] = {
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
            }

        matrix = confusion_matrix(
            labels,
            predictions,
            labels=list(range(len(self.id_to_label))),
        )

        return {
            "per_class": per_class,
            "confusion_matrix": matrix.tolist(),
            "class_names": [
                self.id_to_label[index]
                for index in range(len(self.id_to_label))
            ],
        }

    def _save_model(self) -> str:
        """Sauvegarde le meilleur modèle et le tokenizer associé."""

        if self.current_model_name is None or self.trainer is None:
            raise RuntimeError("Aucun modèle entraîné à sauvegarder.")

        model_config = AVAILABLE_MODELS[self.current_model_name]
        model_path = FINE_TUNED_DIR / model_config["folder"]

        self.trainer.save_model(model_path)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(model_path)

        return str(model_path)

    def predict(
        self,
        model_name: str,
        text: str,
    ) -> dict[str, Any]:
        """Réalise une prédiction avec le modèle fine-tuné actif."""

        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Modèle inconnu : {model_name}")

        model_config = AVAILABLE_MODELS[model_name]
        model_path = FINE_TUNED_DIR / model_config["folder"]

        if not model_path.exists():
            raise FileNotFoundError(
                "Le modèle Fine-Tuné est introuvable."
            )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
        )
        model.eval()

        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(
                probabilities,
                dim=1,
            )

        prediction_id = int(prediction.item())
        label = model.config.id2label.get(
            prediction_id,
            str(prediction_id),
        )

        return {
            "success": True,
            "model_name": model_name,
            "prediction": label,
            "confidence": float(confidence.item()),
            "probabilities": {
                model.config.id2label[index]: float(
                    probabilities[0][index]
                )
                for index in range(probabilities.shape[1])
            },
        }

    def reset_model(
        self,
        model_name: str,
    ) -> dict[str, Any]:
        """Restaure la copie propre du modèle pré-entraîné."""

        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Modèle inconnu : {model_name}")

        model_config = AVAILABLE_MODELS[model_name]
        pretrained_path = PRETRAINED_DIR / model_config["folder"]
        fine_tuned_path = FINE_TUNED_DIR / model_config["folder"]

        if not pretrained_path.exists():
            raise FileNotFoundError(
                "Le modèle pré-entraîné n'existe pas localement."
            )

        if fine_tuned_path.exists():
            shutil.rmtree(fine_tuned_path)

        shutil.copytree(pretrained_path, fine_tuned_path)

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