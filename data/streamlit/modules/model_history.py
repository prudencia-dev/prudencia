from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from services.api import PrudenciaAPI


# -------------------------------------------------------------------
# Fonctions de formatage
# -------------------------------------------------------------------

def _format_percentage(value: Any) -> str:
    """Transforme une valeur comprise entre 0 et 1 en pourcentage."""

    if value is None:
        return "N/A"

    try:
        return f"{float(value) * 100:.2f} %"
    except (TypeError, ValueError):
        return "N/A"


def _format_decimal(value: Any) -> str:
    """Transforme une valeur numérique en nombre à quatre décimales."""

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_duration(milliseconds: Any) -> str:
    """Transforme une durée en millisecondes en secondes ou minutes."""

    try:
        seconds = int(milliseconds or 0) / 1000
    except (TypeError, ValueError):
        return "N/A"

    if seconds < 60:
        return f"{seconds:.1f} s"

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    return f"{minutes} min {remaining_seconds} s"


def _as_bool(value: Any) -> bool:
    """Interprète correctement les booléens venant de PostgreSQL/JSON."""

    return (
        value is True
        or str(value).strip().lower()
        in {"true", "1", "yes", "oui"}
    )


def _extract_metrics(
    output_data: dict[str, Any],
) -> dict[str, Any]:
    """Récupère les métriques dans les différents formats historiques."""

    metrics = output_data.get("metrics")

    if isinstance(metrics, dict):
        return metrics

    training = output_data.get("training")

    if isinstance(training, dict):
        nested_metrics = training.get("metrics")

        if isinstance(nested_metrics, dict):
            return nested_metrics

        evaluation = training.get("evaluation")

        if isinstance(evaluation, dict):
            return evaluation

    return output_data


def _find_metric(
    metrics: dict[str, Any],
    name: str,
) -> Any:
    """Recherche une métrique avec ou sans préfixe eval_."""

    value = metrics.get(name)

    if value is not None:
        return value

    return metrics.get(f"eval_{name}")


# -------------------------------------------------------------------
# Configuration Fine-Tuning
# -------------------------------------------------------------------

def _display_fine_tuning_configuration(
    input_data: dict[str, Any],
) -> None:
    """Affiche les hyperparamètres d'un entraînement Fine-Tuning."""

    st.subheader("⚙️ Configuration expérimentale")

    row_1 = st.columns(5)
    row_1[0].metric(
        "Texte",
        input_data.get("text_column", "—"),
    )
    row_1[1].metric(
        "Label",
        input_data.get("label_column", "—"),
    )
    row_1[2].metric(
        "Epochs",
        input_data.get("epochs", "—"),
    )
    row_1[3].metric(
        "Batch réel",
        input_data.get("batch_size", "—"),
    )
    row_1[4].metric(
        "Batch effectif",
        input_data.get("effective_batch_size", "—"),
    )

    row_2 = st.columns(5)
    row_2[0].metric(
        "Accumulation",
        input_data.get(
            "gradient_accumulation_steps",
            "—",
        ),
    )
    row_2[1].metric(
        "Learning rate",
        input_data.get("learning_rate", "—"),
    )
    row_2[2].metric(
        "Seed",
        input_data.get("seed", "—"),
    )
    row_2[3].metric(
        "Max tokens",
        input_data.get("max_length", "—"),
    )
    row_2[4].metric(
        "Early stopping",
        input_data.get(
            "early_stopping_patience",
            "—",
        ),
    )

    row_3 = st.columns(4)
    row_3[0].metric(
        "Weight decay",
        input_data.get("weight_decay", "—"),
    )
    row_3[1].metric(
        "Warmup ratio",
        input_data.get("warmup_ratio", "—"),
    )
    row_3[2].metric(
        "Métrique cible",
        input_data.get(
            "metric_for_best_model",
            "—",
        ),
    )
    row_3[3].metric(
        "Poids de classes",
        (
            "Oui"
            if _as_bool(
                input_data.get("use_class_weights")
            )
            else "Non"
        ),
    )


# -------------------------------------------------------------------
# Configuration Machine Learning
# -------------------------------------------------------------------

def _display_machine_learning_configuration(
    input_data: dict[str, Any],
) -> None:
    """Affiche les hyperparamètres du Random Forest."""

    st.subheader("⚙️ Configuration expérimentale")

    row_1 = st.columns(5)
    row_1[0].metric(
        "Arbres",
        input_data.get("n_estimators", "—"),
    )
    row_1[1].metric(
        "Profondeur",
        (
            input_data.get("max_depth")
            if input_data.get("max_depth") is not None
            else "Automatique"
        ),
    )
    row_1[2].metric(
        "Min. division",
        input_data.get("min_samples_split", "—"),
    )
    row_1[3].metric(
        "Min. feuille",
        input_data.get("min_samples_leaf", "—"),
    )
    row_1[4].metric(
        "Features / division",
        (
            input_data.get("max_features")
            if input_data.get("max_features") is not None
            else "Toutes"
        ),
    )

    row_2 = st.columns(5)
    row_2[0].metric(
        "Critère",
        input_data.get("criterion", "—"),
    )
    row_2[1].metric(
        "Bootstrap",
        (
            "Oui"
            if _as_bool(input_data.get("bootstrap"))
            else "Non"
        ),
    )
    row_2[2].metric(
        "Poids des classes",
        input_data.get("class_weight") or "Aucun",
    )
    row_2[3].metric(
        "Random State",
        input_data.get("random_state", "—"),
    )

    test_size = input_data.get("test_size")

    row_2[4].metric(
        "Jeu Test",
        (
            _format_percentage(test_size)
            if test_size is not None
            else "—"
        ),
    )

    row_3 = st.columns(4)
    row_3[0].metric(
        "Target",
        input_data.get("target_column", "—"),
    )
    row_3[1].metric(
        "Features",
        len(
            input_data.get(
                "feature_columns",
                [],
            )
            or []
        ),
    )
    row_3[2].metric(
        "Train",
        input_data.get("train_rows", "—"),
    )
    row_3[3].metric(
        "Test",
        input_data.get("test_rows", "—"),
    )


# -------------------------------------------------------------------
# Métriques
# -------------------------------------------------------------------

def _display_global_metrics(
    metrics: dict[str, Any],
    model_type: str,
) -> None:
    """Affiche les métriques globales selon le type de modèle."""

    st.subheader("📈 Métriques globales")

    if model_type == "machine_learning":
        row_1 = st.columns(4)

        row_1[0].metric(
            "Accuracy",
            _format_percentage(
                _find_metric(metrics, "accuracy")
            ),
        )
        row_1[1].metric(
            "Balanced Accuracy",
            _format_percentage(
                _find_metric(
                    metrics,
                    "balanced_accuracy",
                )
            ),
        )
        row_1[2].metric(
            "Precision pondérée",
            _format_percentage(
                _find_metric(
                    metrics,
                    "precision_weighted",
                )
                or _find_metric(
                    metrics,
                    "precision",
                )
            ),
        )
        row_1[3].metric(
            "Recall pondéré",
            _format_percentage(
                _find_metric(
                    metrics,
                    "recall_weighted",
                )
                or _find_metric(
                    metrics,
                    "recall",
                )
            ),
        )

        row_2 = st.columns(3)

        row_2[0].metric(
            "F1 pondéré",
            _format_percentage(
                _find_metric(
                    metrics,
                    "f1_weighted",
                )
                or _find_metric(metrics, "f1")
            ),
        )
        row_2[1].metric(
            "Macro-F1",
            _format_percentage(
                _find_metric(metrics, "macro_f1")
            ),
        )
        row_2[2].metric(
            "MCC",
            _format_decimal(
                _find_metric(metrics, "mcc")
            ),
        )

        return

    columns = st.columns(6)

    columns[0].metric(
        "Accuracy",
        _format_percentage(
            _find_metric(metrics, "accuracy")
        ),
    )
    columns[1].metric(
        "Precision pondérée",
        _format_percentage(
            _find_metric(
                metrics,
                "precision_weighted",
            )
            or _find_metric(
                metrics,
                "precision",
            )
        ),
    )
    columns[2].metric(
        "Recall pondéré",
        _format_percentage(
            _find_metric(
                metrics,
                "recall_weighted",
            )
            or _find_metric(metrics, "recall")
        ),
    )
    columns[3].metric(
        "F1 pondéré",
        _format_percentage(
            _find_metric(
                metrics,
                "f1_weighted",
            )
            or _find_metric(metrics, "f1")
        ),
    )
    columns[4].metric(
        "Macro-F1",
        _format_percentage(
            _find_metric(metrics, "macro_f1")
        ),
    )

    loss = _find_metric(metrics, "loss")

    columns[5].metric(
        "Loss",
        _format_decimal(loss),
    )


def _display_per_class_metrics(
    metrics: dict[str, Any],
) -> None:
    """Affiche les performances de chaque classe."""

    per_class = metrics.get("per_class")

    if not isinstance(per_class, dict) or not per_class:
        return

    st.subheader("🎯 Métriques par classe")

    rows = []

    for class_name, values in per_class.items():
        if not isinstance(values, dict):
            continue

        rows.append(
            {
                "Classe": class_name,
                "Precision": _format_percentage(
                    values.get("precision")
                ),
                "Recall": _format_percentage(
                    values.get("recall")
                ),
                "F1": _format_percentage(
                    values.get("f1")
                ),
                "Support": values.get("support", 0),
            }
        )

    if not rows:
        return

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


def _display_confusion_matrix(
    metrics: dict[str, Any],
) -> None:
    """Affiche la matrice de confusion stockée dans l'historique."""

    matrix = metrics.get("confusion_matrix")
    class_names = metrics.get("class_names")

    if not matrix or not class_names:
        return

    st.subheader("🧩 Matrice de confusion")

    st.dataframe(
        pd.DataFrame(
            matrix,
            index=[
                f"Réel : {name}"
                for name in class_names
            ],
            columns=[
                f"Prédit : {name}"
                for name in class_names
            ],
        ),
        use_container_width=True,
    )


# -------------------------------------------------------------------
# Informations propres au Machine Learning
# -------------------------------------------------------------------

def _extract_feature_importance(
    output_data: dict[str, Any],
    training: dict[str, Any],
) -> list[dict[str, Any]]:
    """Récupère l'importance des variables dans les formats connus."""

    candidates = [
        output_data.get("feature_importance"),
        training.get("feature_importance"),
    ]

    for candidate in candidates:
        if isinstance(candidate, list):
            return [
                row
                for row in candidate
                if isinstance(row, dict)
            ]

    return []


def _display_feature_importance(
    output_data: dict[str, Any],
    training: dict[str, Any],
) -> None:
    """Affiche les variables les plus influentes du Random Forest."""

    feature_importance = _extract_feature_importance(
        output_data=output_data,
        training=training,
    )

    if not feature_importance:
        return

    dataframe = pd.DataFrame(feature_importance)

    if not {
        "feature",
        "importance",
    }.issubset(dataframe.columns):
        return

    dataframe = dataframe[
        ["feature", "importance"]
    ].copy()

    dataframe["importance"] = pd.to_numeric(
        dataframe["importance"],
        errors="coerce",
    )

    dataframe = (
        dataframe
        .dropna(subset=["importance"])
        .sort_values(
            by="importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    if dataframe.empty:
        return

    st.subheader("🌲 Importance des variables")

    maximum = max(
        0.01,
        float(dataframe["importance"].max()),
    )

    st.dataframe(
        dataframe.head(20).rename(
            columns={
                "feature": "Variable encodée",
                "importance": "Importance",
            }
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Importance": st.column_config.ProgressColumn(
                "Importance",
                min_value=0.0,
                max_value=maximum,
                format="%.4f",
            ),
        },
    )

    st.caption(
        "Les variables catégorielles sont décomposées par "
        "le OneHotEncoder. Une Feature peut donc apparaître "
        "sous plusieurs modalités."
    )


def _display_dataset_quality(
    training: dict[str, Any],
    input_data: dict[str, Any],
) -> None:
    """Affiche les contrôles de qualité enregistrés pour le Dataset."""

    quality = training.get("dataset_quality", {})

    if not isinstance(quality, dict) or not quality:
        return

    st.subheader("🔎 Qualité du Dataset")

    columns = st.columns(4)

    columns[0].metric(
        "Doublons",
        quality.get("duplicates", 0),
    )
    columns[1].metric(
        "Valeurs manquantes",
        quality.get("missing_values", 0),
    )
    columns[2].metric(
        "Part manquante",
        _format_percentage(
            quality.get("missing_values_ratio")
        ),
    )

    removed_columns = quality.get(
        "constant_columns_removed",
        [],
    )

    columns[3].metric(
        "Colonnes constantes",
        len(removed_columns or []),
    )

    if removed_columns:
        st.warning(
            "Colonnes constantes supprimées : "
            + ", ".join(map(str, removed_columns))
        )

    st.write(
        "**Séparation stratifiée :** "
        + (
            "Oui"
            if _as_bool(
                input_data.get(
                    "stratification_used"
                )
            )
            else "Non"
        )
    )


def _display_target_distribution(
    output_data: dict[str, Any],
) -> None:
    """Affiche la distribution des classes du Dataset."""

    distribution = output_data.get(
        "target_distribution",
        {},
    )

    if not isinstance(distribution, dict) or not distribution:
        return

    st.subheader("📊 Distribution des classes")

    dataframe = pd.DataFrame(
        {
            "Classe": list(distribution.keys()),
            "Nombre d'exemples": list(
                distribution.values()
            ),
        }
    )

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )


def _display_warnings(
    output_data: dict[str, Any],
) -> None:
    """Affiche les avertissements sauvegardés avec le run."""

    warnings = output_data.get("warnings", [])

    if not isinstance(warnings, list) or not warnings:
        return

    st.subheader("⚠️ Avertissements")

    for warning in warnings:
        st.warning(str(warning))


# -------------------------------------------------------------------
# Informations propres au Fine-Tuning
# -------------------------------------------------------------------

def _display_training_state(
    training: dict[str, Any],
) -> None:
    """Affiche l'état final d'un entraînement Fine-Tuning."""

    training_state = training.get(
        "training_state",
        {},
    )

    if not isinstance(training_state, dict) or not training_state:
        return

    st.subheader("⏱️ État du run")

    state_columns = st.columns(4)

    state_columns[0].metric(
        "Epochs demandés",
        training_state.get(
            "epochs_requested",
            "—",
        ),
    )
    state_columns[1].metric(
        "Epochs réalisés",
        training_state.get(
            "epochs_completed",
            "—",
        ),
    )
    state_columns[2].metric(
        "Meilleure métrique",
        _format_decimal(
            training_state.get("best_metric")
        ),
    )
    state_columns[3].metric(
        "Early stopping",
        (
            "Actif"
            if training_state.get(
                "early_stopping_enabled"
            )
            else "Inactif"
        ),
    )


def _display_token_statistics(
    training: dict[str, Any],
) -> None:
    """Affiche les statistiques de longueur des textes."""

    token_statistics = training.get(
        "token_length_statistics",
        {},
    )

    if (
        not isinstance(token_statistics, dict)
        or not token_statistics
    ):
        return

    st.subheader("📏 Longueur des textes")

    token_columns = st.columns(5)

    token_columns[0].metric(
        "Médiane",
        token_statistics.get("median", "—"),
    )
    token_columns[1].metric(
        "P95",
        token_statistics.get("p95", "—"),
    )
    token_columns[2].metric(
        "Maximum",
        token_statistics.get("maximum", "—"),
    )
    token_columns[3].metric(
        "Max utilisé",
        token_statistics.get(
            "max_length_used",
            "—",
        ),
    )
    token_columns[4].metric(
        "Tronqués",
        (
            f"{token_statistics.get('texts_truncated', 0)} / "
            f"{token_statistics.get('total_texts', 0)}"
        ),
    )


# -------------------------------------------------------------------
# Interface principale
# -------------------------------------------------------------------

def render_model_history() -> None:
    """Affiche l'historique détaillé des entraînements PRUDENCIA."""

    st.subheader("📜 Historique des entraînements")

    st.info(
        "Chaque run conserve son Dataset, ses hyperparamètres, "
        "ses métriques, sa durée et ses données techniques "
        "de traçabilité."
    )

    col_refresh, col_caption = st.columns([1, 4])

    with col_refresh:
        if st.button(
            "🔄 Actualiser",
            key="refresh_model_history",
            use_container_width=True,
        ):
            st.rerun()

    with col_caption:
        st.caption(
            "Données issues de prudencia.model_executions."
        )

    try:
        result = PrudenciaAPI.get(
            "/fine-tuning/history"
        )
        executions = result.get("history", [])

    except Exception as error:
        st.error(
            "Impossible de charger l'historique : "
            f"{error}"
        )
        return

    if not executions:
        st.info("Aucun entraînement enregistré.")
        return

    successful_trainings = [
        execution
        for execution in executions
        if execution.get("success")
        and execution.get("execution_type") == "training"
        and execution.get("model_type") in {
            "fine_tuning",
            "deep_learning",
            "machine_learning",
        }
    ]

    summary_columns = st.columns(3)

    summary_columns[0].metric(
        "Runs enregistrés",
        len(executions),
    )
    summary_columns[1].metric(
        "Entraînements réussis",
        len(successful_trainings),
    )
    summary_columns[2].metric(
        "Modèles actifs",
        sum(
            1
            for execution in executions
            if _as_bool(
                execution.get("is_active")
            )
        ),
    )

    st.divider()

    for execution in executions:
        output_data = execution.get(
            "output_data",
            {},
        ) or {}

        if not isinstance(output_data, dict):
            output_data = {}

        metrics = _extract_metrics(output_data)

        if not isinstance(metrics, dict):
            metrics = {}

        training = output_data.get(
            "training",
            {},
        )

        if not isinstance(training, dict):
            training = {}

        execution_type = execution.get(
            "execution_type",
            "training",
        )

        model_type = execution.get(
            "model_type",
            "inconnu",
        )

        if model_type in {
            "fine_tuning",
            "deep_learning",
        }:
            model_label = (
                "Deep Learning / Fine-Tuning"
            )
            model_icon = "🧠"

        elif model_type == "machine_learning":
            model_label = "Machine Learning"
            model_icon = "🌲"

        else:
            model_label = str(model_type)
            model_icon = "⚙️"

        action_labels = {
            "reset": (
                "Réinitialisation",
                "🟠",
            ),
            "evaluation": (
                "Évaluation",
                "🔵",
            ),
            "benchmark": (
                "Benchmark",
                "🟣",
            ),
            "training": (
                "Entraînement",
                "🟢",
            ),
        }

        action_label, action_icon = action_labels.get(
            execution_type,
            ("Entraînement", "🟢"),
        )

        is_active = _as_bool(
            execution.get("is_active", False)
        )

        active_label = (
            " — 🟢 ACTIF"
            if is_active
            else ""
        )

        model_name = execution.get(
            "model_name",
            "Modèle inconnu",
        )

        title = (
            f"{action_icon} {action_label} — "
            f"{model_icon} {model_label} — "
            f"{model_name}{active_label}"
        )

        with st.expander(title):
            header_columns = st.columns(4)

            header_columns[0].metric(
                "Modèle",
                model_name,
            )
            header_columns[1].metric(
                "Version",
                execution.get(
                    "model_version",
                    "N/A",
                ),
            )
            header_columns[2].metric(
                "Statut",
                (
                    "✅ Succès"
                    if _as_bool(
                        execution.get("success")
                    )
                    else "❌ Échec"
                ),
            )
            header_columns[3].metric(
                "Durée",
                _format_duration(
                    execution.get(
                        "execution_time_ms",
                        0,
                    )
                ),
            )

            if is_active:
                st.success(
                    "Modèle actuellement actif "
                    "dans PRUDENCIA."
                )

            st.divider()

            st.subheader("🧠 Modèle et Dataset")

            detail_columns = st.columns(4)

            detail_columns[0].metric(
                "Dataset",
                execution.get("dataset_name")
                or "Aucun",
            )
            detail_columns[1].metric(
                "Lignes",
                execution.get(
                    "dataset_rows",
                    0,
                )
                or 0,
            )
            detail_columns[2].metric(
                "Train",
                training.get(
                    "training_examples",
                    "—",
                ),
            )
            detail_columns[3].metric(
                "Validation / Test",
                training.get(
                    "validation_examples",
                    "—",
                ),
            )

            base_model = training.get(
                "base_model",
                {},
            )

            if isinstance(base_model, dict) and base_model:
                st.write(
                    "**Checkpoint Hugging Face :** "
                    f"`{base_model.get('huggingface_id', '—')}`"
                )
                st.write(
                    "**Architecture :** "
                    f"`{base_model.get('architecture', '—')}`"
                )

            input_data = execution.get(
                "input_data",
                {},
            ) or {}

            if not isinstance(input_data, dict):
                input_data = {}

            if model_type == "machine_learning":
                _display_machine_learning_configuration(
                    input_data
                )
            else:
                _display_fine_tuning_configuration(
                    input_data
                )

            if execution_type == "training":
                _display_global_metrics(
                    metrics=metrics,
                    model_type=model_type,
                )

                _display_per_class_metrics(metrics)
                _display_confusion_matrix(metrics)

                if model_type == "machine_learning":
                    _display_feature_importance(
                        output_data=output_data,
                        training=training,
                    )

                    _display_dataset_quality(
                        training=training,
                        input_data=input_data,
                    )

                    _display_target_distribution(
                        output_data
                    )

                    _display_warnings(
                        output_data
                    )

                else:
                    _display_training_state(
                        training
                    )

                    _display_token_statistics(
                        training
                    )

            st.divider()

            st.write(
                "**Date :** "
                f"{execution.get('executed_at', 'N/A')}"
            )

            if execution.get("error_message"):
                st.error(
                    execution["error_message"]
                )

            with st.expander(
                "Données techniques",
                expanded=False,
            ):
                st.json(execution)