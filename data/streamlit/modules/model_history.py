from __future__ import annotations

from typing import Any

import streamlit as st

from services.api import PrudenciaAPI


def _format_percentage(value: Any) -> str:
    if value is None:
        return "N/A"

    try:
        return f"{float(value) * 100:.2f} %"
    except (TypeError, ValueError):
        return "N/A"


def _format_duration(milliseconds: Any) -> str:
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
    return (
        value is True
        or str(value).strip().lower()
        in {
            "true",
            "1",
            "yes",
            "oui",
        }
    )


def _extract_metrics(
    output_data: dict[str, Any],
) -> dict[str, Any]:
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
    return (
        metrics.get(name)
        if metrics.get(name) is not None
        else metrics.get(f"eval_{name}")
    )


def render_model_history() -> None:
    """
    Affiche l'historique des Fine-Tuning enregistrés
    dans PostgreSQL.
    """

    st.subheader("📜 Historique Fine-Tuning")

    st.info(
        "Cette section présente les entraînements, "
        "évaluations et réinitialisations enregistrés."
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
            "Données issues de "
            "`prudencia.model_executions`."
        )

    try:
        result = PrudenciaAPI.get(
            "/fine-tuning/history"
        )

        executions = result.get(
            "history",
            [],
        )

    except Exception as error:
        st.error(
            "Impossible de charger l'historique : "
            f"{error}"
        )
        return

    if not executions:
        st.info(
            "Aucun Fine-Tuning enregistré."
        )
        return

    successful_trainings = [
        execution
        for execution in executions
        if execution.get("success")
        and execution.get("execution_type") == "training"
        and execution.get("model_type") == "fine_tuning"
    ]

    st.metric(
        "Entraînements enregistrés",
        len(successful_trainings),
    )

    st.divider()

    for execution in executions:
        output_data = execution.get(
            "output_data",
            {},
        ) or {}

        if not isinstance(output_data, dict):
            output_data = {}

        metrics = _extract_metrics(
            output_data
        )

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
            model_label = "Fine-Tuning"
            model_icon = "🧠"

        else:
            model_label = str(model_type)
            model_icon = "⚙️"

        if execution_type == "reset":
            action_label = "Réinitialisation"
            action_icon = "🟠"

        elif execution_type == "evaluation":
            action_label = "Évaluation"
            action_icon = "🔵"

        elif execution_type == "benchmark":
            action_label = "Benchmark"
            action_icon = "🟣"

        else:
            action_label = "Entraînement"
            action_icon = "🟢"

        is_active = _as_bool(
            execution.get(
                "is_active",
                False,
            )
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
            f"{model_name}"
            f"{active_label}"
        )

        with st.expander(title):
            col_model, col_version, col_status = st.columns(3)

            col_model.write(
                "**Modèle**  \n"
                f"{model_name}"
            )

            col_version.write(
                "**Version**  \n"
                f"{execution.get('model_version', 'N/A')}"
            )

            success = _as_bool(
                execution.get(
                    "success",
                    False,
                )
            )

            col_status.write(
                "**Statut**  \n"
                f"{'✅ Succès' if success else '❌ Échec'}"
            )

            if is_active:
                st.success(
                    "Modèle actuellement actif dans PRUDENCIA."
                )

            # ----------------------------------------------------
            # Informations sur le modèle
            # ----------------------------------------------------

            st.divider()

            st.subheader("🧠 Modèle")

            model_columns = st.columns(3)

            model_columns[0].metric(
                "Nom",
                execution.get("model_name", "—"),
            )

            model_columns[1].metric(
                "Version",
                execution.get("model_version", "—"),
            )

            hf_model = (
                execution.get("output_data", {})
                .get("training", {})
                .get("base_model", {})
                .get("huggingface_id", "—")
            )

            model_columns[2].write(
                f"**Hugging Face**\n\n`{hf_model}`"
            )

            architecture = (
                execution.get("output_data", {})
                .get("training", {})
                .get("base_model", {})
                .get("architecture", "—")
            )

            st.write(
                f"**Architecture :** `{architecture}`"
            )

            st.divider()

            st.divider()

            col_dataset, col_rows, col_duration = st.columns(3)

            col_dataset.write(
                "**Dataset**  \n"
                f"{execution.get('dataset_name') or 'Aucun'}"
            )


            input_data = execution.get(
                "input_data",
                {},
            ) or {}

            st.divider()
            st.subheader("⚙️ Configuration")

            config1, config2, config3, config4, config5 = st.columns(5)

            config1.metric(
                "Texte",
                input_data.get("text_column", "—"),
            )

            config2.metric(
                "Label",
                input_data.get("label_column", "—"),
            )

            config3.metric(
                "Epochs",
                input_data.get("epochs", "—"),
            )

            config4.metric(
                "Batch",
                input_data.get("batch_size", "—"),
            )

            config5.metric(
                "LR",
                input_data.get("learning_rate", "—"),
            )

            col_rows.metric(
                "Exemples",
                execution.get(
                    "dataset_rows",
                    0,
                ) or 0,
            )

            col_duration.metric(
                "Durée",
                _format_duration(
                    execution.get(
                        "execution_time_ms",
                        0,
                    )
                ),
            )

            if execution_type == "training":
                st.divider()

                col_accuracy, col_precision, col_recall, col_f1 = (
                    st.columns(4)
                )

                col_accuracy.metric(
                    "Accuracy",
                    _format_percentage(
                        _find_metric(
                            metrics,
                            "accuracy",
                        )
                    ),
                )

                col_precision.metric(
                    "Precision",
                    _format_percentage(
                        _find_metric(
                            metrics,
                            "precision",
                        )
                    ),
                )

                col_recall.metric(
                    "Recall",
                    _format_percentage(
                        _find_metric(
                            metrics,
                            "recall",
                        )
                    ),
                )

                col_f1.metric(
                    "F1-Score",
                    _format_percentage(
                        _find_metric(
                            metrics,
                            "f1",
                        )
                    ),
                )

                loss = _find_metric(
                    metrics,
                    "loss",
                )

                if isinstance(
                    loss,
                    (int, float),
                ):
                    st.metric(
                        "Loss",
                        f"{float(loss):.4f}",
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
            ):
                st.json(execution)
