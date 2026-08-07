from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")
HISTORY_ENDPOINT = "/fine-tuning/history"


def _format_percentage(value: Any) -> str:
    """Formate une métrique comprise entre zéro et un."""

    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "—"


def _format_duration(milliseconds: Any) -> str:
    """Formate une durée exprimée en millisecondes."""

    try:
        seconds = float(milliseconds) / 1000
    except (TypeError, ValueError):
        return "—"

    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _load_history() -> list[dict[str, Any]]:
    response = requests.get(
        f"{API_URL}{HISTORY_ENDPOINT}",
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    history = payload.get("history", [])
    return [item for item in history if isinstance(item, dict)]


def _display_metrics(output_data: dict[str, Any]) -> None:
    metrics = _as_dict(output_data.get("metrics"))
    if not metrics:
        metrics = output_data

    columns = st.columns(5)
    metric_names = (
        ("Accuracy", "accuracy"),
        ("Précision", "precision_weighted"),
        ("Rappel", "recall_weighted"),
        ("F1", "f1_weighted"),
        ("Macro-F1", "macro_f1"),
    )

    for column, (label, key) in zip(columns, metric_names, strict=True):
        value = metrics.get(key)
        if value is None and key.endswith("_weighted"):
            value = metrics.get(key.removesuffix("_weighted"))
        column.metric(label, _format_percentage(value))

    per_class = metrics.get("per_class")
    if isinstance(per_class, dict) and per_class:
        rows = [
            {"Classe": name, **_as_dict(values)}
            for name, values in per_class.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _display_configuration(input_data: dict[str, Any]) -> None:
    st.write("**Configuration**")
    columns = st.columns(4)
    columns[0].metric("Époques", input_data.get("epochs", "—"))
    columns[1].metric("Batch", input_data.get("batch_size", "—"))
    columns[2].metric(
        "Learning rate",
        input_data.get("learning_rate", "—"),
    )
    columns[3].metric(
        "Validation",
        _format_percentage(input_data.get("validation_split")),
    )


def render_model_history() -> None:
    """Affiche uniquement l'historique Deep Learning et fine-tuning."""

    st.header("📜 Historique des entraînements")
    st.caption("Exécutions JuriBERT et autres modèles Deep Learning.")

    try:
        executions = _load_history()
    except requests.RequestException as error:
        st.error(f"Impossible de charger l'historique : {error}")
        return

    executions = [
        execution
        for execution in executions
        if execution.get("model_type") in {"fine_tuning", "deep_learning"}
    ]

    if not executions:
        st.info("Aucun entraînement Deep Learning enregistré.")
        return

    summary = st.columns(3)
    summary[0].metric("Runs enregistrés", len(executions))
    summary[1].metric(
        "Entraînements réussis",
        sum(1 for item in executions if item.get("success")),
    )
    summary[2].metric(
        "Modèles actifs",
        sum(1 for item in executions if item.get("is_active")),
    )

    for execution in executions:
        model_name = execution.get("model_name", "Modèle Deep Learning")
        version = execution.get("model_version", "—")
        execution_type = execution.get("execution_type", "training")
        success = bool(execution.get("success"))
        icon = "✅" if success else "❌"

        with st.expander(
            f"{icon} {model_name} · {version} · {execution_type}",
            expanded=False,
        ):
            metadata = st.columns(3)
            metadata[0].metric(
                "Dataset",
                execution.get("dataset_name", "—"),
            )
            metadata[1].metric(
                "Exemples",
                execution.get("dataset_rows", "—"),
            )
            metadata[2].metric(
                "Durée",
                _format_duration(execution.get("execution_time_ms")),
            )

            _display_configuration(_as_dict(execution.get("input_data")))
            _display_metrics(_as_dict(execution.get("output_data")))

            error_message = execution.get("error_message")
            if error_message:
                st.error(str(error_message))
