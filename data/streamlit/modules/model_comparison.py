from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("PRUDENCIA_API_URL", os.getenv("API_URL", "http://api:8000"))


@st.cache_data(ttl=30)
def get_training_history() -> list[dict[str, Any]]:
    """Récupère les expériences enregistrées par l'API."""

    response = requests.get(f"{API_URL}/training/history", timeout=10)
    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("history", "executions", "items", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def nested_metric(execution: dict[str, Any], name: str) -> Any:
    """Lit une mesure quel que soit le format du journal."""

    for source in (
        execution,
        execution.get("metrics", {}),
        execution.get("evaluation_metrics", {}),
        execution.get("results", {}),
    ):
        if isinstance(source, dict):
            value = source.get(name)
            if value is None:
                value = source.get(f"eval_{name}")
            if value is not None:
                return value
    return None


def is_deep_learning(execution: dict[str, Any]) -> bool:
    """Conserve les entraînements de modèles de texte."""

    text = " ".join(
        str(execution.get(key, "")).lower()
        for key in ("model_type", "model_name", "training_type")
    )
    return any(name in text for name in ("deep", "camembert", "juribert", "transformer"))


def render_model_comparison() -> None:
    """Compare simplement les expériences Deep Learning disponibles."""

    st.subheader("🏆 Comparaison des expériences")
    st.caption(
        "Cette vue aide à comparer les versions entraînées dans des conditions proches."
    )

    try:
        history = [item for item in get_training_history() if is_deep_learning(item)]
    except requests.RequestException as error:
        st.error("Impossible de récupérer l'historique des entraînements.")
        st.code(str(error))
        return

    if not history:
        st.info("Aucune expérience Deep Learning n'est encore disponible.")
        return

    rows = []
    for execution in history:
        rows.append(
            {
                "Modèle": execution.get("model_name", "Non renseigné"),
                "Version": execution.get("model_version", "—"),
                "Dataset": execution.get("dataset_name", "—"),
                "Accuracy": nested_metric(execution, "accuracy"),
                "Précision": nested_metric(execution, "precision"),
                "Rappel": nested_metric(execution, "recall"),
                "Macro-F1": nested_metric(execution, "macro_f1")
                or nested_metric(execution, "f1"),
                "Durée (s)": execution.get("training_time_seconds")
                or execution.get("duration_seconds"),
            }
        )

    comparison = pd.DataFrame(rows)
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.info(
        "Pour une comparaison juste, utilisez le même jeu de données, "
        "la même séparation train/test et les mêmes graines aléatoires."
    )

    st.download_button(
        "📥 Exporter la comparaison",
        comparison.to_csv(index=False),
        "comparaison_modeles_dl.csv",
        "text/csv",
    )

    with st.expander("Afficher les données enregistrées"):
        st.json(history)
