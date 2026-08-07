from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://api:8000")
EXPECTED_MODELS = {"camembert", "camembertv2", "juribert"}
MODEL_LABELS = {
    "camembert": "CamemBERT",
    "camembertv2": "CamemBERTv2",
    "juribert": "JuriBERT",
}


st.set_page_config(
    page_title="Benchmark NLP",
    page_icon="📊",
    layout="wide",
)


def _as_dict(value: Any) -> dict[str, Any]:
    """Retourne un dictionnaire exploitable à partir d'un champ JSONB."""

    return value if isinstance(value, dict) else {}


def _metric(metrics: dict[str, Any], *names: str) -> float | None:
    """Extrait une métrique numérique avec ses variantes Hugging Face."""

    for name in names:
        for candidate in (name, f"eval_{name}"):
            value = metrics.get(candidate)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _percentage(value: Any) -> str:
    return f"{float(value):.2%}" if isinstance(value, (int, float)) else "—"


def _duration(milliseconds: Any) -> str:
    if not isinstance(milliseconds, (int, float)):
        return "—"
    return f"{float(milliseconds) / 60000:.1f} min"


@st.cache_data(ttl=20)
def _load_runs() -> list[dict[str, Any]]:
    """Charge l'historique persistant des entraînements."""

    response = requests.get(
        f"{API_URL}/fine-tuning/history",
        timeout=30,
    )
    response.raise_for_status()
    history = response.json().get("history", [])
    return [run for run in history if isinstance(run, dict)]


def _normalize_run(run: dict[str, Any]) -> dict[str, Any]:
    """Aplatit un run afin de faciliter tableaux, filtres et graphiques."""

    inputs = _as_dict(run.get("input_data"))
    outputs = _as_dict(run.get("output_data"))
    metrics = _as_dict(outputs.get("metrics"))
    training = _as_dict(outputs.get("training"))
    if not metrics:
        metrics = _as_dict(training.get("metrics"))

    model_id = str(run.get("model_name") or "inconnu")
    return {
        "run_id": str(run.get("id") or "—"),
        "date": str(run.get("executed_at") or "—"),
        "model_id": model_id,
        "model": MODEL_LABELS.get(model_id, model_id),
        "model_version": str(run.get("model_version") or "—"),
        "dataset": str(run.get("dataset_name") or "—"),
        "dataset_rows": run.get("dataset_rows"),
        "dataset_sha256": inputs.get("dataset_sha256"),
        "benchmark_signature": inputs.get("benchmark_signature"),
        "source_code_version": inputs.get("source_code_version"),
        "source_code_sha256": inputs.get("source_code_sha256"),
        "huggingface_id": inputs.get("model_huggingface_id"),
        "profile": inputs.get("training_profile", "Ancien run"),
        "epochs": inputs.get("epochs"),
        "batch_size": inputs.get("batch_size"),
        "gradient_accumulation": inputs.get("gradient_accumulation_steps"),
        "effective_batch_size": inputs.get("effective_batch_size"),
        "learning_rate": inputs.get("learning_rate"),
        "max_length": inputs.get("max_length"),
        "seed": inputs.get("seed"),
        "early_stopping": inputs.get("early_stopping_patience"),
        "weight_decay": inputs.get("weight_decay"),
        "warmup_ratio": inputs.get("warmup_ratio"),
        "selection_metric": inputs.get("metric_for_best_model"),
        "class_weights": inputs.get("use_class_weights"),
        "accuracy": _metric(metrics, "accuracy"),
        "precision": _metric(metrics, "precision_weighted", "precision"),
        "recall": _metric(metrics, "recall_weighted", "recall"),
        "f1_weighted": _metric(metrics, "f1_weighted", "f1"),
        "macro_f1": _metric(metrics, "macro_f1"),
        "loss": _metric(metrics, "loss"),
        "duration_ms": run.get("execution_time_ms"),
        "is_active": bool(run.get("is_active")),
        "success": bool(run.get("success")),
        "metrics": metrics,
    }


st.title("📊 Tableau de bord — Benchmark NLP")
st.caption(
    "Comparaison reproductible de CamemBERT, CamemBERTv2 et JuriBERT "
    "à partir des entraînements enregistrés dans PostgreSQL."
)

try:
    raw_runs = _load_runs()
except requests.RequestException as error:
    st.error(f"Impossible de charger l'historique des entraînements : {error}")
    st.stop()

runs = [
    _normalize_run(run)
    for run in raw_runs
    if run.get("execution_type") == "training"
    and run.get("model_type") in {"deep_learning", "fine_tuning"}
]

if not runs:
    st.info(
        "Aucun entraînement n'est disponible. Lancez un run depuis la page "
        "Fine-Tuning avec le profil « Benchmark reproductible »."
    )
    st.stop()

successful_runs = [run for run in runs if run["success"]]
tested_models = {run["model_id"] for run in successful_runs}
best_run = max(
    successful_runs,
    key=lambda run: run["macro_f1"] if run["macro_f1"] is not None else -1,
    default=None,
)

summary = st.columns(4)
summary[0].metric("Runs enregistrés", len(runs))
summary[1].metric("Runs réussis", len(successful_runs))
summary[2].metric(
    "Modèles BERT testés",
    f"{len(tested_models & EXPECTED_MODELS)} / {len(EXPECTED_MODELS)}",
)
summary[3].metric(
    "Meilleur macro-F1",
    _percentage(best_run["macro_f1"] if best_run else None),
    best_run["model"] if best_run else None,
)

st.subheader("Filtres de comparaison")
filter_columns = st.columns(3)
model_options = sorted({run["model"] for run in runs})
selected_models = filter_columns[0].multiselect(
    "Modèles",
    model_options,
    default=model_options,
)
dataset_options = sorted({run["dataset"] for run in runs})
selected_dataset = filter_columns[1].selectbox(
    "Dataset",
    ["Tous", *dataset_options],
)
signature_options = sorted(
    {
        run["benchmark_signature"]
        for run in runs
        if run["benchmark_signature"]
    }
)
selected_signature = filter_columns[2].selectbox(
    "Configuration comparable",
    ["Toutes", *signature_options],
    help=(
        "Cette signature est identique lorsque le dataset et tous les "
        "hyperparamètres sont identiques, indépendamment du modèle BERT."
    ),
)

filtered = [
    run
    for run in runs
    if run["model"] in selected_models
    and (selected_dataset == "Tous" or run["dataset"] == selected_dataset)
    and (
        selected_signature == "Toutes"
        or run["benchmark_signature"] == selected_signature
    )
]

known_signatures = {
    run["benchmark_signature"]
    for run in filtered
    if run["benchmark_signature"]
}
missing_signatures = any(not run["benchmark_signature"] for run in filtered)
if len(known_signatures) > 1 or missing_signatures:
    st.warning(
        "Les runs affichés ne sont pas tous directement comparables. "
        "Sélectionnez une seule signature de configuration pour un benchmark "
        "strict. Les anciens runs peuvent ne pas disposer de signature."
    )
elif filtered:
    st.success("Les runs sélectionnés partagent une configuration comparable.")

st.subheader("Résultats comparés")
comparison_rows = [
    {
        "Modèle": run["model"],
        "Dataset": run["dataset"],
        "Profil": run["profile"],
        "Macro-F1": _percentage(run["macro_f1"]),
        "F1 pondéré": _percentage(run["f1_weighted"]),
        "Accuracy": _percentage(run["accuracy"]),
        "Précision": _percentage(run["precision"]),
        "Rappel": _percentage(run["recall"]),
        "Durée": _duration(run["duration_ms"]),
        "Actif": "Oui" if run["is_active"] else "Non",
        "Date": run["date"],
    }
    for run in filtered
]
comparison_dataframe = pd.DataFrame(comparison_rows)
st.dataframe(
    comparison_dataframe,
    use_container_width=True,
    hide_index=True,
)

chart_rows = [
    {
        "Run": f"{run['model']} · {run['date'][:10]}",
        "Macro-F1": run["macro_f1"],
        "F1 pondéré": run["f1_weighted"],
        "Accuracy": run["accuracy"],
    }
    for run in filtered
    if any(
        run[name] is not None
        for name in ("macro_f1", "f1_weighted", "accuracy")
    )
]
if chart_rows:
    st.bar_chart(pd.DataFrame(chart_rows).set_index("Run"))

st.subheader("Hyperparamètres")
parameter_rows = [
    {
        "Modèle": run["model"],
        "Profil": run["profile"],
        "Epochs": run["epochs"],
        "Learning rate": run["learning_rate"],
        "Batch": run["batch_size"],
        "Accumulation": run["gradient_accumulation"],
        "Batch effectif": run["effective_batch_size"],
        "Tokens": run["max_length"],
        "Seed": run["seed"],
        "Patience": run["early_stopping"],
        "Weight decay": run["weight_decay"],
        "Warmup": run["warmup_ratio"],
        "Sélection": run["selection_metric"],
        "Poids classes": run["class_weights"],
    }
    for run in filtered
]
st.dataframe(pd.DataFrame(parameter_rows), use_container_width=True, hide_index=True)

with st.expander("Traçabilité et reproductibilité", expanded=False):
    traceability_rows = [
        {
            "Run ID": run["run_id"],
            "Modèle": run["model"],
            "Checkpoint HF": run["huggingface_id"] or "—",
            "Version source": run["source_code_version"] or "—",
            "SHA-256 code": run["source_code_sha256"] or "Non disponible",
            "SHA-256 dataset": run["dataset_sha256"] or "Non disponible",
            "Signature benchmark": (
                run["benchmark_signature"] or "Non disponible"
            ),
            "Version modèle": run["model_version"],
        }
        for run in filtered
    ]
    st.dataframe(
        pd.DataFrame(traceability_rows),
        use_container_width=True,
        hide_index=True,
    )

if filtered:
    st.subheader("Détail d'un run")
    run_labels = {
        f"{run['model']} · {run['date']} · {run['run_id'][:8]}": run
        for run in filtered
    }
    selected_run = run_labels[
        st.selectbox("Run", list(run_labels))
    ]
    detail_columns = st.columns(4)
    detail_columns[0].metric("Macro-F1", _percentage(selected_run["macro_f1"]))
    detail_columns[1].metric("Accuracy", _percentage(selected_run["accuracy"]))
    detail_columns[2].metric("Durée", _duration(selected_run["duration_ms"]))
    detail_columns[3].metric(
        "Modèle actif",
        "Oui" if selected_run["is_active"] else "Non",
    )
    per_class = selected_run["metrics"].get("per_class")
    if isinstance(per_class, dict) and per_class:
        st.dataframe(
            pd.DataFrame(
                [
                    {"Classe": class_name, **_as_dict(values)}
                    for class_name, values in per_class.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
