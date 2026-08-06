from __future__ import annotations

import streamlit as st
from modules.model_base import render_base_model
from modules.model_fine_tuning import render_fine_tuning
from modules.model_history import render_model_history

st.set_page_config(
    page_title="Modèles",
    page_icon="🤖",
    layout="wide",
)

st.title("🧠 Centre des modèles IA")

st.caption(
    "Gestion des modèles et du Fine-Tuning de PRUDENCIA."
)

selected_tab = st.segmented_control(
    "Module",
    [
        "🧠 Modèle de base",
        "🎯 Fine-Tuning",
        "📜 Historique",
    ],
    key="selected_model_tab",
    default="🎯 Fine-Tuning",
)

if selected_tab == "🧠 Modèle de base":
    render_base_model()

elif selected_tab == "🎯 Fine-Tuning":
    render_fine_tuning()

elif selected_tab == "📜 Historique":
    render_model_history()
