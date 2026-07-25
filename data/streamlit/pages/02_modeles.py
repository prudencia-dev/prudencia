from __future__ import annotations

import streamlit as st

from modules.model_base import render_base_model
from modules.model_fine_tuning import render_fine_tuning
from modules.model_machine_learning import render_machine_learning
from modules.model_comparison import render_model_comparison
from modules.model_history import render_model_history


st.set_page_config(
    page_title="Modèles",
    page_icon="🤖",
    layout="wide",
)

st.title("🧠 Centre des modèles IA")

st.caption(
    "Gestion des modèles, Fine-Tuning et Machine Learning de PRUDENCIA."
)

selected_tab = st.segmented_control(
    "Module",
    [
        "🧠 Modèle de base",
        "🎯 Fine-Tuning",
        "🤖 Machine Learning",
        "📜 Historique",
        "🏆 Comparaison",
    ],
    key="selected_model_tab",
    default="🎯 Fine-Tuning",
)

if selected_tab == "🧠 Modèle de base":
    render_base_model()

elif selected_tab == "🎯 Fine-Tuning":
    render_fine_tuning()

elif selected_tab == "🤖 Machine Learning":
    render_machine_learning()

elif selected_tab == "📜 Historique":
    render_model_history()

elif selected_tab == "🏆 Comparaison":
    render_model_comparison()