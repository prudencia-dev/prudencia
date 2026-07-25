import streamlit as st

st.set_page_config(
    page_title="PRUDENCIA",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ PRUDENCIA")

st.markdown("""
## Plateforme d'audit IA

Bienvenue sur la console d'administration de PRUDENCIA.

Utilisez le menu de gauche pour accéder aux différents modules.

### Modules disponibles

- 📚 RAG
- 🤖 CamemBERT
- 🧠 Fine-Tuning
- 🌲 Machine Learning
- ⚙️ Administration
""")

st.info(
    "Cette console permet d'administrer le RAG, le Fine-Tuning et le Machine Learning."
)