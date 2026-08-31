from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="PRUDENCIA",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚖️ PRUDENCIA")
st.subheader("Pré-diagnostic de conformité des systèmes d'intelligence artificielle")

st.markdown(
    """
    **PRUDENCIA** aide à analyser un projet d'intelligence artificielle
    au regard de l'**AI Act européen**. L'application s'appuie sur l'analyse
    de texte par **Deep Learning** et sur une recherche documentaire **RAG**.
    """
)
st.info("Version de référence : PRUDENCIA v1.0.0 — MVP de certification")

st.divider()
st.header("📄 Parcours d'analyse documentaire")
st.markdown(
    """
    1. Import d'un document PDF décrivant le projet.
    2. Extraction du texte par l'API.
    3. Analyse du contenu avec JuriBERT.
    4. Recherche de références juridiques avec le RAG.
    5. Génération d'un rapport de pré-diagnostic.
    """
)
st.code(
    "PDF → FastAPI → JuriBERT → RAG / ChromaDB → Rapport",
    language="text",
)
st.caption("Utilisez le menu latéral pour accéder aux pages de l'application.")

st.divider()
st.header("Technologies utilisées")
technology_columns = st.columns(4)

with technology_columns[0]:
    st.subheader("Interface")
    st.markdown("**Streamlit**\n\nAffichage des analyses, résultats et rapports.")
with technology_columns[1]:
    st.subheader("API")
    st.markdown("**FastAPI**\n\nServices d'analyse et communication entre composants.")
with technology_columns[2]:
    st.subheader("Données")
    st.markdown(
        "**PostgreSQL** pour les projets et historiques.\n\n"
        "**ChromaDB** pour les références vectorielles."
    )
with technology_columns[3]:
    st.subheader("Déploiement")
    st.markdown("**Docker Compose**\n\nExécution coordonnée de tous les services.")

st.subheader("Modèles et recherche documentaire")
model_column_1, model_column_2 = st.columns(2)
with model_column_1:
    st.markdown(
        """
        #### 🧠 JuriBERT

        Modèle de langage spécialisé dans le domaine juridique,
        utilisé pour comprendre et classer les contenus documentaires.
        """
    )
with model_column_2:
    st.markdown(
        """
        #### 🔎 RAG

        Recherche de passages pertinents dans une base vectorielle
        pour enrichir l'analyse avec des sources documentaires.
        """
    )

st.divider()
st.header("Périmètre du MVP")
scope_column_1, scope_column_2 = st.columns(2)
with scope_column_1:
    st.markdown(
        """
        **Fonctionnalités intégrées**

        - analyse de documents PDF ;
        - traitement du texte avec JuriBERT ;
        - recherche RAG dans ChromaDB ;
        - historique des entraînements dans PostgreSQL ;
        - génération de rapports JSON.
        """
    )
with scope_column_2:
    st.markdown(
        """
        **Objectifs**

        - présenter un flux complet de données ;
        - exposer les traitements par API REST ;
        - fournir une interface simple ;
        - produire un pré-diagnostic exploitable.
        """
    )

st.divider()
st.warning(
    "PRUDENCIA fournit une aide à l'analyse. Les résultats ne remplacent "
    "pas la validation d'un professionnel compétent."
)
st.caption("PRUDENCIA — MVP v1.0.0")
