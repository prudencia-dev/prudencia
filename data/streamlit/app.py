from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="PRUDENCIA",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------

st.title("⚖️ PRUDENCIA")
st.subheader(
    "Pré-diagnostic de conformité des systèmes d’intelligence artificielle"
)

st.markdown(
    """
    **PRUDENCIA** est une application d’aide à l’analyse de conformité
    d’un projet d’intelligence artificielle au regard de l’**AI Act européen**.

    Elle combine deux approches complémentaires :

    - une analyse documentaire fondée sur le **Deep Learning** et le **RAG** ;
    - une analyse structurée fondée sur le **Machine Learning**.
    """
)

st.info(
    "Version de référence : PRUDENCIA v1.0.0 — MVP de certification"
)


# ---------------------------------------------------------------------------
# Parcours principaux
# ---------------------------------------------------------------------------

st.divider()
st.header("Deux parcours d’analyse")

documentary_column, questionnaire_column = st.columns(2)

with documentary_column:
    st.subheader("📄 Analyse documentaire")
    st.markdown(
        """
        Ce parcours permet d’analyser un document PDF décrivant un projet
        ou un système d’intelligence artificielle.

        **Étapes principales :**

        1. import du document PDF ;
        2. extraction du texte par l’API ;
        3. analyse NLP avec **JuriBERT** ;
        4. recherche de références avec le **RAG** ;
        5. génération d’un rapport de pré-diagnostic.
        """
    )
    st.caption(
        "Brique principale : Deep Learning, NLP et recherche vectorielle."
    )

with questionnaire_column:
    st.subheader("📋 Analyse par questionnaire")
    st.markdown(
        """
        Ce parcours recueille des informations structurées sur le projet
        d’intelligence artificielle.

        **Étapes principales :**

        1. saisie du questionnaire métier ;
        2. enregistrement des réponses dans PostgreSQL ;
        3. construction des variables du modèle ;
        4. prédiction avec un **Random Forest** ;
        5. génération d’un rapport de pré-diagnostic.
        """
    )
    st.caption(
        "Brique principale : Machine Learning supervisé."
    )

st.caption(
    "Utilisez le menu latéral pour accéder aux différentes pages de l’application."
)


# ---------------------------------------------------------------------------
# Architecture fonctionnelle
# ---------------------------------------------------------------------------

st.divider()
st.header("Fonctionnement général")

st.code(
    """
ANALYSE DOCUMENTAIRE
PDF → FastAPI → JuriBERT → RAG / ChromaDB → Rapport

ANALYSE PAR QUESTIONNAIRE
Questionnaire → FastAPI → PostgreSQL → Random Forest → Rapport
""".strip(),
    language="text",
)


# ---------------------------------------------------------------------------
# Technologies
# ---------------------------------------------------------------------------

st.divider()
st.header("Technologies utilisées")

technology_columns = st.columns(4)

with technology_columns[0]:
    st.subheader("Interface")
    st.markdown(
        """
        **Streamlit**

        Affichage des formulaires, des résultats et des rapports.
        """
    )

with technology_columns[1]:
    st.subheader("API")
    st.markdown(
        """
        **FastAPI**

        Exposition des services d’analyse et communication entre les composants.
        """
    )

with technology_columns[2]:
    st.subheader("Données")
    st.markdown(
        """
        **PostgreSQL**

        Stockage structuré des projets, questionnaires, réponses et historiques.

        **ChromaDB**

        Stockage vectoriel utilisé par le RAG.
        """
    )

with technology_columns[3]:
    st.subheader("Déploiement")
    st.markdown(
        """
        **Docker Compose**

        Exécution isolée et coordonnée de l’API, de Streamlit,
        de PostgreSQL et de ChromaDB.
        """
    )


st.subheader("Modèles d’intelligence artificielle")

model_column_1, model_column_2, model_column_3 = st.columns(3)

with model_column_1:
    st.markdown(
        """
        #### 🌲 Random Forest

        Modèle de Machine Learning supervisé utilisé pour classer
        le niveau de risque d’un projet à partir de variables structurées.
        """
    )

with model_column_2:
    st.markdown(
        """
        #### 🧠 JuriBERT

        Modèle de langage spécialisé dans le domaine juridique,
        utilisé pour l’analyse des contenus documentaires.
        """
    )

with model_column_3:
    st.markdown(
        """
        #### 🔎 RAG

        Recherche de passages pertinents dans une base vectorielle
        afin d’enrichir l’analyse avec des références documentaires.
        """
    )


# ---------------------------------------------------------------------------
# Périmètre du MVP
# ---------------------------------------------------------------------------

st.divider()
st.header("Périmètre du MVP v1.0.0")

scope_column_1, scope_column_2 = st.columns(2)

with scope_column_1:
    st.markdown(
        """
        **Fonctionnalités intégrées**

        - analyse de documents PDF ;
        - extraction et traitement du texte ;
        - analyse NLP avec JuriBERT ;
        - recherche RAG dans ChromaDB ;
        - questionnaire métier ;
        - prédiction Random Forest ;
        - stockage PostgreSQL ;
        - génération de rapports JSON.
        """
    )

with scope_column_2:
    st.markdown(
        """
        **Objectifs du MVP**

        - démontrer deux briques IA distinctes ;
        - présenter un flux complet de données ;
        - exposer les traitements par API REST ;
        - fournir une interface simple aux consultants ;
        - produire un pré-diagnostic exploitable ;
        - constituer la version de référence pour la certification.
        """
    )


# ---------------------------------------------------------------------------
# Répartition des rôles
# ---------------------------------------------------------------------------

st.divider()
st.header("Organisation du projet")

st.markdown(
    """
    - **Jean-Philippe** — conception et développement de la solution IA ;
    - **Sébastien** — expertise métier, scénarios et validation fonctionnelle ;
    - **Cora** — expertise métier, jeux de données et validation fonctionnelle.
    """
)


# ---------------------------------------------------------------------------
# Avertissement
# ---------------------------------------------------------------------------

st.divider()
st.warning(
    """
    PRUDENCIA fournit un pré-diagnostic d’aide à l’analyse.
    Les résultats ne remplacent pas une validation juridique ou réglementaire
    réalisée par un professionnel compétent.
    """
)

st.caption("PRUDENCIA — MVP v1.0.0")