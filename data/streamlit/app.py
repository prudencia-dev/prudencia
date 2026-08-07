from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="PRUDENCIA",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚖️ PRUDENCIA")
st.subheader(
    "Pré-diagnostic documentaire de conformité des systèmes "
    "d'intelligence artificielle"
)

st.markdown(
    """
    **PRUDENCIA** aide à analyser la conformité d'un projet d'intelligence
    artificielle au regard de l'**AI Act européen**. L'application associe
    l'analyse juridique de **JuriBERT** à une recherche documentaire **RAG**.
    """
)

st.info("Version de référence : PRUDENCIA v1.2.0 — MVP stabilisé")

st.divider()
st.header("Parcours d'analyse documentaire")

description, flow = st.columns(2)

with description:
    st.markdown(
        """
        Chargez un PDF décrivant un projet ou un système d'intelligence
        artificielle. PRUDENCIA extrait son contenu, le classe avec JuriBERT,
        recherche les références pertinentes et produit un pré-diagnostic.
        """
    )

with flow:
    st.markdown(
        """
        1. import et extraction du PDF ;
        2. analyse NLP avec JuriBERT ;
        3. recherche sémantique avec BGE-M3 et ChromaDB ;
        4. génération du rapport documentaire.
        """
    )

st.code(
    "PDF → FastAPI → JuriBERT → BGE-M3 / ChromaDB → Rapport",
    language="text",
)
st.caption("Utilisez le menu latéral pour accéder aux pages de l'application.")

st.divider()
st.header("Technologies utilisées")

technology_columns = st.columns(4)
technology_columns[0].subheader("Interface")
technology_columns[0].markdown("**Streamlit** — parcours et restitution.")
technology_columns[1].subheader("API")
technology_columns[1].markdown("**FastAPI** — services et orchestration.")
technology_columns[2].subheader("Données")
technology_columns[2].markdown(
    "**PostgreSQL** — historique.\n\n**ChromaDB** — vecteurs du RAG."
)
technology_columns[3].subheader("Déploiement")
technology_columns[3].markdown("**Docker Compose** — environnement local.")

st.subheader("Modèles d'intelligence artificielle")
model_columns = st.columns(2)

with model_columns[0]:
    st.markdown(
        """
        #### 🧠 JuriBERT

        Modèle de langage spécialisé dans le domaine juridique, utilisé pour
        classifier le contenu documentaire.
        """
    )

with model_columns[1]:
    st.markdown(
        """
        #### 🔎 BGE-M3 et RAG

        Embeddings et recherche de passages pertinents pour enrichir l'analyse
        avec des références documentaires.
        """
    )

st.divider()
st.header("Périmètre du MVP")
st.markdown(
    """
    - import et extraction de documents PDF ;
    - analyse NLP et fine-tuning de JuriBERT ;
    - indexation et recherche RAG dans ChromaDB ;
    - registre et historique des modèles ;
    - génération de rapports documentaires JSON.
    """
)

st.warning(
    "PRUDENCIA fournit un pré-diagnostic automatisé. Les résultats doivent "
    "être vérifiés par un professionnel compétent et ne constituent pas un "
    "avis juridique."
)
