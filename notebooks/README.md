# PRUDENCIA — Notebook Deep Learning

Ce dossier contient la démonstration pédagogique du fine-tuning de JuriBERT réalisée pour la certification Développeur en Intelligence Artificielle.

Le notebook est indépendant de l'application FastAPI et Streamlit. Il explique les étapes du pipeline Deep Learning :

1. chargement du corpus annoté ;
2. exploration et contrôle des classes ;
3. préparation et séparation des données ;
4. tokenisation avec Transformers ;
5. fine-tuning de JuriBERT ;
6. évaluation globale et par classe ;
7. matrice de confusion ;
8. sauvegarde du modèle.

## Arborescence

```text
notebooks/
├── README.md
├── requirements.txt
├── datasets/dl/
│   └── dl_juribert_training_cases_v2.csv
└── DL/
    ├── 02_JuriBERT_FineTuning.ipynb
    └── 02_juribert_finetuning.py
```

## Installation

```bash
python -m venv .venv-notebooks
source .venv-notebooks/bin/activate
pip install -r notebooks/requirements.txt
```

Le script Python peut être utilisé pour reproduire l'expérience sans l'interface Jupyter. Un GPU est recommandé pour réduire le temps d'entraînement.

## Dataset

Le corpus textuel contient des cas juridiques annotés destinés à la démonstration. Il ne doit pas être confondu avec des données de production et ne remplace pas un corpus validé par des experts métier.

## Périmètre

Le volet pédagogique porte exclusivement sur le Deep Learning juridique.
