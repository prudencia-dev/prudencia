# 15 — Tableau de bord Benchmark NLP

## Objectif

Démontrer de manière simple, lisible et reproductible la comparaison de
CamemBERT, CamemBERTv2 et JuriBERT sans ajouter de plateforme MLOps externe.

## Tableau de bord

La nouvelle page `04_benchmark_nlp.py` exploite l'historique PostgreSQL pour
afficher :

- le nombre de runs et de modèles BERT testés ;
- le meilleur modèle selon le macro-F1 ;
- les métriques globales sous forme de tableau et de graphique ;
- les hyperparamètres de chaque run ;
- les informations de traçabilité ;
- les métriques détaillées d'un run ;
- un contrôle automatique de comparabilité.

## Hyperparamètres

L'écran de fine-tuning distingue désormais :

- l'optimisation ;
- les ressources et la reproductibilité ;
- la régularisation et la sélection du checkpoint.

Trois profils sont disponibles : validation rapide, démonstration équilibrée et
benchmark reproductible. Un récapitulatif est présenté avant l'entraînement.
Le document `GUIDE_PROFILS_HYPERPARAMETRES.md` détaille leurs usages, chaque
hyperparamètre et la méthode recommandée pour comparer les trois modèles.
La fiche `COMPRENDRE_PROFILS_ENTRAINEMENT.md` fournit une explication courte et
pédagogique destinée aux utilisateurs non spécialistes.

## Traçabilité

Chaque nouveau run enregistre dans `prudencia.model_executions` :

- l'identifiant du checkpoint Hugging Face ;
- le profil d'entraînement ;
- l'empreinte SHA-256 du dataset ;
- la taille du dataset ;
- la version déclarée de l'application ;
- l'empreinte SHA-256 du code Python exécuté ;
- une signature de benchmark indépendante du modèle comparé ;
- l'ensemble des hyperparamètres et métriques déjà disponibles.

## Choix d'architecture

PostgreSQL reste l'unique registre d'expériences. Aucun service MLflow, DVC,
Grafana ou autre composant d'exploitation n'est ajouté à ce stade. Le lot reste
ainsi adapté au périmètre du MVP.

## Validation

- contrôle Ruff ;
- suite Pytest complète, incluant les empreintes et signatures ;
- test automatisé du rendu de la page Streamlit ;
- contrôle de syntaxe et de format Git.
