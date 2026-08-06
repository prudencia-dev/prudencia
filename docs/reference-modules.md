# Référence des modules

Cette page rattache chaque module source à sa responsabilité principale. Les fichiers `__init__.py` marquent les packages Python et ne portent pas de logique métier.

## API FastAPI

| Module | Responsabilité |
|---|---|
| `app/main.py` | Création de l'application, routeurs et endpoints techniques historiques |
| `app/config.py` | Chemins, catalogue des modèles et paramètres d'entraînement par défaut |
| `app/database.py` | Connexion PostgreSQL, initialisation et sauvegarde d'analyses |
| `app/api/document_analysis.py` | Extraction d'un PDF pour le parcours d'analyse documentaire |
| `app/api/fine_tuning.py` | Catalogue, entraînement, reset et prédiction Deep Learning |
| `app/api/ml.py` | Santé, entraînement, reset et prédiction Machine Learning |
| `app/api/model_registry.py` | Consultation, activation et archivage des modèles enregistrés |
| `app/api/questionnaire_ml.py` | Chargement du questionnaire, réponses et variables destinées au ML |
| `app/api/rag.py` | Statistiques, indexation et administration du RAG |
| `app/api/reports.py` | Génération des rapports documentaires et questionnaires |
| `app/api/training_history.py` | Historique unifié des entraînements |

## Services métier

| Module | Responsabilité |
|---|---|
| `services/analysis_orchestrator.py` | Coordination générale d'une analyse |
| `services/chunking_service.py` | Découpage des textes et persistance des passages |
| `services/document_service.py` | Sauvegarde, empreinte et inventaire des documents |
| `services/indexing_progress_service.py` | Suivi de l'avancement d'une indexation |
| `services/model_registry_service.py` | Accès métier au registre PostgreSQL des modèles |
| `services/model_selection_service.py` | Sélection d'un modèle entraîné exploitable |
| `services/pdf_service.py` | Résolution sécurisée et extraction de PDF |
| `services/rag_service.py` | Pipeline documentaire complet vers ChromaDB |
| `services/report_builder.py` | Assemblage du rapport de conformité |
| `services/training_history_service.py` | Écriture et lecture de l'historique ML/DL |

## Intelligence artificielle

| Module | Responsabilité |
|---|---|
| `ai/bge_m3.py` | Chargement de BGE-M3 et création d'embeddings RAG |
| `ai/camembert.py` | Chargement de CamemBERT et embedding généraliste |
| `ai/rag.py` | Client ChromaDB et opérations vectorielles bas niveau |
| `ai/machine_learning/trainer.py` | Pipeline Random Forest, métriques et artefact |
| `ai/fine_tuning/dataset.py` | Validation et préparation du dataset textuel |
| `ai/fine_tuning/trainer.py` | Fine-tuning Transformers et évaluation |
| `ai/fine_tuning/benchmark.py` | Calcul et comparaison des performances |
| `ai/fine_tuning/experiment.py` | Traçabilité et persistance des expériences |

## Domaine et accès aux données

| Module | Responsabilité |
|---|---|
| `domain/report.py` | Structures Project, classification, risque, recommandation et rapport |
| `repositories/base_repository.py` | Primitive commune de connexion et requêtes PostgreSQL |
| `config/init/04_model_registry.sql` | Schéma du registre et historique d'entraînement |

## Interface Streamlit

| Module | Responsabilité |
|---|---|
| `data/streamlit/app.py` | Accueil de l'application |
| `services/api.py` | Client HTTP commun vers FastAPI |
| `pages/01_rag.py` | Administration documentaire du RAG |
| `pages/02_modeles.py` | Conteneur des vues modèles |
| `pages/03_rapport_documentaire.py` | Parcours PDF vers rapport documentaire |
| `pages/04_rapport_questionnaire.py` | Parcours questionnaire vers rapport ML |
| `modules/model_base.py` | Présentation et essai du modèle JuriBERT de base |
| `modules/model_fine_tuning.py` | Formulaire et résultats du fine-tuning |
| `modules/model_machine_learning.py` | Formulaire et résultats du Random Forest |
| `modules/model_comparison.py` | Comparaison des exécutions ML et DL |
| `modules/model_history.py` | Vue détaillée de l'historique et des métriques |

## Démonstrations et infrastructure

| Fichier | Responsabilité |
|---|---|
| `notebooks/ML/01_RandomForest_Training.ipynb` | Démonstration interactive du pipeline ML |
| `notebooks/ML/01_random_forest_training.py` | Version script de la démonstration ML |
| `notebooks/DL/02_JuriBERT_FineTuning.ipynb` | Démonstration interactive du fine-tuning |
| `notebooks/DL/02_juribert_finetuning.py` | Version script de la démonstration DL |
| `notebooks/datasets/ml/ml_training_dataset_v1.csv` | Données tabulaires pédagogiques |
| `notebooks/datasets/dl/dl_juribert_training_cases_v2.csv` | Corpus textuel annoté pédagogique |
| `compose/dev/compose.yaml` | Services, réseaux implicites, ports et volumes locaux |
| `data/fastapi/Dockerfile` | Image d'exécution de l'API |
| `data/fastapi/requirements.txt` | Dépendances du backend et de l'IA |
| `data/streamlit/requirements.txt` | Dépendances de l'interface |
| `notebooks/requirements.txt` | Dépendances des notebooks |

## Points d'extension

- Une nouvelle route métier commence dans `api/` et délègue à `services/`.
- Un nouveau modèle est encapsulé dans `ai/` et enregistré dans `config.py`.
- Une nouvelle persistance spécialisée étend la couche `repositories/`.
- Une nouvelle vue utilisateur est ajoutée à `pages/` ou comme module réutilisable.
