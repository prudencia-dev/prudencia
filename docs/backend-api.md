# Backend et API

## Démarrage

`data/fastapi/app/main.py` crée l'application FastAPI, enregistre les routeurs et initialise la base au démarrage. Les schémas Pydantic valident les corps JSON. Les fichiers d'entraînement ou documents utilisent `multipart/form-data`.

La spécification exacte des paramètres et réponses est disponible dans OpenAPI sur `/docs`. Le tableau suivant sert de carte fonctionnelle.

## Routes principales

| Méthode | Chemin | Fonction |
|---|---|---|
| GET | `/` | Informations générales de l'API |
| GET | `/health` | Disponibilité du processus FastAPI |
| POST | `/analyse` | Enregistrement simple d'une analyse |
| GET | `/ai/health` | État du modèle CamemBERT |
| POST | `/ai/embedding` | Embedding CamemBERT d'un texte |
| GET | `/documents` | Liste des documents chargés |
| POST | `/documents/upload` | Téléversement d'un document |
| GET | `/documents/{filename}/extract` | Extraction du texte d'un PDF |
| POST | `/documents/{filename}/chunks` | Découpage d'un document |
| POST | `/documents/{filename}/index` | Indexation complète dans ChromaDB |

## RAG

| Méthode | Chemin | Fonction |
|---|---|---|
| GET | `/rag/health` | État de ChromaDB |
| GET | `/rag/collections` | Collections disponibles |
| GET | `/rag/collections/{name}/chunks` | Contenu d'une collection |
| POST | `/rag/chunks` | Ajout manuel d'un passage |
| POST | `/rag/search` | Recherche sémantique |
| GET | `/rag/stats` | Statistiques d'indexation |
| GET | `/rag/documents` | Documents connus du RAG |
| POST | `/rag/index` | Indexation d'un PDF |
| POST | `/rag/reset` | Réinitialisation de la collection |
| DELETE | `/rag/document/{document_id}` | Suppression des vecteurs d'un document |

Les premières routes, définies directement dans `main.py`, sont des primitives techniques. Les routes du routeur `api/rag.py` constituent l'interface métier utilisée par Streamlit.

## Modèles et entraînement

| Méthode | Chemin | Fonction |
|---|---|---|
| GET | `/ml/health` | État du modèle Random Forest |
| POST | `/ml/train` | Entraînement depuis un CSV |
| POST | `/ml/reset` | Désactivation du modèle courant |
| POST | `/ml/predict` | Prédiction tabulaire |
| GET | `/fine-tuning/models` | Modèles DL configurés |
| GET | `/fine-tuning/history` | Historique des entraînements |
| GET | `/fine-tuning/best-model` | Meilleur modèle entraîné |
| POST | `/fine-tuning/train` | Fine-tuning d'un modèle juridique |
| POST | `/fine-tuning/reset/{model_name}` | Réinitialisation d'un modèle |
| POST | `/fine-tuning/predict` | Classification d'un texte |
| GET | `/training/history` | Historique unifié ML/DL |

## Registre, questionnaire et rapports

| Méthode | Chemin | Fonction |
|---|---|---|
| GET | `/models/health` | Disponibilité du registre |
| GET | `/models` | Liste des modèles enregistrés |
| GET | `/models/runs` | Exécutions d'entraînement |
| GET | `/models/{model_id}` | Détail d'un modèle |
| POST | `/models/{model_id}/activate` | Activation exclusive par tâche |
| POST | `/models/{model_id}/archive` | Archivage d'un modèle |
| GET | `/questionnaire-mvp/active` | Questionnaire actif et questions |
| POST | `/questionnaire-mvp/submit` | Sauvegarde des réponses et création des variables ML |
| POST | `/document-analysis/extract` | Extraction d'un PDF destiné à l'analyse |
| GET | `/reports/health` | Disponibilité du générateur |
| POST | `/reports/generate` | Rapport fondé sur texte, DL et RAG |
| POST | `/reports/generate-ml` | Rapport fondé sur le questionnaire et le ML |

## Services applicatifs

- `AnalysisOrchestrator` coordonne les différentes briques d'analyse.
- `DocumentService`, `PdfService` et `ChunkingService` gèrent le cycle documentaire.
- `RagService` orchestre extraction, découpage, embeddings et métadonnées.
- `ModelRegistryService` gère le catalogue et l'activation des modèles.
- `ModelSelectionService` choisit le modèle exploitable.
- `TrainingHistoryService` normalise l'historique ML et DL.
- `PrudenciaReportBuilder` construit l'objet métier final.
- `IndexingProgressService` maintient l'état d'une indexation.

## Gestion des erreurs

Les routes convertissent les erreurs attendues en `HTTPException` avec un code 4xx ou 5xx. Les services lèvent des exceptions Python lorsqu'un fichier, une collection ou un modèle est invalide. Les clients doivent toujours vérifier le statut HTTP et ne pas supposer qu'une réponse contient le résultat demandé.
