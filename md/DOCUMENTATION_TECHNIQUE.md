# Documentation technique détaillée — PRUDENCIA v1.2.0

## 1. Objet du système

PRUDENCIA est une application locale d'aide à l'analyse documentaire de conformité des systèmes d'intelligence artificielle. Elle permet de charger des documents, d'en extraire le texte, de les indexer dans un moteur RAG, d'exécuter des traitements Deep Learning et de construire un rapport structuré. Les résultats produits assistent l'analyse humaine et ne constituent pas un avis juridique automatisé.

Le Machine Learning classique et les notebooks pédagogiques ne font plus partie du dépôt applicatif. Les expérimentations indépendantes sont conservées dans le dépôt privé `prudencia-dev/poc`.

## 2. Vue d'ensemble

Le système est composé de quatre conteneurs Docker :

| Composant | Technologie | Port hôte | Responsabilité |
|---|---|---:|---|
| Interface | Streamlit | `8501` | Parcours utilisateur et restitution |
| API | FastAPI / Uvicorn | `${API_PORT}` ou `8000` | Contrats HTTP, validation et orchestration |
| Base relationnelle | PostgreSQL 16 | `5432` | Documents, analyses, chunks et historique des modèles |
| Base vectorielle | ChromaDB | `8001` | Embeddings, métadonnées et recherche sémantique |

Les modèles Hugging Face et les fichiers téléversés sont stockés dans des volumes montés depuis l'hôte. L'API utilise une distribution CPU de PyTorch afin de ne pas embarquer les bibliothèques CUDA inutiles sur le poste local.

Voir également :

- [Schéma de l'architecture technique](SCHEMA_ARCHITECTURE_TECHNIQUE.md) ;
- [Schéma de l'architecture du code source](SCHEMA_ARCHITECTURE_CODE_SOURCE.md).

## 3. Organisation du dépôt

```text
Prudencia/
├── .github/workflows/       CI GitHub Actions
├── compose/dev/             orchestration Docker locale
├── config/init/             initialisation SQL PostgreSQL
├── data/
│   ├── fastapi/             backend, moteurs IA et tests
│   ├── streamlit/           interface utilisateur
│   ├── postgres/            données PostgreSQL locales ignorées
│   ├── chromadb/            données vectorielles locales ignorées
│   ├── uploads/             documents téléversés ignorés
│   └── models/              cache Hugging Face ignoré
├── md/                      documentation et rapports chronologiques
└── models/                  artefacts de modèles montés dans l'API
```

## 4. Architecture du backend

### 4.1 Point d'entrée FastAPI

`data/fastapi/app/main.py` :

- initialise l'application FastAPI en version `1.2.0` ;
- configure les journaux applicatifs ;
- ajoute le middleware de corrélation des requêtes ;
- enregistre les routeurs spécialisés ;
- initialise le schéma applicatif au démarrage ;
- expose quelques routes historiques pour les documents, CamemBERT et le RAG.

Les routeurs spécialisés se trouvent dans `app/api/`. Ils traduisent les entrées HTTP en appels de services et transforment les erreurs attendues en statuts HTTP explicites.

### 4.2 Couche API

| Module | Préfixe | Responsabilité |
|---|---|---|
| `api/document_analysis.py` | `/document-analysis` | Extraction sécurisée d'un PDF destiné à l'analyse |
| `api/fine_tuning.py` | `/fine-tuning` | Modèles disponibles, entraînement, prédiction et réinitialisation |
| `api/model_registry.py` | `/models` | Catalogue, activation et archivage des modèles |
| `api/rag.py` | `/rag` | Indexation, statistiques, reset et suppression documentaire |
| `api/reports.py` | `/reports` | Génération du rapport PRUDENCIA |
| `api/training_history.py` | `/training` | Historique unifié des exécutions DL |
| `api/error_responses.py` | — | Journalisation interne et réponses publiques sans fuite technique |

### 4.3 Couche services

La couche `app/services/` contient la logique métier indépendante du transport HTTP.

| Service | Rôle |
|---|---|
| `AnalysisOrchestrator` | Assemble les résultats DL et RAG avant construction du rapport |
| `PrudenciaReportBuilder` | Normalise les risques, recommandations et références juridiques |
| `DocumentService` | Sauvegarde les uploads et persiste leurs métadonnées |
| `PdfService` | Résout un chemin confiné et extrait le texte avec PyMuPDF |
| `ChunkingService` | Découpe le texte et persiste les chunks |
| `RagService` | Orchestre upload, extraction, chunks, embeddings et indexation |
| `UploadSecurity` | Valide noms, extensions, tailles, chemins et signature PDF |
| `ModelRegistryService` | Accède au registre des modèles entraînés |
| `ModelSelectionService` | Sélectionne un modèle actif et exploitable |
| `TrainingHistoryService` | Persiste et restitue les exécutions d'entraînement |
| `IndexingProgressService` | Maintient l'état temporaire d'une indexation |

### 4.4 Couche domaine

`app/domain/report.py` définit les objets métier du rapport :

- `Project` ;
- `AIActClassification` ;
- `Risk` ;
- `Recommendation` ;
- `LegalReference` ;
- `PrudenciaReport`.

Ces dataclasses ne lancent aucun traitement. Elles structurent le résultat final et le convertissent en dictionnaire JSON avec `to_dict()`.

### 4.5 Accès aux données

`app/database.py` construit les connexions PostgreSQL à partir de l'environnement. `BaseRepository` fournit un contexte transactionnel commun. Certains services historiques exécutent encore directement leurs requêtes SQL ; une évolution future pourrait uniformiser ces accès derrière des repositories spécialisés.

## 5. Moteurs d'intelligence artificielle

### 5.1 BGE-M3 et RAG

`app/ai/bge_m3.py` charge paresseusement le modèle `BAAI/bge-m3` via Sentence Transformers. Deux fonctions sont exposées :

- `get_embedding(text)` pour une requête ou un passage unique ;
- `get_embeddings(texts)` pour un lot de chunks.

Les vecteurs sont normalisés. `embedding_config.py` centralise le nom du modèle et évite qu'une collection ChromaDB mélange des embeddings incompatibles.

`app/ai/rag.py` encapsule les opérations ChromaDB : santé, collections, ajout, recherche, statistiques, reset et suppression ciblée.

### 5.2 CamemBERT

`app/ai/camembert.py` expose l'ancien service d'embedding français utilisé par les routes `/ai/*`. Le chargement du tokenizer et du modèle est paresseux afin de ne pas charger PyTorch au démarrage lorsque la route n'est pas utilisée.

### 5.3 Fine-tuning Deep Learning

Le package `app/ai/fine_tuning/` regroupe :

- `dataset.py` : validation et préparation du CSV ;
- `trainer.py` : tokenisation, pondération des classes, entraînement et prédiction ;
- `experiment.py` : suivi des expériences ;
- `benchmark.py` : comparaison des métriques.

L'entraînement reste synchrone dans le MVP. Pour une charge de production, il devrait être déporté vers un worker et une file de tâches.

## 6. Flux fonctionnels principaux

### 6.1 Indexation documentaire

1. Streamlit envoie un PDF à `POST /rag/index`.
2. `UploadSecurity` contrôle le nom, l'extension, la taille et la signature `%PDF-`.
3. `DocumentService` génère un nom UUID, écrit le fichier par blocs et calcule son SHA-256.
4. PostgreSQL reçoit les métadonnées du document.
5. `PdfService` extrait le texte et le nombre de pages.
6. `ChunkingService` découpe et persiste les passages.
7. BGE-M3 calcule les embeddings par lot.
8. ChromaDB reçoit les textes, vecteurs et métadonnées.
9. PostgreSQL conserve les identifiants ChromaDB et la dimension des embeddings.
10. L'API retourne le bilan d'indexation à Streamlit.

### 6.2 Recherche RAG

1. Le client envoie une requête à `POST /rag/search`.
2. BGE-M3 calcule l'embedding normalisé de la requête.
3. ChromaDB cherche les passages les plus proches dans la collection configurée.
4. L'API renvoie textes, distances et métadonnées.

### 6.3 Rapport documentaire

1. Streamlit transmet le PDF pour extraction.
2. Les résultats DL et RAG sont normalisés par la couche API.
3. `AnalysisOrchestrator` transmet les données à `PrudenciaReportBuilder`.
4. Le builder crée classification AI Act, risques, recommandations et références.
5. Le rapport structuré est renvoyé à l'interface.

### 6.4 Fine-tuning

1. L'utilisateur charge un CSV et sélectionne les colonnes texte/label.
2. L'API limite la taille du fichier et valide les paramètres.
3. `DatasetManager` contrôle les classes et prépare les jeux d'entraînement.
4. `FineTuningTrainer` lance Transformers sur CPU dans l'image locale.
5. Les métriques et l'exécution sont enregistrées dans PostgreSQL.
6. L'interface affiche le rapport d'entraînement et l'historique.

## 7. Interface Streamlit

`data/streamlit/app.py` configure la navigation et affiche la version `v1.2.0`.

| Vue | Fichier | Fonction |
|---|---|---|
| Administration RAG | `pages/01_rag.py` | Indexation, statistiques et gestion documentaire |
| Modèles | `pages/02_modeles.py` | Accès au modèle de base, fine-tuning et historique |
| Rapport documentaire | `pages/03_rapport_documentaire.py` | Extraction, analyse et restitution du rapport |
| Modèle de base | `modules/model_base.py` | État et essai d'une prédiction |
| Fine-tuning | `modules/model_fine_tuning.py` | Paramétrage, entraînement et métriques |
| Historique | `modules/model_history.py` | Consultation des exécutions |
| Client API | `services/api.py` | Appels HTTP centralisés vers FastAPI |

Dans Docker, `API_URL=http://api:8000`. Les erreurs HTTP sont converties en messages utilisateur sans exposer les détails internes.

## 8. Persistance

### 8.1 PostgreSQL

Les scripts de `config/init/` créent le schéma `prudencia` et six tables principales :

| Table | Contenu |
|---|---|
| `analyses` | Analyses textuelles simples |
| `documents` | Métadonnées, checksum et état d'extraction |
| `document_chunks` | Passages, comptages et identifiants ChromaDB |
| `model_executions` | Historique normalisé des traitements |
| `trained_models` | Registre des modèles et statut actif/archivé |
| `model_training_runs` | Détail des entraînements et métriques |

Les scripts sont idempotents grâce à `IF NOT EXISTS`. Ils s'exécutent automatiquement uniquement lors de l'initialisation d'un nouveau volume PostgreSQL.

### 8.2 ChromaDB

La collection par défaut est `prudencia_legal_documents`. Chaque élément stocke :

- le texte du chunk ;
- son embedding BGE-M3 ;
- l'identifiant PostgreSQL du document ;
- le nom du fichier ;
- l'index du chunk ;
- le modèle et la dimension d'embedding ;
- les paramètres de découpage.

### 8.3 Système de fichiers

| Chemin conteneur | Montage local | Contenu |
|---|---|---|
| `/uploads` | `data/uploads` | PDF téléversés |
| `/models/huggingface` | `data/models/huggingface` | Cache des modèles téléchargés |
| `/models` | `models` | Artefacts entraînés |
| `/var/lib/postgresql/data` | `data/postgres` | Volume PostgreSQL |
| `/data` dans ChromaDB | `data/chromadb` | Index vectoriel |

## 9. Sécurité applicative

### Uploads

- noms clients réduits au nom de fichier ;
- extension imposée selon la route ;
- nom interne UUID non prédictible ;
- chemin résolu et confiné au répertoire autorisé ;
- limites de 20 Mo pour les PDF et 10 Mo pour les CSV par défaut ;
- lecture/copie bornée ;
- signature PDF vérifiée ;
- fichier partiel supprimé en cas d'échec.

### Erreurs

`raise_api_error()` journalise l'exception complète côté serveur et renvoie un message public générique. Les chemins, mots de passe ou traces internes ne doivent jamais apparaître dans les réponses 500.

### Secrets

Le fichier `compose/dev/.env` est ignoré par Git. Seul `.env.example` est versionné avec des valeurs factices. Les secrets de production doivent être gérés hors du dépôt.

### Limites du MVP

L'application ne fournit pas encore d'authentification, d'autorisation, de TLS ni de limitation de débit. Elle doit rester sur un environnement local ou derrière une infrastructure sécurisée.

## 10. Observabilité

Chaque requête reçoit un UUID dans `X-Request-ID`. Un UUID client valide est conservé ; une valeur absente ou invalide est remplacée pour prévenir l'injection dans les logs.

Les logs `prudencia.*` sont émis sur la sortie standard :

- `LOG_FORMAT=json` par défaut pour Docker ;
- `LOG_FORMAT=text` pour la lecture locale ;
- `LOG_LEVEL=INFO` par défaut.

Les événements HTTP contiennent l'horodatage UTC, le niveau, le logger, le message, le `request_id`, la méthode, le chemin, le statut et la durée en millisecondes.

## 11. Configuration

Variables principales :

| Variable | Usage | Défaut ou valeur Compose |
|---|---|---|
| `POSTGRES_DB` | Base PostgreSQL | `prudencia` |
| `POSTGRES_USER` | Rôle PostgreSQL | `prudencia` |
| `POSTGRES_PASSWORD` | Mot de passe local | obligatoire dans `.env` |
| `POSTGRES_HOST` | Hôte PostgreSQL | `postgres` |
| `POSTGRES_PORT` | Port PostgreSQL | `5432` |
| `CHROMA_HOST` | Hôte ChromaDB | `chromadb` |
| `CHROMA_PORT` | Port interne | `8000` |
| `CHROMA_COLLECTION` | Collection RAG | `prudencia_legal_documents` |
| `UPLOAD_DIR` | Stockage des PDF | `/uploads` |
| `MAX_PDF_UPLOAD_BYTES` | Taille PDF maximale | `20971520` |
| `MAX_CSV_UPLOAD_BYTES` | Taille CSV maximale | `10485760` |
| `HF_HOME` | Cache Hugging Face | `/models/huggingface` |
| `API_URL` | API vue par Streamlit | `http://api:8000` |
| `LOG_FORMAT` | Format des logs | `json` |
| `LOG_LEVEL` | Niveau de logs | `INFO` |

## 12. Démarrage et exploitation locale

```bash
cd compose/dev
cp .env.example .env
docker compose up --build
```

Accès :

- Streamlit : `http://localhost:8501` ;
- API : `http://localhost:8000` par défaut ;
- Swagger : `http://localhost:8000/docs` ;
- ReDoc : `http://localhost:8000/redoc` ;
- ChromaDB : `http://localhost:8001`.

Commandes de diagnostic :

```bash
docker compose ps
docker compose logs -f api
curl http://localhost:8000/health
curl http://localhost:8000/rag/health
```

## 13. Tests et intégration continue

Les tests du backend se trouvent dans `data/fastapi/tests/`. Ils couvrent notamment :

- suppression effective du ML ;
- validation des uploads ;
- cohérence des embeddings RAG ;
- masquage des erreurs internes ;
- identifiants de requête ;
- journalisation structurée ;
- composants de benchmark.

La CI GitHub Actions exécute trois jobs :

1. qualité Python, compilation et tests ;
2. validation de la configuration Docker Compose ;
3. création d'une base PostgreSQL propre et vérification des six tables.

Commandes locales :

```bash
PYTHONPATH=data/fastapi pytest data/fastapi/tests
ruff check data/fastapi/app data/fastapi/tests
python -m compileall -q data/fastapi/app data/streamlit
docker compose -f compose/dev/compose.yaml config
```

## 14. Points d'attention pour la maintenance

- ne pas mélanger plusieurs modèles d'embedding dans une collection existante ;
- sauvegarder PostgreSQL et ChromaDB ensemble pour conserver les correspondances ;
- ne jamais versionner `.env`, les documents, caches ou modèles volumineux ;
- vérifier les migrations avant de modifier les scripts d'initialisation ;
- conserver les routes HTTP minces et placer la logique dans les services ;
- préserver le masquage des erreurs internes ;
- exécuter la CI et une recette Docker avant chaque tag ;
- faire tourner les entraînements longs dans un worker avant un usage multi-utilisateur.

## 15. Documentation associée

- [Architecture et flux](architecture.md)
- [Backend et API](backend-api.md)
- [Interface Streamlit](interface-streamlit.md)
- [Intelligence artificielle](intelligence-artificielle.md)
- [Données et persistance](donnees.md)
- [Configuration et déploiement](deploiement.md)
- [Développement et tests](developpement.md)
- [Référence des modules](reference-modules.md)
- [Schéma de l'architecture technique](SCHEMA_ARCHITECTURE_TECHNIQUE.md)
- [Schéma de l'architecture du code source](SCHEMA_ARCHITECTURE_CODE_SOURCE.md)
