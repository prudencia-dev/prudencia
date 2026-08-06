# Configuration et déploiement

## Services Docker

`compose/dev/compose.yaml` démarre :

- PostgreSQL 16 sur le port hôte 5432 ;
- ChromaDB sur le port hôte 8001 ;
- FastAPI, construit depuis `data/fastapi/Dockerfile` ;
- Streamlit sur le port hôte 8501.

Le mode actuel est destiné au développement : montage du code source et rechargement automatique d'Uvicorn.

## Variables d'environnement

Créer `compose/dev/.env` sans le versionner :

```dotenv
POSTGRES_DB=prudencia
POSTGRES_USER=prudencia
POSTGRES_PASSWORD=change-me
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
API_PORT=8000
TZ=Europe/Paris
```

Variables définies ou surchargées par Compose :

| Variable | Rôle | Valeur Docker |
|---|---|---|
| `API_URL` | URL FastAPI vue par Streamlit | `http://api:8000` |
| `CHROMA_HOST` | Hôte ChromaDB | `chromadb` |
| `CHROMA_PORT` | Port interne ChromaDB | `8000` |
| `CHROMA_COLLECTION` | Collection RAG par défaut | `prudencia_legal_documents` |
| `UPLOAD_DIR` | Répertoire des documents | `/uploads` |
| `HF_HOME` | Cache Hugging Face | `/models/huggingface` |

## Commandes utiles

Depuis `compose/dev` :

```bash
docker compose config
docker compose up --build
docker compose ps
docker compose logs -f api
docker compose down
```

`docker compose down` arrête les conteneurs sans supprimer les dossiers de données montés depuis l'hôte.

## Vérification

Après démarrage :

1. vérifier `GET /health` sur l'API ;
2. ouvrir `/docs` pour la documentation OpenAPI ;
3. ouvrir Streamlit sur le port 8501 ;
4. vérifier `/rag/health` avant une indexation ;
5. vérifier les routes de santé ML/DL avant une prédiction.

## Passage en production

Le fichier Compose de développement ne constitue pas un déploiement de production. Il faudrait notamment :

- supprimer `--reload` et les montages du code ;
- figer les versions des images et dépendances ;
- placer un proxy TLS devant l'API et Streamlit ;
- ajouter authentification, autorisation et limitation de débit ;
- isoler les secrets dans un gestionnaire dédié ;
- ajouter des healthchecks, sauvegardes et supervision ;
- exécuter les entraînements longs dans des workers ;
- définir des limites CPU, mémoire et taille d'upload.
