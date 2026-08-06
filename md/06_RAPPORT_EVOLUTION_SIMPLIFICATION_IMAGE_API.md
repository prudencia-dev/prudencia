# Rapport d'évolution 06 — Simplification de l'image API

## Objectif

Réduire l'image Docker de l'API sans retirer les fonctions d'inférence, de RAG ou de fine-tuning encore présentes dans le MVP.

## Diagnostic

L'image locale existante occupait environ 9,3 Go. Son environnement Python contenait notamment :

- environ 1,2 Go pour PyTorch ;
- environ 2,7 Go de bibliothèques NVIDIA/CUDA ;
- une distribution CUDA non utilisée par le poste local.

La reconstruction tentait donc de télécharger plusieurs gigaoctets sans bénéfice fonctionnel pour Docker Desktop.

## Modifications

- installation explicite de PyTorch 2.9.0 depuis l'index CPU officiel ;
- alignement de la version déclarée dans `requirements.txt` ;
- suppression de `--reload` de la commande par défaut de l'image ;
- conservation de `--reload` dans le Compose de développement ;
- mise à jour de la documentation après le transfert des notebooks vers le dépôt privé `prudencia-dev/poc`.

## Périmètre fonctionnel conservé

- API FastAPI ;
- PostgreSQL et ChromaDB ;
- embeddings et recherche RAG ;
- inférence CamemBERT/JuriBERT ;
- fonctions de fine-tuning existantes.

## Résultats de validation

- construction complète de l'image CPU réussie ;
- PyTorch `2.9.0+cpu`, `torch.version.cuda=None` et CUDA indisponible ;
- image réduite de 9,31 Go à 2,73 Go, soit environ 71 % de réduction ;
- démarrage temporaire réussi sur le port isolé `18000` ;
- initialisation PostgreSQL réussie au démarrage ;
- `GET /health` et `GET /rag/health` réussis ;
- génération de la spécification OpenAPI réussie ;
- propagation de `X-Request-ID` et logs JSON vérifiées dans le conteneur ;
- 14 tests Pytest réussis ;
- compilation Python réussie ;
- contrôle Git des espaces et fins de lignes réussi.

Le contrôle Ruff global signale 15 écarts préexistants dans des modules non modifiés par ce lot. Ils ne bloquent pas le fonctionnement vérifié ici et seront traités séparément afin de ne pas élargir ce changement de simplification Docker.
