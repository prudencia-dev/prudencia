# Rapport d'évolution 08 — Intégration et recette du MVP

## Objectif

Consolider les lignes de développement validées dans une branche unique, résoudre leurs chevauchements et vérifier le fonctionnement réel du MVP avant fusion dans `develop`.

## Périmètre intégré

- sécurisation des uploads PDF et CSV ;
- cohérence des embeddings BGE-M3 et du RAG ;
- suppression de la fonctionnalité Machine Learning classique ;
- maintien du Deep Learning, du fine-tuning et du RAG ;
- initialisation reproductible de PostgreSQL ;
- contrôles CI GitHub Actions ;
- gestion sécurisée des erreurs API ;
- documentation centralisée dans `md/` ;
- identifiants de corrélation et logs JSON ;
- image Docker PyTorch CPU ;
- nettoyage Ruff du backend ;
- retrait des notebooks transférés vers `prudencia-dev/poc`.

## Résolution des chevauchements

Les conflits entre la suppression du ML et le durcissement des uploads ont été résolus selon les règles suivantes :

- ne pas restaurer les routes ni modules Random Forest ;
- conserver les contrôles de taille, de nom, de chemin et de signature PDF ;
- conserver les réponses d'erreur sans fuite d'informations internes ;
- conserver BGE-M3 comme moteur d'embedding du RAG ;
- conserver la journalisation et `X-Request-ID` ;
- retirer l'intégralité des notebooks du dépôt applicatif.

## Recette automatisée

- 25 tests Pytest réussis ;
- Ruff réussi sur le backend et ses tests ;
- contrôle CI des erreurs Python bloquantes réussi sur tout `data/` ;
- compilation de tous les fichiers Python réussie ;
- configuration Docker Compose validée ;
- contrôle Git des espaces et fins de lignes réussi ;
- aucun identifiant de la base distante détecté dans le dépôt ;
- aucun fichier Markdown placé hors de `md/`.

## Recette Docker et fonctionnelle

Une image API consolidée a été construite puis démarrée sur un port isolé, sans remplacer les services locaux existants.

Contrôles réussis :

- initialisation PostgreSQL ;
- connexion à ChromaDB ;
- `GET /health` ;
- `GET /rag/health` ;
- `GET /documents` ;
- `GET /reports/health` ;
- génération de la spécification OpenAPI ;
- propagation de `X-Request-ID` ;
- production de logs JSON structurés ;
- PyTorch `2.9.0+cpu`, sans CUDA ;
- démarrage de Streamlit sur un port isolé ;
- communication réelle de Streamlit vers l'API consolidée.

Les conteneurs temporaires de recette ont été supprimés après validation.

## Version

La version exposée par FastAPI et par la route racine est alignée sur le MVP de certification : `1.0.0`.

## Conclusion

La branche consolidée est fonctionnelle et proposée dans la pull request finale #14 vers `develop`.

La CI distante GitHub Actions a validé les trois contrôles sur l'état combiné : qualité et tests Python, configuration Docker Compose et initialisation du schéma PostgreSQL. La branche est donc prête pour la décision de fusion définitive.
