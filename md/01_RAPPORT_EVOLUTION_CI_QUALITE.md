# Rapport d'évolution 01 — CI et contrôles qualité

## Contexte

Les pull requests PRUDENCIA ne disposaient d'aucun contrôle automatique. Les validations étaient réalisées localement, ce qui ne garantissait pas qu'une branche publiée restait testable après une nouvelle modification.

## Objectif du lot

Mettre en place un socle CI rapide et reproductible sans télécharger les modèles IA volumineux. Les contrôles ciblent les erreurs bloquantes et les composants testables indépendamment de JuriBERT et BGE-M3.

## Modifications

### Workflow GitHub Actions

Le fichier `.github/workflows/ci.yml` exécute trois jobs indépendants :

1. `python-quality` vérifie les erreurs Python critiques, compile les sources et exécute Pytest ;
2. `compose-validation` valide la structure Docker Compose avec des variables non sensibles ;
3. `postgres-schema` initialise une base PostgreSQL 16 vierge et vérifie les six tables du schéma `prudencia`.

### Dépendances CI

`data/fastapi/requirements-ci.txt` sépare les outils de contrôle des dépendances d'exécution lourdes. La CI installe uniquement Pytest et Ruff.

### Configuration locale d'exemple

`compose/dev/.env.example` documente les variables obligatoires avec un mot de passe factice. Le vrai fichier `.env` reste ignoré par Git.

## Choix techniques

- Les jobs sont parallèles afin de réduire le temps de retour.
- Le workflow s'exécute sur toutes les pull requests, y compris les PR empilées avant leur intégration dans `develop`.
- Les permissions GitHub sont limitées à la lecture du dépôt.
- Les exécutions obsolètes d'une même PR sont annulées automatiquement.
- Ruff contrôle d'abord les erreurs susceptibles de casser l'exécution. L'activation de toutes les règles sera réalisée après normalisation du code historique.
- PostgreSQL est testé au moyen du mécanisme réel `docker-entrypoint-initdb.d`.
- Aucun secret ni identifiant de base distante n'est utilisé dans la CI.

## Vérifications locales prévues

- `ruff check --select E9,F63,F7,F82 data notebooks` ;
- compilation de `data/fastapi/app` et `data/streamlit` ;
- tests de `data/fastapi/tests` ;
- `docker compose config --quiet` ;
- initialisation PostgreSQL 16 et comptage des tables.

## Évolutions futures

1. augmenter la couverture des routes FastAPI ;
2. ajouter des tests d'intégration RAG avec services simulés ;
3. produire un rapport de couverture Pytest ;
4. activer progressivement l'ensemble des règles Ruff ;
5. figer les dépendances applicatives et ajouter un audit de vulnérabilités.
