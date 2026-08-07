# 10 — Documentation technique et schémas d'architecture

## Objectif

Ce lot consolide la documentation technique du MVP PRUDENCIA v1.2.0 et fournit deux représentations visuelles complémentaires : l'architecture d'exécution et l'organisation du code source.

## Modifications réalisées

- refonte de `DOCUMENTATION_TECHNIQUE.md` en guide technique détaillé ;
- description des composants Streamlit, FastAPI, PostgreSQL, ChromaDB et des services d'intelligence artificielle ;
- documentation des flux d'indexation RAG, de recherche, de génération des rapports et de fine-tuning ;
- ajout des informations de configuration, sécurité, observabilité, tests, exploitation et maintenance ;
- création de `SCHEMA_ARCHITECTURE_TECHNIQUE.md` avec des diagrammes Mermaid de l'architecture Docker et des principaux flux ;
- création de `SCHEMA_ARCHITECTURE_CODE_SOURCE.md` avec les dépendances entre couches, l'arborescence fonctionnelle et les chaînes d'appels ;
- export des huit diagrammes Mermaid au format image SVG dans `md/images/` ;
- intégration de chaque image dans sa section tout en conservant le diagramme Mermaid modifiable ;
- ajout des nouveaux documents à l'index `md/README.md`.

## Fichiers concernés

- `md/DOCUMENTATION_TECHNIQUE.md`
- `md/SCHEMA_ARCHITECTURE_TECHNIQUE.md`
- `md/SCHEMA_ARCHITECTURE_CODE_SOURCE.md`
- `md/images/*.svg`
- `md/README.md`
- `md/10_RAPPORT_EVOLUTION_DOCUMENTATION_ARCHITECTURE.md`

## Contrôles effectués

- contrôle des erreurs de format avec `git diff --check` ;
- vérification automatique de tous les liens relatifs des 23 fichiers Markdown ;
- contrôle de la présence et de la fermeture des blocs Mermaid ;
- rendu des huit diagrammes avec Mermaid CLI, sans erreur de génération ;
- comparaison de la documentation avec la structure et les composants de la version v1.2.0.

## Résultat

La documentation permet désormais de comprendre le déploiement du projet, ses responsabilités techniques, les dépendances entre modules et les parcours de données sans devoir parcourir préalablement l'ensemble du code source.
