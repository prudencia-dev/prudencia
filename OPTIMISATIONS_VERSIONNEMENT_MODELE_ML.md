# Versionnement et cache du modèle Machine Learning

## Objectif

Ce lot remplace le fichier Random Forest unique par un stockage versionné avec
activation atomique et cache en mémoire. Il est développé sur la branche
`codex/versioned-model-cache`, indépendamment du lot uploads/RAG.

## Changements

### Versions immuables

Chaque entraînement crée désormais un répertoire distinct :

```text
models/machine_learning/versions/<version>/
├── model.joblib
└── metadata.json
```

La version combine un horodatage UTC précis et un identifiant aléatoire. Un
nouvel entraînement ne remplace donc plus le modèle précédent.

### Activation atomique

Le fichier suivant désigne le modèle actif :

```text
models/machine_learning/active_model.json
```

Le nouveau pointeur est d'abord écrit dans un fichier temporaire, puis activé
avec `os.replace()`. Une prédiction ne peut donc pas observer un pointeur
partiellement écrit.

Le chemin lu depuis ce fichier est résolu et contrôlé afin qu'il ne puisse pas
sortir du répertoire des modèles.

### Métadonnées

Chaque version conserve notamment :

- la version du logiciel ;
- l'algorithme utilisé ;
- les variables attendues ;
- la colonne cible ;
- les classes apprises ;
- les métriques de validation ;
- la date de création ;
- le chemin relatif du modèle.

### Cache des prédictions

Le modèle actif est chargé avec `joblib.load()` uniquement lorsque :

- aucun modèle n'est encore présent dans le cache ;
- le chemin actif a changé ;
- la date de modification du fichier a changé.

Les prédictions suivantes réutilisent directement le pipeline en mémoire.

### Réinitialisation réversible

La réinitialisation supprime le pointeur actif et vide le cache, mais conserve
les répertoires versionnés. Les artefacts restent donc disponibles pour une
future fonctionnalité de restauration.

L'ancien fichier `random_forest.joblib` reste accepté comme mécanisme de
migration. Il est identifié avec la version `legacy-v1.1.0`.

### API

La route de santé ML retourne maintenant :

- la version réellement active ;
- le chemin du fichier actif ;
- l'état `ready` ou `not_trained`.

Les réponses de prédiction incluent également la version utilisée, ce qui
permet de rattacher chaque résultat à un artefact précis.

## Tests

Cinq tests ciblés vérifient :

- la création de versions distinctes et immuables ;
- l'activation de la dernière version ;
- un seul chargement Joblib pour plusieurs prédictions ;
- la conservation des versions après désactivation ;
- le refus d'un chemin sortant du répertoire autorisé ;
- l'intégration complète entraînement, publication, prédiction et reset.

## Validations

```text
5 tests réussis
Lint Ruff réussi
git diff --check réussi
```

## Fichiers concernés

- `data/fastapi/app/services/ml_model_store.py` ;
- `data/fastapi/app/ai/machine_learning/trainer.py` ;
- `data/fastapi/app/api/ml.py` ;
- `data/fastapi/tests/test_ml_model_store.py` ;
- `data/fastapi/tests/test_ml_trainer_versioning.py`.

## État avant publication

- aucun commit créé ;
- aucun push effectué ;
- aucun modèle réel ni donnée utilisateur modifié.
