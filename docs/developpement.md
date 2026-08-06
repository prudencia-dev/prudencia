# Développement et tests

## Environnement API local

Depuis la racine du dépôt :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r data/fastapi/requirements.txt
PYTHONPATH=data/fastapi pytest -q data/fastapi/tests
```

Certains tests ou imports IA peuvent nécessiter les modèles, services ou variables d'environnement associés. Pour une exécution intégrée, utiliser Docker Compose.

## Tests présents

| Fichier | Couverture fonctionnelle |
|---|---|
| `tests/test_trainer.py` | Préparation et entraînement du modèle ML |
| `tests/test_benchmark.py` | Calcul et organisation des métriques DL |

Les lots d'optimisation ajoutent leurs propres tests ciblés sur leurs branches avant intégration dans `develop`.

## Conventions

- Respecter PEP 8 et utiliser des annotations de type pour les interfaces publiques.
- Donner une responsabilité principale à chaque fonction ou classe.
- Placer les contrats HTTP dans `api/` et la logique réutilisable dans `services/`.
- Ne pas introduire d'accès direct aux bases depuis Streamlit.
- Ajouter une docstring aux classes et fonctions publiques lorsque le contrat n'est pas évident.
- Commenter les raisons et contraintes, pas la traduction littérale du code.
- Ne jamais versionner `.env`, secrets, modèles volumineux ou documents utilisateur.

## Ajouter une route

1. Choisir ou créer un routeur dans `app/api/`.
2. Définir les schémas Pydantic d'entrée et, si possible, de sortie.
3. Déléguer le cas d'usage à un service.
4. Convertir les erreurs attendues en statuts HTTP explicites.
5. Enregistrer le routeur dans `app/main.py`.
6. Ajouter des tests de succès, validation et erreur.
7. Mettre à jour `docs/backend-api.md`.

## Ajouter un modèle

1. Définir son identifiant et ses paramètres dans `app/config.py`.
2. Encapsuler chargement, entraînement et prédiction dans `app/ai/`.
3. Enregistrer dataset, hyperparamètres, métriques et artefact.
4. Exposer une route de santé et une erreur claire si aucun artefact n'est disponible.
5. Ajouter une vue Streamlit seulement après stabilisation du contrat API.

## Revue avant publication

```bash
git status --short
git diff --check
PYTHONPATH=data/fastapi pytest -q data/fastapi/tests
docker compose -f compose/dev/compose.yaml config
```

Contrôler aussi que la documentation correspond au comportement réel et que les fichiers générés ou secrets restent ignorés.
