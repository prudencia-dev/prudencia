# Rapport d'évolution — Centralisation des fichiers Markdown

## Objectif

Regrouper toute la documentation du projet PRUDENCIA dans un dossier unique `md/` à la racine, conformément à la convention choisie pour la suite du développement.

## Périmètre

Quatorze fichiers existants ont été déplacés :

- présentation générale du projet ;
- documentation technique et architecture ;
- backend, API et interface Streamlit ;
- intelligence artificielle, données et déploiement ;
- développement, tests et référence des modules ;
- documentation des notebooks ;
- procédure de base de données locale ;
- rapports d'évolution CI et gestion des erreurs API.

## Règles de nommage

Les trois anciens fichiers nommés `README.md` auraient provoqué une collision. Ils ont donc reçu des noms explicites :

| Ancien emplacement | Nouvel emplacement |
|---|---|
| `README.md` | `md/README.md` |
| `docs/README.md` | `md/DOCUMENTATION_TECHNIQUE.md` |
| `notebooks/README.md` | `md/NOTEBOOKS.md` |
| `compose/dev/README.md` | `md/BASE_DONNEES_LOCALE.md` |

Les autres fichiers conservent leur nom d'origine dans `md/`.

## Liens

Tous les liens Markdown relatifs ont été adaptés au nouvel emplacement. `md/README.md` constitue désormais l'index principal et référence également les rapports d'évolution.

## Vérifications

- aucun fichier `.md` en dehors de `md/` ;
- noms de fichiers uniques ;
- liens relatifs résolus vers des fichiers existants ;
- aucune erreur d'espacement détectée par `git diff --check`.

## Convention future

Tout nouveau document ou rapport Markdown doit être créé directement dans `md/`. Les fichiers applicatifs peuvent référencer cette documentation avec un chemin commençant par `md/`.
