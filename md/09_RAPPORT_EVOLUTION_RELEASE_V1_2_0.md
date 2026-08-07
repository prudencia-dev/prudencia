# Rapport d'évolution 09 — Release v1.2.0

## Objectif

Publier l'état consolidé et validé du MVP avec un numéro de version cohérent avec l'historique Git existant.

## Choix de version

Le tag `v1.0.0` existe déjà sur une ancienne livraison et ne doit pas être déplacé ou remplacé. Le dépôt contient également le tag historique `v1.1.0-ml`.

La version stabilisée adopte donc `v1.2.0`, conformément à l'ordre des versions déjà publiées.

## Modifications

- version FastAPI et OpenAPI alignée sur `1.2.0` ;
- version affichée par Streamlit alignée sur `v1.2.0` ;
- conservation des versions historiques dans les précédents rapports ;
- préparation d'un tag Git annoté sur le commit fusionné dans `develop`.

## Validation réalisée avant publication

- Ruff et compilation Python réussis ;
- 25 tests Pytest réussis ;
- configuration Docker Compose validée ;
- API `1.2.0` confirmée par la route racine et OpenAPI ;
- santé de Streamlit confirmée après rechargement ;
- aucun secret distant détecté dans les changements ;
- aucun fichier Markdown placé hors de `md/`.

La CI distante et le tag seront vérifiés après fusion de la pull request de release.
