# 12 — Port PostgreSQL local configurable

## Objectif

Éviter le conflit entre PostgreSQL installé sur le poste hôte et PostgreSQL exécuté par Docker pour PRUDENCIA.

## Modifications réalisées

- remplacement du port hôte fixe `5432` par `POSTGRES_HOST_PORT` ;
- définition de `5433` comme valeur locale par défaut ;
- conservation du port interne Docker `5432` utilisé par FastAPI ;
- mise à jour de l'exemple d'environnement et des guides de déploiement Windows ;
- configuration du poste macOS sur le port hôte `5433`.

## Connexion depuis un outil local

```text
Hôte       : 127.0.0.1
Port       : 5433
Base       : prudencia
Utilisateur: prudencia
```

Le mot de passe reste défini uniquement dans `compose/dev/.env`.

## Impact

Ce changement ne modifie ni les données PostgreSQL ni la connexion entre conteneurs. Seuls les clients exécutés sur le poste hôte, comme DBeaver, doivent utiliser le nouveau port.
