# Rapport d'évolution 05 — Journalisation structurée

## Objectif

Rendre les journaux de l'API exploitables dans Docker et dans un outil de centralisation, tout en conservant une sortie lisible pour le développement local.

## Modifications réalisées

- ajout d'une configuration commune pour les loggers `prudencia.*` ;
- sortie JSON par défaut sur la sortie standard du conteneur ;
- ajout automatique du `request_id` depuis le contexte de la requête ;
- conservation des champs HTTP structurés : méthode, chemin, statut et durée ;
- prise en charge de `LOG_LEVEL` et `LOG_FORMAT` ;
- repli sûr sur `INFO` et `json` en cas de valeur invalide ;
- configuration idempotente pour éviter les doublons pendant les rechargements ;
- tests unitaires du format et de la propagation du contexte.

## Exploitation

La configuration par défaut ne crée pas de fichiers de logs dans le conteneur. Docker collecte la sortie standard, ce qui évite la gestion de fichiers temporaires et permet de transmettre les événements à la solution d'observabilité choisie.

Pour une lecture locale compacte :

```env
LOG_FORMAT=text
LOG_LEVEL=DEBUG
```

Pour un environnement intégré ou de production :

```env
LOG_FORMAT=json
LOG_LEVEL=INFO
```

## Validation réalisée

- 14 tests Pytest réussis ;
- contrôle Ruff réussi sur les fichiers concernés ;
- compilation Python réussie ;
- vérification Git des espaces et fins de lignes réussie ;
- aucun identifiant de la base distante présent dans les changements ;
- ordre chronologique des rapports vérifié.
