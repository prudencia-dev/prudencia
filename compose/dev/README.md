# Base de données locale

PostgreSQL utilise le dossier local `data/postgres`. Les scripts de `config/init` sont exécutés automatiquement uniquement lorsque ce dossier est vide.

## Sauvegarde logique

```bash
docker compose exec -T postgres \
  pg_dump -U prudencia -d prudencia -Fc \
  -f /tmp/prudencia.dump

docker cp prudencia-postgres:/tmp/prudencia.dump ./prudencia.dump
```

Conserver le dump hors du dépôt Git et vérifier son contenu avec `pg_restore -l`.

## Recréation

1. Arrêter les services avec `docker compose down`.
2. Déplacer `data/postgres` vers un dossier de sauvegarde daté.
3. Recréer un dossier `data/postgres` vide.
4. Relancer avec `docker compose up -d --build`.
5. Vérifier les tables et les routes de santé.

`docker compose down -v` ne supprime pas `data/postgres`, car il s'agit d'un montage de dossier et non d'un volume Docker nommé.

## Tables attendues

- `public.analyses` ;
- `prudencia.documents` ;
- `prudencia.document_chunks` ;
- `prudencia.model_executions` ;
- `prudencia.trained_models` ;
- `prudencia.model_training_runs`.

Les scripts sont idempotents, mais ils ne remplacent pas un outil de migration pour les évolutions futures d'une base déjà initialisée.
