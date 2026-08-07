# Installation de PRUDENCIA sous Windows 11 — consignes pour Codex

## Mission

Installer et valider localement PRUDENCIA sous Windows 11 de la même manière que sur le poste macOS : quatre services Docker Compose, données persistantes sur l'hôte et branche Git `develop`.

Ce document peut être transmis directement à Codex sur le poste Windows. Codex doit exécuter les étapes progressivement, contrôler chaque résultat et ne pas effacer une installation ou des données existantes sans autorisation explicite.

## Résultat attendu

| Composant | Technologie | Accès depuis Windows |
|---|---|---|
| Interface | Streamlit | `http://localhost:8501` |
| API | FastAPI | `http://localhost:8000` |
| Documentation API | OpenAPI | `http://localhost:8000/docs` |
| Base relationnelle | PostgreSQL 16 | `localhost:5433` |
| Base vectorielle | ChromaDB | `http://localhost:8001` |

Les conteneurs doivent être des conteneurs **Linux** gérés par Docker Desktop avec le moteur WSL 2.

## Principes de sécurité

- ne jamais inscrire de mot de passe réel dans Git ou dans ce document ;
- conserver `compose/dev/.env` uniquement sur le poste local ;
- ne pas utiliser la base distante pour les essais locaux ;
- ne pas exécuter de suppression récursive sur `data/postgres`, `data/chromadb`, `data/models` ou `data/uploads` ;
- si ces dossiers existent déjà, les inspecter et demander confirmation avant toute réinitialisation ;
- ne pas utiliser `docker compose down -v` dans l'intention de supprimer les données : les données du projet sont montées depuis des dossiers locaux.

## 1. Vérifier l'environnement Windows

Ouvrir PowerShell et exécuter :

```powershell
git --version
docker --version
docker compose version
docker info --format '{{.OSType}}'
```

La dernière commande doit retourner `linux`. Vérifier également que Docker Desktop est démarré.

Si le dépôt doit être placé dans WSL plutôt que dans le système de fichiers Windows, employer toutes les commandes depuis la même distribution WSL. Ne pas mélanger des chemins Windows et Linux au cours de l'installation.

## 2. Cloner ou mettre à jour le dépôt

### Nouvelle installation

Adapter le dossier parent si nécessaire :

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\Documents\GitHub"
Set-Location "$env:USERPROFILE\Documents\GitHub"
git clone --branch develop https://github.com/prudencia-dev/prudencia.git Prudencia
Set-Location .\Prudencia
```

### Dépôt déjà présent

Ne pas écraser les modifications locales. Contrôler d'abord :

```powershell
Set-Location "CHEMIN_VERS\Prudencia"
git status -sb
git branch --show-current
git remote -v
```

Si le dépôt est propre :

```powershell
git switch develop
git pull --ff-only origin develop
```

Si le dépôt n'est pas propre, arrêter la mise à jour et présenter les fichiers modifiés à l'utilisateur.

## 3. Créer la configuration locale

Depuis la racine du dépôt :

```powershell
Copy-Item .\compose\dev\.env.example .\compose\dev\.env
notepad .\compose\dev\.env
```

Valeurs attendues :

```dotenv
POSTGRES_DB=prudencia
POSTGRES_USER=prudencia
POSTGRES_PASSWORD=CHOISIR_UN_MOT_DE_PASSE_LOCAL
TZ=Europe/Paris
API_PORT=8000
POSTGRES_HOST_PORT=5433
RAG_INDEX_TIMEOUT_SECONDS=1800
LOG_FORMAT=json
LOG_LEVEL=INFO
```

Utiliser un mot de passe local dédié. Ne pas reprendre automatiquement les identifiants d'une base distante.

Vérifier que le fichier est ignoré par Git :

```powershell
git check-ignore .\compose\dev\.env
```

La commande doit afficher le chemin du fichier `.env`.

## 4. Contrôler les ports

```powershell
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object LocalPort -In 5433,8000,8001,8501 |
    Select-Object LocalAddress,LocalPort,OwningProcess
```

Si un port est déjà occupé, identifier le programme avant de l'arrêter. Ne pas tuer un processus sans validation. Les ports API et PostgreSQL peuvent être modifiés avec `API_PORT` et `POSTGRES_HOST_PORT`.

## 5. Valider la configuration Docker

```powershell
Set-Location .\compose\dev
docker compose config --quiet
docker compose config --services
```

Les services attendus sont :

- `postgres` ;
- `chromadb` ;
- `api` ;
- `streamlit`.

## 6. Construire et démarrer PRUDENCIA

Le premier build télécharge Python, PyTorch CPU et les dépendances. Il peut prendre plusieurs minutes.

```powershell
docker compose up -d --build
docker compose ps
```

Attendre que `postgres` soit `healthy` et que les quatre services soient actifs. En cas d'échec :

```powershell
docker compose logs --tail=200 api postgres chromadb streamlit
```

Ne pas poursuivre aveuglément si un conteneur redémarre en boucle.

## 7. Précharger les modèles Hugging Face

Les modèles sont téléchargés à la demande et partagent le cache persistant `data/models/huggingface`. Pour reproduire l'ensemble des fonctionnalités du poste macOS, précharger les quatre dépôts Hugging Face avant la recette.

| Modèle | Identifiant Hugging Face | Usage | Nécessité |
|---|---|---|---|
| BGE-M3 | `BAAI/bge-m3` | Embeddings et recherche RAG | obligatoire pour le RAG |
| JuriBERT | `dascim/juribert-base` | Modèle Deep Learning par défaut et fine-tuning juridique | obligatoire pour le parcours DL par défaut |
| CamemBERT | `almanach/camembert-base` | Analyse textuelle historique et modèle alternatif de fine-tuning | requis pour tester toutes les routes existantes |
| CamemBERTv2 | `almanach/camembertv2-base` | Modèle alternatif de fine-tuning | facultatif, sauf si ce modèle est sélectionné |

### Télécharger les quatre dépôts dans le cache persistant

Cette commande ne charge pas simultanément tous les modèles en mémoire : elle télécharge uniquement leurs fichiers dans le cache Docker.

```powershell
docker compose exec api python -c "from huggingface_hub import snapshot_download; models=['BAAI/bge-m3','dascim/juribert-base','almanach/camembert-base','almanach/camembertv2-base']; [print(m, snapshot_download(m)) for m in models]; print('Tous les modèles sont présents dans le cache')"
```

Le volume nécessaire dépend des versions publiées sur Hugging Face. Vérifier l'espace libre dans Docker Desktop avant le téléchargement.

### Charger BGE-M3 avant la première indexation

Le premier chargement de BGE-M3 peut prendre plusieurs minutes même après son téléchargement. Il faut le terminer avant le premier test d'indexation afin d'éviter le timeout HTTP de 600 secondes de Streamlit.

```powershell
docker compose exec api python -c "from app.ai.bge_m3 import get_model; get_model(); print('BGE-M3 chargé avec succès')"
```

Laisser les commandes aller jusqu'aux messages de succès. Le cache est conservé dans `data/models/huggingface` grâce au montage Docker `/models/huggingface`.

En cas d'interruption réseau, relancer la même commande : Hugging Face doit reprendre le téléchargement depuis le cache. Ne pas supprimer un fichier `.incomplete` sans diagnostic préalable.

## 8. Vérifier les services

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/rag/health
docker compose ps
```

Ouvrir ensuite :

```powershell
Start-Process http://localhost:8000/docs
Start-Process http://localhost:8501
```

## 9. Vérifier PostgreSQL

Lister les tables du schéma applicatif :

```powershell
docker compose exec -T postgres psql -U prudencia -d prudencia -c "\dt prudencia.*"
```

Tables attendues :

- `prudencia.analyses` ;
- `prudencia.documents` ;
- `prudencia.document_chunks` ;
- `prudencia.model_executions` ;
- `prudencia.trained_models` ;
- `prudencia.model_training_runs`.

Les scripts de `config/init` ne sont exécutés automatiquement que lorsque `data/postgres` est vide lors du premier démarrage de PostgreSQL.

## 10. Recette fonctionnelle minimale

Effectuer les tests suivants dans Streamlit :

1. ouvrir l'accueil et confirmer l'absence d'erreur de connexion API ;
2. ouvrir la page RAG ;
3. importer un petit PDF textuel, non confidentiel ;
4. lancer l'indexation et confirmer le nombre de pages, chunks et embeddings ;
5. effectuer une recherche sémantique simple ;
6. vérifier que le document apparaît dans le corpus ;
7. consulter les logs de l'API en cas d'erreur.

```powershell
docker compose logs --tail=200 api
```

## 11. Contrôles Git finaux

Le démarrage local crée des données ignorées par Git. Il ne doit pas modifier le code versionné :

```powershell
Set-Location ..\..
git status -sb
git log -1 --oneline --decorate
```

La branche doit être `develop`. Signaler toute modification inattendue avant de la commiter.

## 12. Commandes d'exploitation courantes

Depuis `compose/dev` :

```powershell
# État
docker compose ps

# Journaux en continu
docker compose logs -f api streamlit

# Redémarrage simple
docker compose restart api streamlit

# Arrêt sans effacer les données
docker compose down

# Démarrage ultérieur
docker compose up -d
```

## 13. Diagnostic des problèmes fréquents

### Timeout pendant l'indexation

Vérifier d'abord le téléchargement de BGE-M3 avec la commande de préchargement de l'étape 7. Contrôler ensuite :

```powershell
docker compose logs --tail=300 api
docker stats --no-stream
```

### Docker ne peut pas monter les dossiers

Vérifier que Docker Desktop a accès au disque contenant le dépôt. Éviter les dossiers synchronisés à la demande par OneDrive si leurs fichiers ne sont pas réellement présents localement.

### PostgreSQL ne s'initialise pas

Inspecter les logs et le contenu de `data/postgres`. S'il s'agit d'une ancienne base, la sauvegarder ou la déplacer vers un dossier daté uniquement après accord de l'utilisateur, puis recréer un dossier vide.

### Les scripts shell ont des erreurs de fin de ligne

Conserver les fins de ligne Git prévues par le dépôt. Ne pas convertir globalement tous les fichiers. Contrôler la configuration avec :

```powershell
git config --get core.autocrlf
git status --short
```

### Performances insuffisantes

BGE-M3 fonctionne ici sur CPU. Dans Docker Desktop, prévoir idéalement au moins 8 Go de mémoire disponible pour les conteneurs. La première génération est plus lente ; les appels suivants réutilisent le modèle chargé et son cache.

## Compte rendu attendu de Codex Windows

À la fin, Codex doit fournir :

- le chemin local du dépôt ;
- la branche et le commit installés ;
- l'état des quatre conteneurs ;
- le résultat des routes `/health` et `/rag/health` ;
- la confirmation de la présence des quatre dépôts Hugging Face dans le cache ;
- la confirmation du chargement de BGE-M3 ;
- le résultat de la recette d'indexation ;
- les éventuelles différences avec l'installation macOS ;
- la confirmation qu'aucun secret ni fichier de données n'a été ajouté à Git.
