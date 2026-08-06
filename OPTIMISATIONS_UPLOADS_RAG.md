# Optimisations des uploads et du RAG

## Contexte

Ce document décrit le premier lot d'optimisations réalisé sur PRUDENCIA avant
publication. Les modifications sont développées sur la branche
`codex/secure-uploads-rag` à partir de `develop`.

Le lot poursuit quatre objectifs :

1. sécuriser les fichiers PDF et CSV reçus par l'API ;
2. limiter leur consommation de mémoire et d'espace disque ;
3. garantir la cohérence des embeddings stockés dans ChromaDB ;
4. remplacer les scripts de démonstration par de vrais tests automatisés.

## 1. Sécurisation des fichiers téléversés

### Service de sécurité commun

Le fichier `data/fastapi/app/services/upload_security.py` centralise désormais
les contrôles appliqués aux uploads.

Il fournit notamment :

- la normalisation du nom transmis par le client ;
- la suppression des composantes de chemin comme `../` ou `../../` ;
- la validation de l'extension attendue ;
- la création d'un nom interne aléatoire avec un UUID ;
- la vérification que le chemin final reste dans le dossier autorisé ;
- la lecture et la copie avec une limite stricte de taille ;
- la suppression d'un fichier partiellement écrit lorsqu'une erreur survient ;
- la vérification de la signature `%PDF-` pour les fichiers PDF.

### Limites de taille

Les limites par défaut sont :

| Type | Limite |
|---|---:|
| PDF | 20 Mo |
| CSV | 10 Mo |

Elles sont configurables avec les variables d'environnement suivantes :

```text
MAX_PDF_UPLOAD_BYTES
MAX_CSV_UPLOAD_BYTES
```

Les valeurs de développement sont déclarées dans
`compose/dev/compose.yaml` :

```yaml
MAX_PDF_UPLOAD_BYTES: 20971520
MAX_CSV_UPLOAD_BYTES: 10485760
```

Lorsqu'un fichier dépasse la limite, l'API répond avec le code HTTP `413`
au lieu de poursuivre son chargement ou son traitement.

### Lecture contrôlée

Les fichiers ne sont plus lus sans limite. Le service lit au maximum la taille
autorisée plus un octet. Cet octet supplémentaire permet de détecter un
dépassement sans charger arbitrairement tout le fichier en mémoire.

Pour les fichiers persistants, la copie est effectuée par blocs de 1 Mo.

### Contrôle réel des PDF

Le champ HTTP `content_type` est fourni par le client et ne constitue pas une
preuve suffisante. PRUDENCIA contrôle maintenant :

1. l'extension `.pdf` ;
2. la signature binaire `%PDF-` ;
3. la capacité de PyMuPDF à ouvrir le document.

Un fichier renommé artificiellement en `.pdf` est donc refusé.

### Noms originaux et noms internes

Le nom original est conservé dans PostgreSQL pour l'affichage. Le fichier
physique reçoit un nom interne de la forme :

```text
f6a8f4d08e934365861d07658c9a4c21.pdf
```

Cette séparation évite :

- les collisions entre deux fichiers portant le même nom ;
- l'écrasement silencieux d'un document existant ;
- l'utilisation du nom client comme chemin sur le serveur ;
- la divulgation directe des noms métiers dans le stockage physique.

### Nettoyage après erreur

Si la copie, la validation ou l'insertion PostgreSQL échoue, le fichier créé
est supprimé. Cela évite de conserver des fichiers incomplets ou non référencés
dans la base.

La suppression d'un document résout également son chemin sous le dossier
`UPLOAD_DIR`. Un chemin historique enregistré hors de ce dossier ne peut plus
être supprimé par le service.

### Routes concernées

Les protections sont appliquées aux fonctionnalités suivantes :

- upload documentaire général ;
- indexation PDF dans le RAG ;
- analyse ponctuelle d'un PDF ;
- entraînement Machine Learning depuis un CSV ;
- fine-tuning depuis un CSV.

## 2. Uniformisation et optimisation du RAG

### Modèle unique

Le RAG utilise désormais exclusivement :

```text
BAAI/bge-m3
```

La constante est centralisée dans
`data/fastapi/app/ai/embedding_config.py`.

Auparavant, certains chunks étaient encodés avec CamemBERT tandis que les
requêtes étaient encodées avec BGE-M3. Ces modèles peuvent produire des
vecteurs de dimensions différentes, ce qui rend une collection ChromaDB
incohérente ou inutilisable.

CamemBERT reste disponible pour les fonctionnalités qui ne participent pas à
la collection vectorielle du RAG.

### Embeddings par lot

La fonction `get_embeddings()` encode tous les chunks d'un document en un seul
appel à Sentence Transformers.

Cette approche réduit :

- le nombre d'appels Python vers le modèle ;
- le coût de préparation de chaque inférence ;
- le temps total d'indexation ;
- les transferts répétés entre le processeur et l'accélérateur éventuel.

Les embeddings restent normalisés avec `normalize_embeddings=True`.

### Métadonnées ChromaDB

Chaque collection et chaque chunk documentent maintenant :

```text
embedding_model
embedding_dimension
embedding_normalized
```

Pour BGE-M3, la dimension attendue est déterminée depuis le vecteur réellement
produit, au lieu d'être écrite en dur.

### Contrôles de cohérence

Avant une insertion, PRUDENCIA vérifie que :

- le nombre d'identifiants correspond au nombre de textes ;
- le nombre d'embeddings correspond au nombre de métadonnées ;
- tous les embeddings d'un lot ont la même dimension ;
- la dimension reçue correspond à celle de la collection ;
- la collection déclare bien BGE-M3 comme modèle ;
- la recherche utilise la même dimension que l'indexation.

Une incohérence produit une erreur explicite avant l'appel à ChromaDB.

## 3. Migration de la collection existante

Une collection ChromaDB existante peut contenir des embeddings CamemBERT ou ne
pas posséder les nouvelles métadonnées. PRUDENCIA refuse volontairement de
continuer avec une collection non identifiable afin de ne pas mélanger des
vecteurs incompatibles.

### Procédure recommandée

1. vérifier que les PDF sources sont encore disponibles ;
2. réinitialiser la collection RAG depuis l'interface ou l'API ;
3. réimporter et réindexer les documents ;
4. contrôler dans les statistiques que les chunks ont été recréés ;
5. effectuer une recherche de validation.

La réinitialisation supprime les embeddings, les chunks, les enregistrements
documentaires et les PDF importés. Il faut donc conserver une copie des sources
avant cette opération.

## 4. Tests automatisés

Les anciens fichiers `test_benchmark.py` et `test_trainer.py` étaient des
scripts exécutés au moment de leur import. L'un d'eux pouvait lancer un véritable
fine-tuning sans fournir d'assertions automatisées.

Ils ont été remplacés par des tests unitaires qui :

- n'entraînent pas de modèle Transformers ;
- vérifient le classement du benchmark ;
- vérifient le comportement sans résultat ;
- contrôlent que le dataset de démonstration est exploitable.

### Tests de sécurité ajoutés

`test_upload_security.py` vérifie :

- la neutralisation des chemins fournis par le client ;
- la génération de noms internes uniques ;
- le confinement dans le dossier autorisé ;
- le refus d'un faux PDF ;
- la suppression d'un fichier invalide ;
- le refus et le nettoyage d'un fichier trop volumineux ;
- l'acceptation d'un fichier exactement égal à la limite.

### Tests RAG ajoutés

`test_rag_embeddings.py` vérifie :

- l'initialisation des métadonnées BGE-M3 ;
- le refus d'une collection historique non identifiable ;
- le refus d'un lot contenant plusieurs dimensions d'embedding.

### Dépendances de développement

Le fichier `data/fastapi/requirements-dev.txt` référence les dépendances de
l'application et ajoute `pytest` pour l'exécution des tests.

## 5. Validations réalisées

Les contrôles suivants ont été exécutés :

```text
12 tests réussis
Lint Ruff réussi sur les fichiers modifiés
Configuration Docker Compose valide
git diff --check réussi
```

La commande de tests utilisée est :

```bash
PYTHONPATH=data/fastapi pytest -q data/fastapi/tests
```

Le serveur Docker n'était pas actif pendant les contrôles. Les tests utilisent
donc des doubles pour ChromaDB et ne constituent pas encore un test d'intégration
avec PostgreSQL, ChromaDB et les modèles Transformers réels.

## 6. Fichiers principaux ajoutés

- `data/fastapi/app/services/upload_security.py` ;
- `data/fastapi/app/ai/embedding_config.py` ;
- `data/fastapi/tests/conftest.py` ;
- `data/fastapi/tests/test_upload_security.py` ;
- `data/fastapi/tests/test_rag_embeddings.py` ;
- `data/fastapi/requirements-dev.txt`.

## 7. Fichiers principaux modifiés

- `compose/dev/compose.yaml` ;
- `data/fastapi/app/main.py` ;
- `data/fastapi/app/ai/bge_m3.py` ;
- `data/fastapi/app/ai/rag.py` ;
- `data/fastapi/app/api/document_analysis.py` ;
- `data/fastapi/app/api/fine_tuning.py` ;
- `data/fastapi/app/api/ml.py` ;
- `data/fastapi/app/api/rag.py` ;
- `data/fastapi/app/services/document_service.py` ;
- `data/fastapi/app/services/pdf_service.py` ;
- `data/fastapi/app/services/rag_service.py` ;
- `data/fastapi/tests/test_benchmark.py` ;
- `data/fastapi/tests/test_trainer.py`.

## 8. État avant publication

- branche : `codex/secure-uploads-rag` ;
- modifications présentes uniquement dans le worktree dédié ;
- aucun commit créé pour ce lot ;
- aucun push effectué pour ce lot.
