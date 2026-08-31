# Guide complet de révision — Deep Learning niveau 3 : Prudencia

## 1. Présenter Prudencia

Prudencia est un MVP d’aide à l’analyse de projets d’intelligence artificielle au regard de l’AI Act. L’application associe classification de textes, recherche documentaire, génération d’un rapport structuré et conservation des expériences.

La décision finale reste humaine. Prudencia ne délivre pas une décision juridique automatique.

### Présentation en 30 secondes

« Prudencia est mon projet Deep Learning de niveau 3. Une interface Streamlit communique avec une API FastAPI. Le backend peut préparer un dataset, fine-tuner un modèle de langue, réaliser une prédiction, rechercher des passages réglementaires avec un RAG et produire un rapport. PostgreSQL conserve les données et l’historique, ChromaDB stocke les vecteurs documentaires et Docker Compose réunit les services. »

### Présentation en une minute

« Le cœur du projet traite du texte. Pour la classification, un modèle Transformer pré-entraîné est adapté à partir d’un CSV contrôlé par le backend. Pour compléter la prédiction, le RAG découpe et vectorise des documents, puis recherche les passages proches de la demande. FastAPI sépare ces fonctions de l’interface Streamlit. PostgreSQL conserve les documents, exécutions et modèles, tandis que ChromaDB est spécialisé dans la recherche vectorielle. Prudencia démontre ainsi le passage d’un entraînement isolé à une application modulaire et traçable, tout en restant un MVP qui nécessite une validation humaine. »

## 2. Position dans la progression

| Niveau | Projet | Apport principal |
|---|---|---|
| 1 | `poc` | comprendre un premier fine-tuning JuriBERT |
| 2 | `certification-dl` | comparer baseline, stratégies, métriques et ressources |
| 3 | Prudencia | intégrer le DL dans une application avec API, interface, données et conteneurs |

Prudencia est un projet Deep Learning. Il ne contient pas de projet Machine Learning à présenter au jury.

## 3. Architecture générale

```text
Utilisateur
   ↓
Streamlit : écrans, formulaires et affichage
   ↓ HTTP
FastAPI : validation, orchestration et services
   ├── fine-tuning et prédiction Transformer
   ├── extraction et traitement des documents
   ├── RAG et recherche vectorielle
   ├── construction des rapports
   └── registre et historique des modèles
   ↓                         ↓
PostgreSQL                ChromaDB
données structurées       vecteurs documentaires
```

Docker Compose démarre quatre services : PostgreSQL, API FastAPI, ChromaDB et Streamlit.

## 4. Responsabilité des composants

### Streamlit

- collecte les fichiers et paramètres ;
- appelle l’API avec `requests` ;
- affiche qualité du dataset, entraînement, métriques et historique ;
- présente la comparaison des modèles et les rapports ;
- ne doit pas contenir la logique principale d’entraînement.

### FastAPI

- expose les routes HTTP ;
- valide les entrées avec des modèles Pydantic ;
- orchestre datasets, modèles, RAG et rapports ;
- fournit automatiquement la documentation Swagger ;
- sépare l’interface de la logique métier.

### PostgreSQL

- documents et métadonnées ;
- chunks associés aux documents ;
- exécutions d’entraînement ;
- historique des modèles ;
- statut actif ou archivé ;
- données utiles à la traçabilité.

### ChromaDB

- embeddings des passages documentaires ;
- métadonnées nécessaires à la recherche ;
- recherche par proximité vectorielle ;
- rôle différent de PostgreSQL, qui stocke les données relationnelles.

### Docker Compose

- décrit les services et leurs dépendances ;
- fixe les ports et volumes ;
- facilite la reproduction de l’environnement ;
- ne remplace pas à lui seul les tests ni le monitoring.

## 5. Pipeline de fine-tuning actuel

```text
CSV envoyé depuis Streamlit
→ endpoint FastAPI `/fine-tuning/train`
→ chargement et validation du dataset
→ rapport de qualité
→ préparation texte + label
→ tokenizer du modèle sélectionné
→ split entraînement/évaluation
→ fine-tuning avec Trainer
→ métriques
→ sauvegarde modèle + tokenizer
→ enregistrement de l’exécution dans PostgreSQL
→ affichage dans Streamlit
```

### Modules principaux

| Module | Rôle |
|---|---|
| `ai/fine_tuning/dataset.py` | charge, valide et prépare le CSV |
| `ai/fine_tuning/trainer.py` | tokenise, entraîne, évalue, sauvegarde et prédit |
| `ai/fine_tuning/experiment.py` | crée et suit les exécutions en base |
| `ai/fine_tuning/benchmark.py` | compare des résultats enregistrés |
| `api/fine_tuning.py` | expose entraînement, historique, meilleur modèle, reset et prédiction |
| `services/training_history_service.py` | persiste l’historique et le modèle actif |
| `services/model_registry_service.py` | liste, active et archive les modèles |

## 6. Préparation du dataset

`DatasetManager` contrôle notamment :

- existence des colonnes choisies ;
- textes et labels exploitables ;
- valeurs manquantes et doublons ;
- distribution des classes ;
- volume minimal ;
- possibilité d’entraîner ;
- recommandations liées à la qualité.

Le rapport de qualité évite de lancer silencieusement un entraînement sur un dataset inutilisable.

## 7. Entraînement Transformer

### Étapes conceptuelles

1. charger checkpoint et tokenizer ;
2. créer les mappings `label2id` et `id2label` ;
3. tokeniser les textes ;
4. créer les ensembles d’entraînement et d’évaluation ;
5. lancer le `Trainer` ;
6. calculer accuracy, précision, rappel et F1 ;
7. sauvegarder modèle et tokenizer ;
8. enregistrer la configuration et les métriques.

### Paramètres exposés

| Paramètre | Signification |
|---|---|
| `model_name` | modèle pré-entraîné sélectionné |
| `epochs` | nombre maximal de passages sur l’entraînement |
| `batch_size` | exemples traités simultanément |
| `learning_rate` | amplitude de correction des poids |
| colonne texte | contenu fourni au tokenizer |
| colonne cible | classe attendue |

### Limite actuelle importante

Le backend actuel réalise un fine-tuning standard. Il ne permet pas encore de choisir explicitement entre encodeur gelé, fine-tuning complet et LoRA. Il ne faut donc pas annoncer cette fonctionnalité comme opérationnelle dans Prudencia.

La comparaison de ces stratégies est démontrée dans `certification-dl`. Son intégration future dans Prudencia reste une évolution possible.

## 8. Prédiction

```text
texte utilisateur
→ POST `/fine-tuning/predict`
→ chargement du modèle et du tokenizer sauvegardés
→ tokenisation identique à l’entraînement
→ passage en mode évaluation sans gradients
→ logits → probabilités
→ classe, confiance et détail des probabilités
```

Une probabilité élevée n’est pas une garantie juridique. Le résultat doit être présenté avec ses limites.

## 9. RAG

RAG signifie Retrieval-Augmented Generation, ou génération augmentée par recherche. Dans Prudencia, la brique essentielle est la recherche documentaire : elle retrouve des passages pertinents pour enrichir l’analyse.

### Indexation

```text
PDF → extraction du texte → découpage en chunks
→ embeddings BGE-M3 → stockage ChromaDB
→ métadonnées PostgreSQL
```

### Recherche

```text
question → embedding → comparaison vectorielle
→ passages les plus proches → contexte du rapport
```

### Fine-tuning et RAG

| Fine-tuning | RAG |
|---|---|
| modifie les poids | ne modifie pas les poids |
| adapte une tâche | apporte des documents |
| demande un entraînement | corpus actualisable sans réentraînement |
| connaissance inscrite dans le modèle | connaissance récupérée au moment de l’analyse |

Ajouter un document dans ChromaDB ne l’apprend pas au modèle de classification.

## 10. Construction du rapport

`AnalysisOrchestrator` coordonne les données de l’analyse. `PrudenciaReportBuilder` construit un objet métier structuré comprenant notamment :

- description du projet ;
- classification AI Act proposée ;
- risques détectés ;
- recommandations ;
- références juridiques ;
- conclusion et statut indicatif.

Le builder déduplique et normalise les informations. Il sépare la production des éléments de leur affichage dans Streamlit.

## 11. Registre et historique

Le registre permet de :

- lister les modèles ;
- consulter leurs métadonnées ;
- activer une version ;
- archiver une version ;
- retrouver les runs d’entraînement.

L’historique conserve configuration, métriques, durée, statut et erreurs. Cette traçabilité est un principe MLOps utile, sans constituer une plateforme MLOps complète.

## 12. Notions MLOps réellement présentes

- séparation entraînement et inférence ;
- configuration enregistrée ;
- historique des expériences ;
- registre et activation des modèles ;
- conteneurisation ;
- endpoints de santé ;
- volumes persistants ;
- logs et états d’exécution.

## 13. Ce qui reste nécessaire pour une production industrielle

- authentification et gestion fine des droits ;
- protection et rotation des secrets ;
- tests d’intégration complets ;
- pipeline CI/CD démontré ;
- supervision disponibilité, latence, mémoire et erreurs ;
- détection de dérive des textes et prédictions ;
- stratégie formelle de rollback ;
- sauvegardes et reprise après incident ;
- validation sur un jeu de test indépendant plus solide ;
- audit de sécurité et conformité des données.

## 14. Questions du jury

### Pourquoi séparer Streamlit et FastAPI ?

Streamlit gère l’expérience utilisateur. FastAPI centralise validation, logique métier et modèles. Cette séparation permet de changer l’interface ou d’ajouter un autre client sans réécrire le backend.

### Pourquoi PostgreSQL et ChromaDB ?

PostgreSQL gère les relations et l’historique structuré. ChromaDB est optimisé pour les vecteurs et la recherche par similarité.

### Pourquoi Docker Compose ?

Le projet est un MVP à quatre services. Compose suffit pour décrire et démarrer cette architecture sans ajouter la complexité de Kubernetes.

### Pourquoi un modèle juridique ?

Un modèle pré-entraîné sur du texte juridique peut fournir des représentations plus proches du domaine. Cette adéquation reste une hypothèse à vérifier par les métriques du projet.

### Prudencia décide-t-il de la conformité ?

Non. Il prépare une analyse et des références. La décision finale reste humaine.

### Le RAG rend-il le modèle plus intelligent ?

Il lui fournit un contexte documentaire plus pertinent. Il ne modifie pas automatiquement ses poids et ne garantit pas que la réponse soit correcte.

### Prudencia utilise-t-il LoRA ?

Pas dans le backend actuel. LoRA est étudié dans le projet DL niveau 2 et constitue une évolution possible.

### Prudencia est-il en production ?

Il est présenté comme un MVP déployable et conteneurisé. Une production industrielle demanderait davantage de sécurité, tests, monitoring et procédures opérationnelles.

### Que monitorez-vous ?

Le code conserve les exécutions et propose des états de santé. Un monitoring complet devrait aussi suivre disponibilité, latence, erreurs, ressources et dérive.

## 15. Questions pièges

### Avec peu de données, le score est-il fiable ?

Il décrit uniquement le protocole et le jeu évalué. Il faut présenter le volume, les classes, les erreurs et les limites, puis confirmer sur un test indépendant plus important.

### Une confiance de 95 % signifie-t-elle que la décision est correcte ?

Non. Il s’agit d’une probabilité produite par le modèle. Sa fiabilité dépend notamment de la calibration et de la proximité entre les nouvelles données et les données d’entraînement.

### Pourquoi ne pas utiliser uniquement un service généraliste ?

Prudencia permet de maîtriser le dataset, le modèle, le corpus documentaire, les étapes d’analyse et la traçabilité. Cela ne garantit pas automatiquement une meilleure performance.

## 16. Limites à annoncer honnêtement

- dataset spécialisé limité ;
- métriques dépendantes du split et du protocole ;
- pas de test indépendant clairement séparé dans le trainer actuel ;
- pas de choix gelé/complet/LoRA dans le backend actuel ;
- pas de matrice de confusion et macro-F1 complètes dans le trajet applicatif actuel ;
- RAG dépendant de la qualité du corpus et du découpage ;
- application d’aide, pas avis juridique ;
- monitoring et sécurité à renforcer.

## 17. Fiche express

- Prudencia = Deep Learning niveau 3, sans projet ML.
- Streamlit = interface ; FastAPI = logique et API.
- PostgreSQL = données structurées ; ChromaDB = vecteurs.
- Transformer = classification de textes.
- fine-tuning = adaptation des poids.
- RAG = recherche de contexte sans modifier les poids.
- registre = versions et activation des modèles.
- historique = traçabilité des entraînements.
- Docker Compose = quatre services reproductibles.
- MVP fonctionnel, pas plateforme industrielle complète.
- décision finale humaine.
- LoRA et test indépendant restent des évolutions, pas des fonctions actuelles.

## 18. Phrase finale pour la soutenance

« Prudencia montre le passage d’un modèle expérimental à une application structurée. J’ai séparé l’interface, l’API, les données relationnelles, la recherche vectorielle et les services de Deep Learning. Je peux expliquer ce qui fonctionne aujourd’hui, les limites du MVP et les étapes nécessaires avant une véritable mise en production. »
