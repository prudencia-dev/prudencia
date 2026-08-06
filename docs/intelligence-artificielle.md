# Intelligence artificielle

PRUDENCIA combine trois approches complémentaires. Le Random Forest exploite des variables structurées, JuriBERT classe des descriptions juridiques et le RAG retrouve les passages documentaires pertinents.

## Machine Learning tabulaire

`ai/machine_learning/trainer.py` contient `MachineLearningTrainer`.

Pipeline :

1. lecture du CSV avec pandas ;
2. validation de la colonne cible et des variables ;
3. séparation entraînement/test avec scikit-learn ;
4. encodage des variables catégorielles ;
5. entraînement d'un `RandomForestClassifier` ;
6. calcul de l'accuracy, précision, rappel, F1 et matrice de confusion ;
7. calcul de l'importance des variables ;
8. sauvegarde de l'artefact avec joblib ;
9. enregistrement de l'exécution dans l'historique.

Le fichier sauvegardé contient le modèle et les informations nécessaires pour reproduire le prétraitement lors d'une prédiction. Les colonnes et catégories du CSV d'inférence doivent rester compatibles avec celles de l'entraînement.

## Deep Learning juridique

Le package `ai/fine_tuning/` sépare quatre responsabilités :

- `dataset.py` : inspection du CSV, contrôle qualité, distribution des classes et préparation des jeux ;
- `trainer.py` : tokenisation, pondération éventuelle des classes, entraînement Transformers et évaluation ;
- `benchmark.py` : comparaison et synthèse des performances ;
- `experiment.py` : persistance des exécutions et métadonnées.

Les modèles configurés dans `app/config.py` incluent JuriBERT et les paramètres par défaut. Le fine-tuning est coûteux : GPU recommandé, dataset équilibré et jeu de validation représentatif.

## RAG et embeddings

`ai/bge_m3.py` charge paresseusement `BAAI/bge-m3` avec Sentence Transformers. `ai/rag.py` communique avec ChromaDB et fournit les opérations bas niveau : collection, ajout, recherche, lecture et suppression.

`services/rag_service.py` porte le cas d'usage complet :

- validation de `chunk_size` et `chunk_overlap` ;
- extraction et mise à jour des informations documentaires ;
- création de métadonnées par passage ;
- indexation des embeddings ;
- statistiques, reset et suppression ciblée.

La pertinence dépend du découpage, de la qualité du document et de la requête. Une similarité vectorielle élevée n'est pas une preuve juridique ; les références doivent être relues dans leur contexte.

## CamemBERT

`ai/camembert.py` expose un embedding généraliste français et son état de chargement. Cette brique correspond aux endpoints historiques `/ai/*`. Le RAG principal utilise BGE-M3.

## Artefacts et reproductibilité

Les notebooks de `notebooks/` documentent les expériences pédagogiques indépendamment de l'application. Leurs datasets sont versionnés pour la démonstration. Pour une expérimentation reproductible, conserver au minimum :

- version et empreinte du dataset ;
- variables ou colonnes utilisées ;
- graine aléatoire ;
- hyperparamètres ;
- versions des bibliothèques ;
- métriques globales et par classe ;
- chemin et version de l'artefact.

## Interprétation responsable

Les prédictions sont probabilistes et sensibles aux données d'entraînement. Elles doivent être accompagnées de métriques par classe, d'un contrôle des déséquilibres et d'une validation humaine. PRUDENCIA ne constitue pas un avis juridique automatisé.
