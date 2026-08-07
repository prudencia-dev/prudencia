# Interface Streamlit

## Point d'entrée

`data/streamlit/app.py` configure l'application et présente la page d'accueil. Streamlit détecte automatiquement les scripts de `pages/` et construit la navigation multipage.

`services/api.py` centralise les appels HTTP courants dans `PrudenciaAPI`. Certaines pages spécialisées possèdent aussi des fonctions clientes locales pour leurs contrats particuliers.

## Pages

| Fichier | Parcours utilisateur |
|---|---|
| `app.py` | Accueil, contexte et navigation |
| `pages/01_rag.py` | Import, indexation et gestion des documents RAG |
| `pages/02_modeles.py` | Accès aux vues des modèles et entraînements |
| `pages/03_rapport_documentaire.py` | Analyse d'un PDF, classification DL, recherche RAG et rapport |
| `pages/04_benchmark_nlp.py` | Comparaison reproductible des modèles BERT entraînés |

## Modules de la page Modèles

- `model_base.py` présente JuriBERT Base et permet une prédiction de référence.
- `model_fine_tuning.py` charge un CSV, configure le fine-tuning et affiche son rapport.
- `model_history.py` restitue les configurations et métriques des entraînements Deep Learning.

## Gestion de l'état

Streamlit réexécute le script après chaque interaction. Les valeurs devant survivre à cette réexécution sont conservées dans `st.session_state`. Les opérations longues utilisent des messages d'attente et un délai HTTP pouvant atteindre 3 600 secondes pour le fine-tuning.

## Contrat avec l'API

L'URL de base provient de `API_URL` et vaut `http://api:8000` dans Docker. L'interface :

- envoie les documents et datasets en multipart ;
- envoie les prédictions et rapports en JSON ;
- interprète le statut HTTP avant de lire la réponse ;
- transforme les réponses techniques en libellés et indicateurs compréhensibles.

## Ajouter une page

1. Créer un fichier numéroté dans `data/streamlit/pages/`.
2. Définir le titre et l'icône avec `st.set_page_config` si nécessaire.
3. Isoler les appels HTTP et prévoir les erreurs réseau.
4. Stocker uniquement l'état utile dans `st.session_state`.
5. Déplacer les composants réutilisables dans `modules/`.

Ne jamais connecter directement une page à PostgreSQL ou ChromaDB : l'API demeure la frontière applicative.
## Fine-tuning et Benchmark NLP

L'écran de fine-tuning propose trois profils d'hyperparamètres : validation
rapide, démonstration équilibrée et benchmark reproductible. Les paramètres
sont regroupés par optimisation, ressources, reproductibilité, régularisation
et sélection du checkpoint. Un récapitulatif complet est affiché avant le
lancement.

La page `04_benchmark_nlp.py` compare les entraînements de CamemBERT,
CamemBERTv2 et JuriBERT enregistrés dans PostgreSQL. Elle présente :

- les runs réussis et la couverture des trois modèles ;
- le meilleur macro-F1 ;
- un tableau et un graphique des principales métriques ;
- les hyperparamètres de chaque expérience ;
- les empreintes du dataset et du code source ;
- une alerte lorsque les runs sélectionnés ne partagent pas la même signature
  de benchmark.
