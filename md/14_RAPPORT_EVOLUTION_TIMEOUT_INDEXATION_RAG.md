# 14 — Fiabilisation des indexations RAG longues

## Problème observé

Sur CPU, certaines générations d'embeddings BGE-M3 dépassaient le délai HTTP fixe de 600 secondes. L'API terminait correctement l'indexation, mais Streamlit affichait déjà une erreur de timeout. Des indexations concurrentes pouvaient également ralentir fortement le modèle.

## Modifications réalisées

- délai de l'indexation RAG rendu configurable avec `RAG_INDEX_TIMEOUT_SECONDS` ;
- valeur locale fixée à 1800 secondes ;
- conservation du délai générique de 600 secondes pour les autres appels ;
- sérialisation des inférences BGE-M3 dans le processus API ;
- URL de l'API Streamlit lue depuis `API_URL` au lieu d'être uniquement codée en dur ;
- mise à jour des guides macOS et Windows.

## Impact

La correction ne change ni les embeddings produits ni les paramètres de découpage. Elle évite les erreurs d'interface prématurées et la contention CPU entre plusieurs calculs BGE-M3 simultanés.
