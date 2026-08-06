# Rapport d'évolution 02 — Gestion sécurisée des erreurs API

## Contexte

Plusieurs routes FastAPI concaténaient directement l'exception Python dans le champ `detail` d'une réponse HTTP. Selon l'erreur rencontrée, cette pratique pouvait révéler un chemin local, une requête SQL, un nom de service ou une information de connexion.

## Objectifs

- séparer le message destiné au client du diagnostic serveur ;
- conserver les traces nécessaires à l'exploitation ;
- uniformiser les réponses des principaux routeurs ;
- préserver les erreurs métier utiles en 400, 404 et 409 ;
- rendre ce comportement testable.

## Modifications

### Fonction centralisée

`app/api/error_responses.py` fournit `raise_api_error`. La fonction :

1. journalise le traceback avec le logger `prudencia.api` ;
2. ajoute un identifiant d'opération structuré au journal ;
3. construit une `HTTPException` avec un message public stable ;
4. conserve l'exception d'origine comme cause Python sans l'exposer au client.

### Routes adaptées

- rapports documentaires ;
- statistiques, documents, indexation, reset et suppression RAG ;
- santé, lecture, activation et archivage du registre des modèles ;
- historique des entraînements.
- entraînement, réinitialisation et prédiction Fine-Tuning.

Les erreurs de validation métier restent explicites. Seules les erreurs internes inattendues sont masquées.

### Tests

Les tests vérifient qu'une chaîne sensible fictive :

- est absente de la réponse HTTP ;
- reste présente dans les logs pour le diagnostic ;
- est associée au bon identifiant d'opération ;
- n'empêche pas l'utilisation d'un statut personnalisé comme 503.
- reste masquée lorsqu'elle traverse réellement la route de génération de rapport.

## Impact

Les clients disposent de messages prévisibles et l'équipe technique conserve les tracebacks complets. Le contrat HTTP devient moins dépendant du texte variable des bibliothèques ou de PostgreSQL.

## Suite recommandée

1. appliquer le même mécanisme aux anciens endpoints encore définis dans `main.py` ;
2. ajouter un identifiant de corrélation par requête ;
3. sérialiser les logs en JSON en production ;
4. centraliser les exceptions métier dans des classes dédiées ;
5. connecter les journaux à un outil de supervision.
