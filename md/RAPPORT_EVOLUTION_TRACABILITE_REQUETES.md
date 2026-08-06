# Rapport d'évolution — Traçabilité des requêtes API

## Contexte

Les erreurs internes sont maintenant journalisées sans être exposées dans les réponses HTTP. Il restait toutefois difficile de retrouver le bon traceback lorsqu'un utilisateur signalait une erreur parmi plusieurs requêtes simultanées.

## Objectif

Associer chaque requête à un identifiant de corrélation stable, visible à la fois par le client et dans les logs du serveur.

## Modifications

### Middleware de contexte

`app/request_context.py` ajoute `RequestContextMiddleware`. Pour chaque requête, il :

1. lit l'en-tête `X-Request-ID` ;
2. accepte la valeur uniquement si elle constitue un UUID valide ;
3. génère sinon un UUID aléatoire ;
4. lie l'identifiant au contexte asynchrone courant ;
5. mesure la durée du traitement ;
6. ajoute `X-Request-ID` à la réponse ;
7. journalise méthode, chemin, statut et durée ;
8. restaure le contexte pour éviter toute fuite entre requêtes.

### Corrélation des erreurs

`raise_api_error` ajoute désormais le même `request_id` aux logs d'exception. Un utilisateur peut communiquer la valeur reçue dans l'en-tête de réponse afin que l'équipe retrouve immédiatement le diagnostic correspondant.

### Sécurité

Une valeur libre n'est jamais recopiée dans les logs. Tout identifiant fourni par le client est analysé comme UUID ; une valeur invalide, notamment avec un saut de ligne, est remplacée. Cette règle réduit le risque d'injection dans les journaux.

## Tests

Les tests couvrent :

- la normalisation d'un UUID valide ;
- le remplacement d'une valeur invalide ;
- l'isolation et la restauration du contexte ;
- le retour de l'en-tête par le middleware sur une requête HTTP réelle de test ;
- la présence du `request_id` dans un log d'erreur ;
- la non-exposition des informations internes dans la réponse.

## Exploitation

Lors d'un incident, relever l'en-tête `X-Request-ID` de la réponse, puis rechercher cette valeur dans les logs `prudencia.requests` et `prudencia.api`.

## Suite recommandée

1. produire des logs JSON en environnement de production ;
2. ajouter l'identifiant aux réponses d'erreur globales non gérées ;
3. connecter les logs à une plateforme de supervision ;
4. définir une politique de rétention et de masquage des données.
