# Documentation technique

Cette documentation décrit le code source du MVP PRUDENCIA. Elle s'adresse aux développeurs, évaluateurs techniques et futurs mainteneurs.

## Parcours conseillé

1. [Architecture et flux](architecture.md) pour comprendre les composants.
2. [Backend et API](backend-api.md) pour les contrats HTTP et la logique serveur.
3. [Interface Streamlit](interface-streamlit.md) pour les parcours utilisateur.
4. [Intelligence artificielle](intelligence-artificielle.md) pour les pipelines ML, DL et RAG.
5. [Données et persistance](donnees.md) pour PostgreSQL, ChromaDB et les artefacts.
6. [Configuration et déploiement](deploiement.md) pour exécuter le projet.
7. [Développement et tests](developpement.md) pour contribuer.
8. [Référence des modules](reference-modules.md) pour retrouver le rôle de chaque fichier.

## Principe de documentation

La documentation porte sur les responsabilités, contrats, entrées, sorties et dépendances. Le code n'est pas commenté ligne par ligne : les commentaires sont réservés aux décisions non évidentes, tandis que les noms et les fonctions décrivent le comportement courant.

La documentation interactive des routes est également générée automatiquement par FastAPI aux chemins `/docs` et `/redoc` lorsque l'API fonctionne.
