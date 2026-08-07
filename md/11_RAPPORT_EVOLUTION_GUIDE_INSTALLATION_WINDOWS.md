# 11 — Guide d'installation Windows 11

## Objectif

Fournir un document autonome transmissible à Codex sur un poste Windows 11 afin de reproduire l'environnement local Docker de PRUDENCIA installé sous macOS.

## Modification réalisée

- ajout d'un parcours PowerShell pour cloner ou mettre à jour la branche `develop` ;
- configuration locale sécurisée du fichier `.env` ;
- construction et démarrage des quatre services Docker ;
- préchargement persistant de BGE-M3, JuriBERT, CamemBERT et CamemBERTv2 ;
- distinction entre les modèles obligatoires et les modèles alternatifs afin d'éviter les téléchargements inutiles ;
- chargement explicite de BGE-M3 pour éviter le timeout initial ;
- vérification de FastAPI, Streamlit, PostgreSQL et ChromaDB ;
- recette fonctionnelle minimale de l'indexation RAG ;
- diagnostic des problèmes propres à Windows, Docker Desktop, WSL 2 et aux fins de ligne ;
- définition du compte rendu attendu de Codex Windows.

## Fichiers concernés

- `md/INSTALLATION_WINDOWS_11_CODEX.md`
- `md/11_RAPPORT_EVOLUTION_GUIDE_INSTALLATION_WINDOWS.md`
- `md/README.md`

## Sécurité

Le guide ne contient aucun secret réel. Il demande l'emploi d'un mot de passe exclusivement local et interdit toute suppression de données existantes sans accord préalable.
