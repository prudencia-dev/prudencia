# Questionnaire PRUDENCIA MVP
## Jeux de réponses pour la démonstration

Version : Certification RNCP38616
Projet : PRUDENCIA
Auteur : Jean-Philippe

---

# Cas n°1 — IA à Haut Risque (Recommandé pour la démonstration)

## Nom du projet

RecrutIA

## Description

Application d'intelligence artificielle permettant d'assister les recruteurs
dans la présélection automatique des candidats.

## Réponses

| Question | Réponse |
|----------|----------|
| Secteur | RH / Emploi |
| Données personnelles | Oui |
| Données sensibles | Non |
| Type d'IA | Scoring |
| Rôle | Déployeur |

## Résultat attendu

Classe :

```
haut_risque
```

Pourquoi ?

- décision ayant un impact sur une personne
- domaine RH
- données personnelles

---

# Cas n°2 — Risque Minimal

## Nom

Assistant documentaire interne

## Description

Assistant conversationnel utilisé uniquement par les collaborateurs pour
rechercher des procédures internes.

## Réponses

| Question | Réponse |
|----------|----------|
| Secteur | Services |
| Données personnelles | Non |
| Données sensibles | Non |
| Type d'IA | Chatbot documentaire |
| Rôle | Déployeur |

## Résultat attendu

```
risque_minimal
```

---

# Cas n°3 — Risque Limité

## Nom

Chatbot Service Client

## Description

Assistant conversationnel répondant automatiquement aux questions des clients.

## Réponses

| Question | Réponse |
|----------|----------|
| Secteur | Commerce |
| Données personnelles | Oui |
| Données sensibles | Non |
| Type d'IA | Chatbot |
| Rôle | Déployeur |

## Résultat attendu

```
risque_limite
```

---

# Cas n°4 — Pratique Interdite

## Nom

Social Scoring Citoyen

## Description

IA attribuant une note aux citoyens afin de limiter l'accès à certains services.

## Réponses

| Question | Réponse |
|----------|----------|
| Secteur | Public |
| Données personnelles | Oui |
| Données sensibles | Oui |
| Type d'IA | Social Scoring |
| Rôle | Fournisseur |

## Résultat attendu

```
interdit
```

---

# Déroulement conseillé pendant la soutenance

1. Création du projet
2. Remplissage du questionnaire
3. Lancement de l'analyse
4. Affichage de la prédiction Machine Learning
5. Génération automatique du rapport PRUDENCIA
6. Présentation des obligations AI Act
7. Présentation des recommandations

---

# Conseils

Pour la démonstration officielle, privilégier le **Cas n°1 (RecrutIA)**.

Il met en évidence :

- la création du projet ;
- l'enregistrement du questionnaire ;
- l'utilisation du modèle Random Forest ;
- la génération du rapport automatique ;
- l'interprétation réglementaire de l'AI Act.

Ce scénario est suffisamment riche pour démontrer l'ensemble de la chaîne de traitement de PRUDENCIA tout en restant simple à expliquer devant le jury.