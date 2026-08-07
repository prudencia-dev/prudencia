# Comprendre les trois profils d'entraînement

## La différence en une phrase

- **Validation rapide** : est-ce que le système fonctionne ?
- **Démonstration équilibrée** : est-ce que je peux présenter un entraînement crédible sans attendre trop longtemps ?
- **Benchmark reproductible** : quel modèle est réellement le meilleur ?

## Comparaison simple

| Profil | Objectif | Durée | Valeur des résultats |
|---|---|---:|---|
| Validation rapide | Vérifier la chaîne technique | Très courte | Insuffisante pour comparer les modèles |
| Démonstration équilibrée | Présenter le fonctionnement de PRUDENCIA | Moyenne | Crédible pour une démonstration |
| Benchmark reproductible | Comparer CamemBERT, CamemBERTv2 et JuriBERT | Longue | Adaptée à une comparaison rigoureuse |

## Validation rapide

La Validation rapide est un **test technique**.

Elle permet de vérifier que :

- le fichier CSV est accepté ;
- les colonnes texte et label sont valides ;
- le modèle peut être chargé ;
- l'entraînement démarre et se termine ;
- les métriques sont calculées ;
- le run est enregistré dans PostgreSQL ;
- le run apparaît dans le tableau de bord Benchmark NLP.

Ce profil utilise seulement un epoch. Le modèle n'a généralement pas assez
appris pour que ses résultats permettent de juger sa qualité.

Il répond à la question :

> Est-ce que mon système d'entraînement fonctionne correctement ?

Il ne permet pas de répondre sérieusement à la question :

> Quel est le meilleur modèle NLP ?

## Démonstration équilibrée

La Démonstration équilibrée est un **compromis entre rapidité et qualité**.

Elle permet de montrer :

- le choix du modèle ;
- le réglage des hyperparamètres ;
- la progression de l'entraînement ;
- les métriques globales ;
- les résultats par classe ;
- la matrice de confusion ;
- l'enregistrement du modèle et du run.

Le modèle dispose de davantage de temps pour apprendre que dans la Validation
rapide, mais l'expérience reste plus courte que le Benchmark reproductible.

Il répond à la question :

> Puis-je présenter un entraînement crédible de PRUDENCIA sans attendre trop longtemps ?

Ce profil est adapté à une démonstration devant un évaluateur. Il peut donner
une première indication sur les performances, mais il ne constitue pas la
configuration de référence pour comparer officiellement les modèles.

## Benchmark reproductible

Le Benchmark reproductible est une **expérience de comparaison**.

Il sert à entraîner successivement :

1. CamemBERT ;
2. CamemBERTv2 ;
3. JuriBERT.

Pour les trois entraînements, il faut conserver :

- le même fichier CSV ;
- les mêmes colonnes texte et label ;
- la même seed ;
- le même découpage entraînement et validation ;
- les mêmes hyperparamètres ;
- les mêmes règles de préparation des données ;
- la même version du code.

Seul le modèle de base doit changer.

Il répond à la question :

> Entre CamemBERT, CamemBERTv2 et JuriBERT, lequel obtient les meilleurs résultats sur mon dataset ?

Le tableau de bord Benchmark NLP vérifie la signature des expériences. Il
signale les runs qui ne partagent pas le même dataset, le même code ou les mêmes
hyperparamètres.

## Quel profil choisir ?

### Je viens d'installer PRUDENCIA

Choisir **Validation rapide**.

L'objectif est de détecter immédiatement un problème de dataset, de modèle, de
connexion ou d'enregistrement.

### Je prépare une présentation courte

Choisir **Démonstration équilibrée**.

L'objectif est de montrer le fonctionnement complet avec des résultats plus
représentatifs qu'un simple test technique.

### Je prépare les résultats de la certification

Choisir **Benchmark reproductible**.

Exécuter trois entraînements distincts, un pour chaque modèle, en conservant
strictement la même configuration.

## Parcours recommandé pour PRUDENCIA

1. Lancer une Validation rapide avec le dataset choisi.
2. Corriger les éventuelles erreurs ou alertes de qualité.
3. Utiliser une Démonstration équilibrée pour répéter la présentation.
4. Sélectionner le profil Benchmark reproductible.
5. Entraîner CamemBERT.
6. Réutiliser le même CSV et entraîner CamemBERTv2.
7. Réutiliser le même CSV et entraîner JuriBERT.
8. Ouvrir le tableau de bord Benchmark NLP.
9. Filtrer les résultats sur une signature de benchmark unique.
10. Comparer le macro-F1, les résultats par classe et les matrices de confusion.

## À retenir

```text
Validation rapide
    = test technique

Démonstration équilibrée
    = présentation fonctionnelle

Benchmark reproductible
    = comparaison officielle des modèles
```
