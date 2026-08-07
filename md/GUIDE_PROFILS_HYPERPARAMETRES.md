# Guide des profils et hyperparamètres NLP

## Objectif

Le profil de configuration est un ensemble prédéfini d'hyperparamètres. Il
permet de lancer rapidement un entraînement cohérent et de limiter les erreurs
de configuration manuelle.

PRUDENCIA propose trois profils adaptés à des usages différents : validation
technique, démonstration et benchmark reproductible.

## Comparaison des profils

| Paramètre | Validation rapide | Démonstration équilibrée | Benchmark reproductible |
|---|---:|---:|---:|
| Epochs maximum | 1 | 5 | 10 |
| Batch réel | 2 | 4 | 4 |
| Accumulation de gradients | 1 | 2 | 2 |
| Batch effectif | 2 | 8 | 8 |
| Learning rate | `5e-5` | `2e-5` | `2e-5` |
| Longueur maximale | 128 tokens | 256 tokens | 256 tokens |
| Seed | 42 | 42 | 42 |
| Patience early stopping | 1 | 2 | 3 |
| Weight decay | `0.01` | `0.01` | `0.01` |
| Warmup ratio | `0.05` | `0.10` | `0.10` |
| Métrique de sélection | Accuracy | Macro-F1 | Macro-F1 |
| Poids de classes | Désactivés | Activés | Activés |

## Validation rapide

Ce profil sert à vérifier le fonctionnement de la chaîne complète : lecture du
CSV, préparation du dataset, chargement du modèle, entraînement, évaluation et
enregistrement dans PostgreSQL.

Il est utile pour :

- valider une nouvelle installation ;
- contrôler le format d'un dataset ;
- détecter rapidement une erreur technique ;
- vérifier qu'un modèle Hugging Face est disponible.

Un seul epoch ne permet généralement pas au modèle de converger. Les résultats
de ce profil ne doivent donc pas servir à conclure qu'un modèle est meilleur
qu'un autre.

## Démonstration équilibrée

Ce profil recherche un compromis entre durée, consommation de mémoire et
qualité des résultats. Il convient à une présentation fonctionnelle de
PRUDENCIA lorsque l'objectif n'est pas de produire une comparaison scientifique
complète.

Les cinq epochs maximum donnent au modèle davantage de temps pour apprendre,
tandis que l'early stopping peut interrompre le run si le macro-F1 ne progresse
plus. Le batch effectif de huit reste adapté à l'environnement Docker CPU du
poste local.

## Benchmark reproductible

Ce profil est destiné à comparer :

- CamemBERT — `almanach/camembert-base` ;
- CamemBERTv2 — `almanach/camembertv2-base` ;
- JuriBERT — `dascim/juribert-base`.

Il utilise dix epochs maximum, une seed fixe et le macro-F1 comme métrique de
sélection. Les poids de classes réduisent l'influence d'une éventuelle classe
majoritaire.

L'early stopping n'empêche pas la comparaison : chaque modèle peut s'arrêter à
un moment différent si ses performances ne progressent plus. Le nombre réel
d'epochs et la durée doivent néanmoins apparaître dans le rapport final.

## Conditions d'un benchmark valide

Pour comparer les trois modèles, conserver exactement :

- le même fichier CSV ;
- les mêmes colonnes texte et label ;
- le même profil Benchmark reproductible ;
- la même seed ;
- le même découpage entraînement et validation ;
- les mêmes règles de nettoyage ;
- les mêmes poids de classes ;
- la même métrique de sélection ;
- la même version du code.

Seul le modèle de base doit changer.

PRUDENCIA calcule une signature de benchmark à partir de l'empreinte du dataset,
de l'empreinte du code et des hyperparamètres. Le tableau de bord avertit
l'utilisateur lorsque les runs sélectionnés ne partagent pas la même signature.

## Définition des hyperparamètres

### Epochs

Un epoch correspond à un passage complet sur le dataset d'entraînement. Un
nombre trop faible peut produire un sous-apprentissage. Un nombre trop élevé
peut provoquer un surapprentissage, surtout avec un petit dataset.

La valeur saisie est un maximum : l'early stopping peut arrêter l'entraînement
avant la dernière epoch.

### Learning rate

Le learning rate détermine l'amplitude des corrections appliquées aux poids du
modèle.

- une valeur trop élevée peut rendre l'entraînement instable ;
- une valeur trop faible peut ralentir la convergence ;
- `2e-5` constitue une valeur de départ courante pour le fine-tuning des modèles
  BERT.

### Batch réel

Le batch réel est le nombre d'exemples traités simultanément. Une valeur plus
élevée peut accélérer l'entraînement, mais consomme davantage de mémoire.

### Accumulation de gradients

L'accumulation additionne les gradients de plusieurs petits batches avant de
mettre à jour le modèle. Elle simule ainsi un batch plus grand sans charger tous
les exemples simultanément en mémoire.

### Batch effectif

Le batch effectif est calculé ainsi :

```text
batch effectif = batch réel × accumulation de gradients
```

Avec un batch réel de quatre et une accumulation de deux, le batch effectif est
de huit.

### Longueur maximale

La longueur maximale indique le nombre maximal de tokens transmis au modèle.
Les textes plus longs sont tronqués.

Une longueur importante conserve davantage de contexte, mais augmente le temps
de calcul et la consommation mémoire. Le rapport d'entraînement indique le
nombre de textes tronqués.

### Seed

La seed fixe les opérations pseudo-aléatoires, notamment le découpage du dataset
et certaines étapes de l'entraînement. Une seed identique améliore la
reproductibilité, mais ne garantit pas à elle seule un résultat strictement
identique sur toutes les plateformes matérielles.

### Early stopping

L'early stopping arrête l'entraînement après plusieurs epochs sans amélioration
de la métrique surveillée.

La patience correspond au nombre d'epochs tolérées sans progression. Une valeur
de zéro désactive ce mécanisme dans PRUDENCIA.

### Weight decay

Le weight decay est une régularisation qui limite l'augmentation excessive des
poids. Il contribue à réduire le surapprentissage.

### Warmup ratio

Le warmup augmente progressivement le learning rate au début du run. Il évite
de modifier trop brutalement les poids pré-entraînés pendant les premières
étapes.

Un ratio de `0.10` signifie que la montée progressive occupe environ 10 % des
étapes d'entraînement.

### Métrique du meilleur checkpoint

Cette métrique détermine le checkpoint conservé comme meilleur modèle.

- **Accuracy** : proportion totale de prédictions correctes ;
- **F1 pondéré** : tient compte du nombre d'exemples de chaque classe ;
- **Macro-F1** : calcule le F1 de chaque classe puis leur moyenne non pondérée ;
- **Loss** : mesure l'erreur optimisée par le modèle.

Le macro-F1 est recommandé pour PRUDENCIA, car il donne la même importance aux
classes rares et fréquentes.

### Poids de classes

Les poids de classes augmentent le coût des erreurs commises sur les classes
minoritaires. Ils sont utiles lorsque les catégories du dataset sont
déséquilibrées.

## Méthode recommandée

1. Tester le CSV avec le profil Validation rapide.
2. Corriger les éventuelles alertes de qualité du dataset.
3. Sélectionner le profil Benchmark reproductible.
4. Entraîner CamemBERT.
5. Réutiliser exactement le même CSV et entraîner CamemBERTv2.
6. Réutiliser exactement le même CSV et entraîner JuriBERT.
7. Ouvrir la page Benchmark NLP.
8. Filtrer sur une signature de configuration unique.
9. Comparer en priorité le macro-F1 et les métriques par classe.
10. Examiner également la durée, les textes tronqués et la matrice de confusion.

Le meilleur modèle n'est pas nécessairement celui qui possède uniquement la
meilleure accuracy. Pour une classification déséquilibrée, le macro-F1 et les
performances des classes minoritaires sont généralement plus informatifs.
