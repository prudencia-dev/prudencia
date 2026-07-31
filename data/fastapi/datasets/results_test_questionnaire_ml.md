# Résultat attendu

## Prédiction Machine Learning

| Élément | Valeur attendue |
|----------|-----------------|
| Classe prédite | haut_risque |
| Libellé | Système potentiellement à haut risque |
| Niveau de confiance | ≈ 80 à 95 % (variable selon le modèle entraîné) |

## Pourquoi ?

Le système :

- intervient dans le recrutement ;
- traite des données personnelles ;
- assiste une décision ayant un impact sur une personne.

Ces caractéristiques correspondent à un cas de **Haut Risque** au sens de l'AI Act.

## Rapport PRUDENCIA attendu

Le rapport doit contenir :

### Classification

```
Système potentiellement à haut risque
```

### Obligations

- Gestion des risques
- Documentation technique
- Supervision humaine
- Journalisation
- Évaluation des biais
- Robustesse
- Cybersécurité

### Recommandations

- Validation juridique
- Documentation du système
- Contrôle des biais
- Tests de robustesse
- Mise en place d'une supervision humaine

### Conclusion

Le système peut relever de la catégorie **Haut Risque** de l'AI Act.
Une analyse juridique détaillée reste nécessaire avant toute mise en production.

## Historique attendu

Une nouvelle ligne doit apparaître dans :

```
Historique des analyses
```

avec :

| Champ | Valeur attendue |
|--------|-----------------|
| Type | Questionnaire |
| Projet | RecrutIA |
| Modèle | Random Forest |
| Statut | completed |
| Historique | enregistré |
| Rapport | généré |
