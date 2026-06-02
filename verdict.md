# Verdict — Modèle de scoring Pyrenex Crédit v2

> Document destiné à Sophie Léger (Lead Data, Pyrenex Crédit).
> 1 page max.

## Contexte

Pyrenex Crédit souhaitait challenger la baseline historique `pyrenex_risk_v1` (2017) sur le nouveau dataset Lending Club 2025 et décider d'un éventuel remplacement par un modèle v2 plus efficace sur la détection des défauts.

## Démarche

Le travail a été réalisé sur `lending_club_train.csv` (24k lignes), avec un split interne `test_size=0.2`, `stratify=y`, `random_state=42`, puis une évaluation finale unique sur `lending_club_holdout.csv` (6k lignes). 
Nous avons comparé 12 configurations RandomForest (`default`, `balanced`, `trial_01..trial_10`) avec pipeline reproductible (imputation + standardisation + one-hot). 
Le modèle retenu avant holdout est `trial_05`, sélectionné principalement sur les métriques orientées risque défaut (`F1 défaut`, `recall défaut`, `balanced accuracy`).

## Verdict chiffré

| Métrique | Baseline 2017 (Pyrenex-risk-v1) | Modèle retenu (v2) | Variation |
|---|---|---|---|
| F1 macro (holdout) | 0.5018 | 0.6108 | +0.1090 |
| F1 défaut | 0.0859 | 0.4347 | +0.3488 |
| ROC-AUC | 0.7296 | 0.7348 | +0.0052 |
| Recall défaut | 0.0500 | 0.6473 | +0.5973 |

**Configuration retenue** : `RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20, class_weight='balanced_subsample', max_features='sqrt', random_state=42, n_jobs=-1)`.

## Trade-off explicité au métier

Le gain principal est la détection des défauts: le rappel défaut passe de 5.0% à 64.7%, soit environ 13x plus de défauts identifiés. 
Le coût de ce gain est une hausse des faux positifs: la précision défaut baisse (0.6100 → 0.3272) et l'accuracy globale baisse (0.8492 → 0.6905). 
En clair, le v2 est moins conservateur: il attrape beaucoup plus de dossiers risqués, mais au prix d'un volume plus élevé d'alertes à traiter métier.

## Précautions avant mise en production

- Vérifier que le **schéma d'entrée** en production correspond exactement
  au schéma d'entraînement (cf. `pyrenex_risk_v2.json` → `feature_columns`)
- Re-évaluer le **seuil de décision** (0.5 par défaut) avec l'équipe
  métier — un seuil 0.3 peut être plus adapté selon l'appétence au risque
- Mettre en place un **monitoring** dès le déploiement (cf. M5/M6)
- Surveiller les **variables sensibles** identifiées (FICO, état US,
  revenu) — risque de disparate impact à auditer (M2/M7)

## Recommandation

✅ **Remplacer Pyrenex-risk-v1** par v2, car l'objectif risque (ne pas laisser passer les défauts) est nettement mieux atteint, avec une amélioration massive du rappel et du F1 défaut, sous réserve d'aligner le seuil de décision et la capacité opérationnelle à gérer davantage de faux positifs.

---

*Signé : Franck BEUGNET, le 2026-06-02*
