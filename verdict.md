# Verdict — Modèle de scoring Pyrenex Crédit v2

> Document destiné à Sophie Léger (Lead Data, Pyrenex Crédit).
> 1 page max.

## Contexte

(En une phrase : pourquoi ce travail, qu'est-ce qui était attendu.)

## Démarche

(En 2-3 phrases : dataset utilisé, split, nombre de configurations testées,
critères d'évaluation.)

## Verdict chiffré

| Métrique | Baseline 2017 (Pyrenex-risk-v1) | Modèle retenu (v2) | Variation |
|---|---|---|---|
| F1 macro (holdout) | … | … | … |
| F1 défaut | … | … | … |
| ROC-AUC | … | … | … |
| Recall défaut | … | … | … |

**Configuration retenue** : (rappel des hyperparamètres principaux)

## Trade-off explicité au métier

(2-3 phrases : qu'est-ce que le client gagne ? qu'est-ce qu'il perd ?
Par exemple : *« le rappel défaut passe de 14% à 61% — soit 4× plus de
mauvais payeurs détectés — au prix d'une précision défaut qui passe de
38% à 41%. En clair : pour rattraper plus de défauts, le modèle déclenche
davantage de fausses alertes. »*)

## Précautions avant mise en production

- Vérifier que le **schéma d'entrée** en production correspond exactement
  au schéma d'entraînement (cf. `pyrenex_risk_v2.json` → `feature_columns`)
- Re-évaluer le **seuil de décision** (0.5 par défaut) avec l'équipe
  métier — un seuil 0.3 peut être plus adapté selon l'appétence au risque
- Mettre en place un **monitoring** dès le déploiement (cf. M5/M6)
- Surveiller les **variables sensibles** identifiées (FICO, état US,
  revenu) — risque de disparate impact à auditer (M2/M7)

## Recommandation

✅ **Remplacer Pyrenex-risk-v1** par v2 *OU* ⛔ **Ne pas remplacer** —
choisis et justifie en une phrase.

---

*Signé : <prenom> <nom>, FastIA, le YYYY-MM-DD*
