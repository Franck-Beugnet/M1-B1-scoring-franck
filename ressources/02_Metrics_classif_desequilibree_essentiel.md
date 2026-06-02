# Métriques de classification déséquilibrée — Mini-cours

> Brief associé : M1-B1
> Durée de lecture + pratique : ~30 min
> Pré-requis : split réalisé, modèle entraîné capable de `predict` + `predict_proba`.

## Pourquoi cette techno ?

Sur un dataset **déséquilibré** (Lending Club : ~80% remboursés, ~20% en
défaut), la métrique par défaut **accuracy** ment :

> Un modèle qui répond toujours `0` (remboursé) atteint **80% d'accuracy**…
> et **laisse passer 100% des mauvais payeurs**. Côté métier, c'est une
> catastrophe — côté score, ça passe inaperçu.

Il faut donc choisir des métriques qui **rendent visible** le comportement
sur la classe minoritaire (la classe la plus intéressante en pratique :
détection de défaut, fraude, anomalie, maladie…).

**Alternatives à connaître selon le métier :**

| Métrique | Quand l'utiliser ? |
|---|---|
| **Accuracy** | Classes équilibrées **uniquement** (rare en métier). Ne pas reporter seule. |
| **F1 macro** | Notre standard : moyenne des F1 par classe. Pénalise la classe minoritaire mal détectée. |
| **F1 binary (positive)** | Si on cible une classe précise (défaut, fraude). |
| **ROC-AUC** | Mesure la capacité à classer les exemples. Indépendant du seuil. |
| **Precision-Recall AUC** | Préférable à ROC-AUC en très fort déséquilibre (< 5%). |
| **Recall (classe minoritaire)** | Si **rater un défaut coûte cher** (santé, fraude). |
| **Precision (classe minoritaire)** | Si **fausse alerte coûte cher** (filtre anti-spam). |

Sur M1-B1, on rapporte **F1 macro + ROC-AUC + matrice de confusion** au
minimum. Le client (Pyrenex Crédit) regardera surtout **recall sur les
défauts** (rater un mauvais payeur = perte sèche).

## Concepts clés

- **Matrice de confusion** : tableau 2×2 (binaire) qui croise réel vs prédit.
  Les vraies métriques en dérivent.

  ```text
                  Prédit 0          Prédit 1
  Réel 0     │     TN         │      FP        │
  Réel 1     │     FN         │      TP        │
  ```

- **Precision** = TP / (TP + FP) — *« quand le modèle dit "défaut", il a
  raison X% du temps »*.
- **Recall** = TP / (TP + FN) — *« le modèle détecte X% des vrais défauts »*.
- **F1** = 2 × (Precision × Recall) / (Precision + Recall) — moyenne
  harmonique. Punit fort si l'un des deux est bas.
- **F1 macro** = moyenne **non pondérée** des F1 par classe — donne autant
  de poids à la minorité qu'à la majorité.
- **ROC-AUC** : aire sous la courbe ROC. Mesure la **capacité du modèle à
  classer les observations positives devant les négatives**, indépendamment
  du seuil choisi. 0.5 = aléatoire (le modèle ne fait pas mieux qu'un tirage
  au sort), 1.0 = parfait. ⚠️ ROC-AUC mesure **l'ordre des scores**, pas la
  **calibration** des probabilités — un modèle peut avoir un ROC-AUC excellent
  et produire des probabilités peu fiables (sujet à part, abordé en M2/M6).
- **`classification_report(y_true, y_pred)`** : génère precision, recall,
  f1 et **support** par classe — à coller dans le notebook.
  - **Support** = nombre **réel** d'observations présentes pour chaque classe
    dans le jeu évalué. C'est le dénominateur des proportions et il révèle
    immédiatement le degré de déséquilibre (ex. `support=400` pour la classe
    *défaut* contre `support=1600` pour *remboursé* → ratio 1:4).

### Comment lire un classification_report ligne par ligne

Sortie typique sur Pyrenex Crédit (chiffres illustratifs) :

```text
              precision    recall  f1-score   support

   Remboursé       0.85      0.92      0.88      1600   ← classe majoritaire
      Défaut       0.62      0.45      0.52       400   ← classe minoritaire (celle qui compte)

    accuracy                           0.82      2000   ← piège : 82% mais on rate >50% des défauts
   macro avg       0.74      0.69      0.70      2000   ← moyenne NON pondérée → représentative
weighted avg       0.80      0.82      0.81      2000   ← moyenne pondérée par support → écrasée par la majorité
```

**Règles de lecture** :

1. **Lis d'abord la ligne de la classe minoritaire** (ici *Défaut*) — c'est elle qui décide si le modèle est exploitable.
2. **Compare precision et recall sur cette ligne** : precision 0.62 = quand le modèle dit « défaut », il a raison 62% du temps ; recall 0.45 = il ne détecte que 45% des vrais défauts.
3. **Ignore `accuracy`** en classification déséquilibrée — c'est le piège.
4. **`macro avg` > `weighted avg` en pertinence métier** : macro donne autant de poids à chaque classe, weighted écrase la minorité.
5. **`support`** te dit le degré de déséquilibre — si ratio > 1:5, ROC-AUC peut mentir, regarde aussi PR-AUC.

### Le piège du seuil par défaut (0.5)

`model.predict()` applique implicitement un seuil de 0.5 sur la probabilité.
**Ce seuil n'est pas optimal** en déséquilibre. Pour Pyrenex, baisser le
seuil à 0.3 peut faire chuter la précision mais explose le recall sur les
défauts — souvent ce que le métier veut. **À discuter en restitution.**

> ⚠️ **Anti-magie ROC-AUC** : un ROC-AUC élevé (0.85, 0.90) **ne garantit
> pas** un modèle exploitable. Il dit seulement que le modèle sait *ordonner*
> les observations. Si tu utilises un mauvais seuil de décision derrière, le
> modèle peut être totalement inutile en pratique (tout en classe 0, tout en
> classe 1, alertes ingérables…). La chaîne complète à valider est toujours :
> **métrique globale → distribution des probabilités → seuil de décision →
> impact métier**. ROC-AUC seul ne suffit jamais à conclure.

## Exemple minimal qui tourne

```python
# example_metrics.py — versions testées : python 3.11+, scikit-learn 1.5+
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import numpy as np


def evaluate_classifier(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probabilité classe 1

    return {
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_default": f1_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, target_names=["Remboursé", "Défaut"]),
    }


# Exemple d'usage
metrics = evaluate_classifier(model, X_test, y_test)
print(f"F1 macro: {metrics['f1_macro']:.3f}")
print(f"F1 défaut: {metrics['f1_default']:.3f}")
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
print("\nMatrice de confusion :")
print(np.array(metrics["confusion"]))
print("\n" + metrics["report"])
```

## Exercice guidé

À partir d'un modèle entraîné sur Lending Club :

1. Calcule la **matrice de confusion** sur `X_test`. Combien de vrais
   défauts as-tu raté (FN) ?
2. Calcule **F1 macro et F1 binaire (classe défaut)** : lequel est le
   plus bas ? Pourquoi ?
3. Calcule **ROC-AUC**. Compare à 0.5 (aléatoire) — combien de points
   au-dessus ?
4. **Joue avec le seuil** : `(model.predict_proba(X_test)[:, 1] > 0.3).astype(int)`.
   Recompute la matrice de confusion. Qu'est-ce qui bouge ?
5. Trace la **distribution des probabilités** (`sns.histplot(y_proba, bins=50)`).
   Y a-t-il un pic « clair » qui justifie un seuil naturel ?

**Solution attendue (point 2)** : F1 binaire (défaut) est plus bas que F1
macro. Pourquoi : F1 macro est la **moyenne non pondérée** des F1 *par classe*
— il agrège donc la classe défaut (plus dure, F1 plus bas) **et** la classe
remboursé (généralement plus facile à prédire, F1 plus haut). Cette moyenne
reste **meilleure** que le seul F1 défaut, mais sans masquer la performance
de la minorité comme le ferait l'accuracy ou le F1 *weighted*.

## Pièges fréquents

| Piège | Conséquence |
|---|---|
| Rapporter uniquement l'accuracy | Verdict faux, modèle « tout en 0 » passe pour bon |
| Confondre `predict` et `predict_proba` | ROC-AUC calculé sur classes au lieu de probas, valeur incorrecte |
| Oublier `pos_label=1` ou `average=...` en classif déséquilibrée | F1 calculé sur la classe majoritaire, sans pertinence métier |
| Calculer toutes les métriques sur le train set | Évaluation faussement optimiste (overfit invisible) |
| Reporter ROC-AUC en déséquilibre extrême (< 5%) | ROC-AUC peut rester haut alors que le modèle est inutile en pratique (préférer PR-AUC) |
| Choisir le seuil pour optimiser un score test | Surapprentissage du seuil — réserver le holdout |

### Symptôme → cause probable

| Symptôme | Cause probable |
|---|---|
| Accuracy 0.80, F1 défaut 0.00 | Modèle qui prédit toujours la classe majoritaire — essayer `class_weight='balanced'` |
| ROC-AUC = 0.5 exact | Modèle équivalent au tirage aléatoire — entraînement raté ou cible inversée |
| ROC-AUC élevé (0.85) mais F1 bas (0.30) | Seuil 0.5 mal calé — distribution des probas concentrée d'un côté |
| `ValueError: pos_label=1 is not a valid label` | Cible mal encodée — vérifier `y.unique()` |
| F1 macro `nan` ou warning | Une classe n'apparaît pas dans `y_pred` (modèle qui ne prédit qu'une classe) |

## Pour aller plus loin

- Doc officielle : [scikit-learn — Classification report](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report)
- Article : [Why accuracy is not enough — towards data science](https://towardsdatascience.com/the-5-classification-evaluation-metrics-every-data-scientist-must-know-aa97784ff226)
- Référence : [Davis & Goadrich, 2006 — PR vs ROC](https://www.cs.ru.nl/~tomh/onderwijs/dm/dm_files/roc_auc.pdf)
- Aurélien Géron, *ML avec scikit-Learn* ch. 3 — *Classification*.

## Vérification (checklist apprenant)

- [ ] Je peux expliquer en 1 phrase pourquoi l'accuracy ment sur Lending Club
- [ ] Je sais lire une matrice de confusion sans me tromper de cellule
- [ ] J'ai calculé **F1 macro, F1 défaut, ROC-AUC** dans mon notebook
- [ ] J'ai testé un seuil alternatif (0.3 par ex.) et observé l'effet
- [ ] J'ai rédigé une phrase de verdict métier (« le client doit accepter X% de fausses alertes pour rattraper Y% de défauts en plus »)