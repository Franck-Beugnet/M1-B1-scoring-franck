# RandomForest — Hyperparamètres clés — Mini-cours

> Brief associé : M1-B1
> Durée de lecture + pratique : ~25 min
> Pré-requis : modèle compris en haut niveau (qu'est-ce qu'un arbre de décision), split réalisé.

## Pourquoi cette techno ?

**RandomForest** est l'un des meilleurs modèles de référence pour les
données tabulaires :

- **Robuste** par défaut (peu d'ajustements nécessaires)
- **Insensible au scaling** (contrairement à régression logistique / SVM)
- **Gère bien les variables catégorielles une fois encodées**
- **Fournit une mesure d'importance des features** (`feature_importances_`) — voir
  note d'usage plus bas
- **Parallélisable** (`n_jobs=-1`)

En production, on lui préfère souvent **XGBoost** ou **LightGBM** pour
quelques pourcentages de F1, mais **RandomForest reste la baseline** :
rapide à entraîner, bien comprise, et **moins sensible à l'overfit qu'un
arbre de décision unique** — sans pour autant en être immunisée (petits
datasets, profondeur infinie, bruit ou leakage peuvent toujours faire
dérailler la forêt).

> ⚠️ **À propos de `feature_importances_`** — Cette mesure indique
> **l'utilité statistique d'une feature pour le modèle**, autrement dit *à
> quel point elle a aidé à réduire l'impureté lors des splits*. **Ce n'est
> pas une mesure de causalité métier.** Une feature peut être très
> *importante* parce qu'elle est fortement corrélée à la cible, sans en être
> la cause (ex. *montant du prêt* est corrélé au défaut, mais n'en est pas
> la cause — le contexte économique l'est davantage). À expliciter dans le
> `verdict.md` si tu mobilises ces importances pour conseiller le client.

Pour le brief M1-B1, c'est la famille **imposée** par la baseline
`pyrenex-risk-v1` — on reproduit la démarche du modèle 2017.

**Alternatives à connaître :**

| Famille | Quand l'utiliser ? |
|---|---|
| **RandomForest** | Baseline tabulaire par défaut. Notre cas en M1. |
| **GradientBoosting / HistGradientBoosting (scikit-learn)** | Souvent meilleur que RF en performance. Voir mission étoile ⭐. |
| **XGBoost / LightGBM** | Production, compétitions Kaggle. Plus d'hyperparamètres à régler. |
| **Régression logistique** | Simple, interprétable, sensible au scaling. Bonne complémentaire à comparer. |
| **Réseaux de neurones** | Sur tabulaire structuré, rarement justifié vs un boosting. À éviter par défaut. |

## Concepts clés

Les hyperparamètres importants à manipuler en M1-B1 :

- **`n_estimators`** (par défaut 100) — nombre d'arbres dans la forêt.
  Plus d'arbres = plus stable, mais **coût d'entraînement linéaire** :
  doubler `n_estimators` double approximativement le temps d'entraînement
  (et la mémoire occupée par le modèle). **Sweet spot** : 100-300 pour la
  plupart des cas. Au-delà, gains marginaux.
- **`max_depth`** (par défaut `None`, croissance illimitée) — profondeur
  max de chaque arbre. **Sans limite, surapprentissage**. Imposer 8-15
  est une bonne baseline.
- **`min_samples_split`** (par défaut 2) — nombre min d'échantillons pour
  séparer un nœud. Monter à 10-50 si dataset bruité.
- **`min_samples_leaf`** (par défaut 1) — nombre min d'échantillons par
  feuille. Monter à 5-20 lisse les prédictions, évite l'overfit.
- **`max_features`** (par défaut `sqrt(n_features)`) — nombre de features
  considérées à chaque split. Garder par défaut sauf cas particulier.
- **`class_weight`** — **central en déséquilibre**. `'balanced'` ajuste
  automatiquement selon les fréquences. Voir ci-dessous.
- **`n_jobs=-1`** — parallélisation sur tous les cœurs CPU. Toujours
  utiliser en M1+.
- **`random_state=42`** — reproductibilité.

### `class_weight` — le levier déséquilibre

Sur Lending Club, ~20% de défauts. Trois options :

```python
# Option A — par défaut : favorise la classe majoritaire (remboursés)
RandomForestClassifier(random_state=42)

# Option B — balanced : poids inversement proportionnels à la fréquence
RandomForestClassifier(class_weight="balanced", random_state=42)

# Option C — dict explicite : tu choisis (utile si l'asymétrie de coût est non-symétrique)
RandomForestClassifier(class_weight={0: 1, 1: 3}, random_state=42)
```

**Trade-off** : `balanced` augmente le **recall** sur les défauts (on
détecte plus de mauvais payeurs) au prix d'une **chute de précision**
(plus de fausses alertes). C'est souvent ce que le métier veut. À
**discuter explicitement** dans le `verdict.md`.

## Exemple minimal qui tourne

```python
# example_rf.py — versions testées : python 3.11+, scikit-learn 1.5+
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
import time


def train_rf(X_train, y_train, X_test, y_test, params: dict) -> dict:
    model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "model": model,
        "params": params,
        "fit_time_sec": round(fit_time, 2),
        "f1_macro": f1_score(y_test, model.predict(X_test), average="macro"),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


# Jeu A — défaut
result_a = train_rf(X_train, y_train, X_test, y_test, params={})

# Jeu B — tuné déséquilibre
result_b = train_rf(
    X_train, y_train, X_test, y_test,
    params={"n_estimators": 200, "max_depth": 10, "class_weight": "balanced"},
)

for name, r in [("Défaut", result_a), ("Tuné B", result_b)]:
    print(f"{name:8} | F1 macro = {r['f1_macro']:.3f} | ROC-AUC = {r['roc_auc']:.3f} | {r['fit_time_sec']}s")
```

Tu obtiens un mini-benchmark prêt à coller dans ton `experiments.md`.

## Exercice guidé

À partir du squelette M1-B1 :

1. Entraîne un **RandomForest par défaut** sur tes données préparées. Note
   F1 macro, ROC-AUC, et le **rappel sur la classe défaut** (`classification_report`).
2. Entraîne un deuxième modèle avec **`class_weight='balanced'`** + reste
   inchangé. Compare le rappel défaut.
3. Entraîne un troisième modèle avec **`n_estimators=300`, `max_depth=8`,
   `min_samples_leaf=10`, `class_weight='balanced'`**. Compare.
4. Plot `model.feature_importances_` pour ton meilleur modèle (top-15
   features). Que comprends-tu ?
5. **Mission étoile ⭐ optionnelle** : remplace par
   `HistGradientBoostingClassifier(random_state=42)` (laisse les autres
   hyperparams par défaut pour cette première comparaison). Quelle famille
   gagne sur F1 macro et ROC-AUC ?

**Solution attendue (point 2)** : avec `balanced`, le rappel défaut doit
augmenter sensiblement (souvent +10 à +20 points), mais la **précision
défaut** doit baisser. C'est le trade-off déséquilibre.

## Pièges fréquents

| Piège | Conséquence |
|---|---|
| `n_estimators=10` | Forêt trop petite, variance élevée, résultats instables |
| `max_depth=None` sans contrôle de leaf | Surapprentissage sur le train (F1 train ≈ 1.0, F1 test plus bas) |
| Oublier `class_weight` en déséquilibre | F1 défaut catastrophique (0.10-0.30) |
| Oublier `random_state` | Résultats variables, comparaisons impossibles |
| Oublier `n_jobs=-1` | Entraînement 4-8× plus lent |
| Comparer des modèles avec des splits différents | Comparaison invalide — toujours fixer le split |
| Toucher au test set pour ajuster les hyperparamètres | Fuite — utiliser cross-validation sur train, holdout final |

### Symptôme → cause probable

| Symptôme | Cause probable |
|---|---|
| F1 train 0.99, F1 test 0.65 | Surapprentissage — limiter `max_depth`, monter `min_samples_leaf` |
| F1 défaut ≈ 0 quel que soit le réglage | Très fort déséquilibre — essayer `class_weight={0:1, 1:5}` ou rééchantillonner (SMOTE) |
| Entraînement très lent (> 1 min sur 30k lignes) | `n_estimators` trop haut, oubli de `n_jobs=-1`, ou trop de features one-hot |
| Résultats différents à chaque run | `random_state` absent quelque part |
| `feature_importances_` toutes proches | Variables redondantes ou colinéaires — discuter en restitution |

## Pour aller plus loin

- Doc officielle : [scikit-learn — RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- Tutoriel : [scikit-learn — Ensemble methods (Bagging, Forests, Boosting)](https://scikit-learn.org/stable/modules/ensemble.html)
- Référence : Breiman, *Random Forests*, 2001 — l'article fondateur.
- Aurélien Géron, *ML avec scikit-Learn* ch. 7 — *Ensembles*.

## Vérification (checklist apprenant)

- [ ] Je peux nommer les **5 hyperparamètres** RandomForest les plus importants
- [ ] Je comprends le trade-off `class_weight='balanced'` (recall ↑, precision ↓)
- [ ] J'ai entraîné **au moins 2 jeux d'hyperparamètres** et tracé les résultats
- [ ] J'ai inspecté `feature_importances_` et identifié les 3 features les plus prédictives
- [ ] Mon `verdict.md` mentionne explicitement le trade-off rappel/précision pour le client Pyrenex