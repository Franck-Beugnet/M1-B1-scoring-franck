# pyrenex_risk_v1 — Baseline historique Pyrenex Crédit

> Modèle de scoring du risque de défaut, entraîné en 2017, encore en
> production. Référence de comparaison pour le brief **M1-B1** :
> remplaçons-nous ce modèle ou pas ?

## Contexte

En 2017, Pyrenex Crédit a déployé un modèle RandomForest pour scorer les
demandes de prêt à la conso. Le modèle tourne en production depuis. La
démarche utilisée à l'époque est consignée dans `src/train_v1.py` — code
historique, conservé tel quel.

Aujourd'hui (M1-B1) : on dispose d'un nouveau dataset (~30 k lignes) et
on veut savoir s'il faut **réentraîner**, **améliorer la démarche**, ou
**garder l'existant**.

## Métriques rapportées (test split 20% de 12k)

| Métrique | Valeur | Note |
|---|---|---|
| **Accuracy** | **0.8492** | C'est ce qui a été rapporté à l'époque. |
| F1 macro | 0.5018 | Recalculé a posteriori — non rapporté en 2017. |
| ROC-AUC | 0.7296 | Recalculé a posteriori. |
| Precision (Charged Off) | 0.61 | Sur 28 prédits défaut, 17 le sont vraiment. |
| **Recall (Charged Off)** | **0.05** | ⚠️ Sur 368 vrais défauts, on en détecte 17 (~5%). |

**Matrice de confusion** (lignes = vérité, colonnes = prédiction) :

|  | Pred Fully Paid | Pred Charged Off |
|---|---|---|
| **Vrai Fully Paid** | 2021 | 11 |
| **Vrai Charged Off** | 351 | 17 |

> 💡 Lecture critique (à mener par l'apprenant) : 95% des défauts ne sont
> pas détectés. Le modèle est très conservateur — il dit presque toujours
> « pas de défaut ». Le coût métier de ces faux négatifs est colossal.

## Démarche datée — angles morts assumés

Le code 2017 (`src/train_v1.py`) contient des choix qu'on **ne ferait plus**
aujourd'hui :

1. **Pas de `stratify`** dans le `train_test_split` → la classe minoritaire
   peut être sur/sous-représentée dans le test.
2. **Pas de `class_weight`** → le modèle a peu d'incitation à apprendre la
   classe minoritaire.
3. **Hyperparamètres par défaut** (`n_estimators=100`, `max_depth=None`) →
   aucun tuning.
4. **Preprocessing fit sur tout le dataset** avant le split → fuite légère.
5. **Métrique rapportée = accuracy** → trompeuse en classe déséquilibrée.

Ces angles morts sont **l'objet du brief M1-B1** : les détecter, les
documenter, proposer un v2 qui les corrige (ou justifier de ne pas le
faire).

## Schéma du dataset (commun avec le 2025)

15 colonnes — cf. [`data/generate_2017.py`](data/generate_2017.py) pour les
définitions précises. Cible : `loan_status` ∈ {`Fully Paid`, `Charged Off`}.

Différences 2017 vs 2025 (à observer en EDA côté apprenant) :
- 2017 : 12 k lignes, ~15% défauts, pas de NaN
- 2025 : 30 k lignes, ~18% défauts, NaN sur `emp_length` (4%) et `revol_util` (1.5%)
- Distribution des grades plus risquée en 2025 (mix moins safe)

## Reproduire l'entraînement baseline

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. (Re)générer le dataset 2017
cd data && python generate_2017.py && cd ..

# 2. (Re)entraîner le modèle baseline
cd src && python train_v1.py
# → produit ../models/pyrenex_risk_v1.joblib + .json
```

## Charger le modèle baseline depuis ton repo M1-B1

```python
import joblib
bundle = joblib.load("path/to/pyrenex_risk_v1.joblib")

# Le bundle contient toutes les briques pour appliquer le pipeline 2017 :
num_imputer = bundle["num_imputer"]
scaler = bundle["scaler"]
cat_imputer = bundle["cat_imputer"]
encoder = bundle["encoder"]
model = bundle["model"]

# Pour scorer un nouveau DataFrame X (mêmes colonnes que 2017) :
X_num = scaler.transform(num_imputer.transform(X[bundle["numeric_features"]]))
X_cat = encoder.transform(cat_imputer.transform(X[bundle["categorical_features"]]))
X_prepared = np.hstack([X_num, X_cat])
y_pred = model.predict(X_prepared)
```

> ⚠️ Tu peux scorer le nouveau dataset 2025 avec ce modèle pour mesurer
> sa **dérive de performance** — c'est un excellent point de comparaison
> avant ton réentraînement. Mais c'est **toi** qui choisis si tu fais ça.

## Métadonnées complètes

Voir [`models/pyrenex_risk_v1.json`](models/pyrenex_risk_v1.json).

## Pour le brief M1-B1

- Ce repo est **en lecture seule** côté apprenant.
- Tu **ne dois pas** retoucher au code 2017 — c'est une référence figée.
- Tu peux **charger** le `.joblib`, le **scorer** sur le 2025, et **comparer**
  aux performances rapportées ci-dessus.
- Ton verdict final (`verdict.md` dans ton repo perso) doit dire : remplace
  ou garde, et pourquoi (chiffres à l'appui).