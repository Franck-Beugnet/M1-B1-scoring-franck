# Pandas + scikit-learn split — Mini-cours

> Brief associé : M1-B1
> Durée de lecture + pratique : ~30 min
> Pré-requis : Python 3.11+, pandas, scikit-learn, environnement virtuel actif.

> 🧭 **Ce qu'on attend réellement de toi en M1-B1**
>
> Le but n'est **pas** d'obtenir « le meilleur modèle du monde » ni de battre
> Kaggle. Le but est de montrer que tu sais :
>
> - **entraîner proprement** (split stratifié, reproductibilité, pas de fuite) ;
> - **comparer honnêtement** (mêmes données d'évaluation, mêmes métriques) ;
> - **justifier un choix** (hyperparams, métrique de référence, verdict
>   argumenté chiffré) ;
> - **rendre le travail reproductible** (`random_state=42`, traçage des runs,
>   modèle persisté avec métadonnées).
>
> Un score moyen mais une démarche rigoureuse > un score brillant non
> reproductible. C'est ce cadrage que Sophie Léger (Pyrenex) attend, et
> c'est aussi celui de la certif CISIA.

## Pourquoi cette techno ?

Avant tout entraînement, deux gestes structurent toute la démarche ML :

1. **Charger et explorer** un dataset proprement avec pandas (sans
   pré-juger des features qu'on va utiliser).
2. **Séparer** les données en jeux d'entraînement et de test, de façon
   **stratifiée** et **reproductible**, pour pouvoir mesurer la
   performance réelle du modèle.

Ces deux gestes paraissent triviaux mais c'est là que se nichent **80%
des erreurs de débutant·e** : fuite de données, déséquilibre masqué,
non-reproductibilité, mauvaise interprétation des modalités.

**Alternatives à connaître :**

| Approche | Quand l'utiliser ? |
|---|---|
| `pd.read_csv` + `train_test_split` | Standard pour datasets < 10 M lignes. Notre cas. |
| `polars` + `train_test_split` | Si volume très lourd (lazy evaluation). Cas avancés. |
| `Dataset` HuggingFace | Pour du texte/image avec datasets natifs HF. Pas notre cas. |
| Split temporel custom | Pour des séries temporelles ou un dataset avec date d'origination (M6 — concept drift). Pas notre cas en M1. |

## Concepts clés

- **`pd.read_csv(path, dtype=...)`** : charger un CSV en spécifiant les types
  attendus quand possible — évite les conversions silencieuses (ex. `int` →
  `float` à cause d'un NaN).

  ```python
  pd.read_csv(path, dtype={"age": "Int64", "code_postal": "category"})
  ```

  ⚠️ Note importante : **`Int64`** (avec un **I majuscule**) est le type
  entier *nullable* de pandas — il **accepte les `NaN`**, contrairement à
  `int64` natif qui force une conversion silencieuse vers `float64` dès qu'une
  valeur manque. Utilise `Int64` pour les colonnes entières susceptibles
  d'avoir des trous (âge, montants, scores…) — ça t'évite des bugs
  monstrueux plus tard quand tu compareras tes features à celles attendues
  par le modèle.
- **`df.info()` et `df.describe(include="all")`** : premier aperçu — types,
  manquants, distributions.
- **Variables cible vs features** : convention `y` pour la cible, `X` pour
  les features. Ne pas mélanger.
- **`train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`** :
  - **`stratify=y`** : indispensable en classification déséquilibrée. Garantit
    que les proportions de classes sont les mêmes dans `train` et `test`.
  - **`random_state=42`** : fige le tirage aléatoire → reproductibilité.
- **Holdout** : un **deuxième** jeu de données, **jamais touché** pendant
  l'entraînement et la sélection de modèle. Sert uniquement à l'évaluation
  finale. En M1-B1, c'est `lending_club_holdout.csv` fourni séparément.

### Différence `test_split` vs `holdout` — souvent confondu

```text
lending_club_train.csv         ← tu travailles dessus toute la journée
       │
       ├── train_test_split(stratify=y)
       │       ├── X_train, y_train   ← pour fit()
       │       └── X_test, y_test     ← pour mesurer pendant la sélection
       │
lending_club_holdout.csv       ← tu y touches UNE SEULE FOIS, à la toute fin
       │                         (pour le verdict)
       └── X_holdout, y_holdout
```

Si tu utilises `holdout` pour choisir tes hyperparamètres → **fuite de
données**, ton verdict est faux.

## Exemple minimal qui tourne

```python
# example_split.py — versions testées : python 3.11+, pandas 2.2+, scikit-learn 1.5+
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_lending_club(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df["loan_status"].map({"Fully Paid": 0, "Charged Off": 1})
    X = df.drop(columns=["loan_status"])
    return X, y


X, y = load_lending_club(Path("data/lending_club_train.csv"))
print(f"Dataset shape: {X.shape}, target balance: {y.value_counts(normalize=True).round(3).to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train shape: {X_train.shape}, test shape: {X_test.shape}")
print(f"Train balance: {y_train.value_counts(normalize=True).round(3).to_dict()}")
print(f"Test balance:  {y_test.value_counts(normalize=True).round(3).to_dict()}")
```

Lance-le : les proportions de classes doivent être **identiques** entre `train`
et `test` à 2 décimales près (effet `stratify`).

## Exercice guidé

À partir de l'exemple ci-dessus :

1. Charge `lending_club_train.csv` et affiche les **5 premières lignes**.
2. Affiche le **type** de chaque colonne avec `df.dtypes`. Note la colonne
   qui te surprend le plus.
3. Affiche le **taux de manquants** par colonne (`df.isna().mean().sort_values(ascending=False).head(10)`).
4. **Sans stratify** (`stratify=None`) puis **avec** : compare les distributions
   de `y_train` et `y_test`. Que constates-tu ?
5. Lance le split **2 fois de suite avec `random_state=42`** : les shapes
   et le contenu de `X_train.iloc[0]` doivent être identiques.

**Solution attendue (point 4)** : sans stratify, les distributions sont
**proches mais pas identiques** (variance liée à l'échantillonnage). Avec
stratify, elles sont identiques à 0.001 près. La différence est
particulièrement visible si le déséquilibre est marqué (Lending Club : ~80/20).

## Pièges fréquents

> 💡 **Petite nuance préalable sur EDA et fuite de données.** Une **EDA
> initiale** (regarder les `dtypes`, le volume du dataset, le taux global
> de manquants, les modalités d'une variable catégorielle) peut se faire
> sur le **dataset complet** — pas de drame. Ce qui est dangereux, c'est de
> **calibrer une transformation** (imputation par la médiane, normalisation,
> encodage, sélection de features…) en utilisant des statistiques calculées
> sur le dataset complet. La règle d'or : **tout ce qui « apprend » des
> paramètres se `fit` sur train uniquement**, puis se `transform` sur test
> et holdout.

| Piège | Conséquence |
|---|---|
| Oublier `stratify=y` en classif déséquilibrée | Métriques de test instables, biais d'évaluation |
| `random_state=42` à un seul endroit | Reproductibilité partielle. Mettre **partout** où il y a de l'aléa |
| Utiliser `df = pd.read_csv(...)` puis `df.drop(...)` sans `inplace=False` | Mutations silencieuses entre cellules |
| Calibrer des transformations (imputer, scaler, encoder) sur le dataset complet avant split | Fuite de données — toujours `fit` sur train, puis `transform` sur test/holdout |
| Splitter **après** avoir transformé | Même piège : la transformation a « vu » le test |
| Confondre `test_split` et `holdout` | Verdict final faussé |
| Charger un CSV sans `dtype=...` quand `int` contient des NaN | Conversion silencieuse `int → float64`, casse `category` |

### Symptôme → cause probable

| Symptôme | Cause probable |
|---|---|
| `y_train.value_counts(normalize=True)` ≠ `y_test.value_counts(normalize=True)` | Oubli de `stratify=y` |
| Résultats différents à chaque exécution | `random_state` absent quelque part (split, modèle, sampling) |
| `ValueError: y contains previously unseen labels` | Modalité catégorielle présente en test, absente en train (utiliser `OneHotEncoder(handle_unknown="ignore")` ou stratify) |
| Score test irréaliste (> 0.99) | Fuite de données : variable corrélée à la cible dans `X` |
| `MemoryError` à l'EDA | CSV trop gros, charger en `chunks` ou passer en Parquet (M2) |

> 🔜 **Cap vers M2** — En M1, tes transformations restent **manuelles** dans
> le notebook : tu écris à la main un `fit_transform` sur train puis un
> `transform` sur test. C'est volontairement explicite pour que tu voies
> **où** ça se joue. En **M2**, tu automatiseras cette chaîne avec
> `sklearn.pipeline.Pipeline` (et `ColumnTransformer`), qui enferment la
> séparation train/test par construction et rendent la fuite de données
> **structurellement impossible**. Garde ça en tête — ce que tu fais à la
> main aujourd'hui prépare le terrain à un outillage plus carré dans 7 jours.

## Pour aller plus loin

- Doc officielle : [scikit-learn — train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- Tutoriel approfondi : [Practical pandas data exploration](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html)
- Article de référence : [Reproducibility in machine learning, Joelle Pineau](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)
- Aurélien Géron, *ML avec scikit-Learn* (3ᵉ éd.), ch. 2 — *« End-to-end ML project »*.

## Vérification (checklist apprenant)

- [ ] J'ai fait tourner l'exemple minimal et compris ce que `stratify=y` change
- [ ] Je peux expliquer en 2 phrases la différence entre `test_split` et `holdout`
- [ ] Je sais où `random_state=42` doit apparaître dans mon code
- [ ] J'ai identifié au moins 1 colonne du Lending Club dont le `dtype` me surprend
- [ ] J'ai fait l'exercice guidé et les 5 points sont validés