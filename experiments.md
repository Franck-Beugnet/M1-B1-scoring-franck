# Expériences — M1-B1 Pyrenex Crédit (Lending Club)

> Trace tes runs au fur et à mesure. Format imposé : un bloc par run, avec
> date, modèle, hyperparams, métriques **test interne uniquement**, verdict.
> Commit à chaque run final (pas à chaque essai jetable).
>
> ⚠️ **Règle d'or — comparabilité.** Le holdout **n'apparaît jamais** dans les
> blocs `exp_NNN`. Il sort **une seule fois**, pour le modèle retenu, dans
> la section finale en bas de fichier. Cf. mini-cours 04.

---

## exp_001 — RF par défaut

- **Date** : YYYY-MM-DD HH:MM
- **Modèle** : RandomForestClassifier (sklearn X.Y.Z)
- **Dataset** : lending_club_train.csv (sha256 …), n=…
- **Split** : test_size=0.2, stratify=y, random_state=42
- **Hyperparamètres** : tous par défaut, `n_jobs=-1`, `random_state=42`
- **Pré-traitement** : OneHotEncoder + StandardScaler (Pipeline scikit-learn)
- **Métriques (test interne)** :
  - F1 macro : …
  - F1 défaut : …
  - ROC-AUC : …
  - Recall défaut : …
- **Temps d'entraînement** : … s
- **Verdict** : …

---

## exp_002 — RF balanced (TODO — remplis avec ta config)

- **Date** :
- **Modèle** :
- **Hyperparamètres** :
- **Pré-traitement** :
- **Métriques (test interne)** :
- **Temps d'entraînement** :
- **Verdict** :

---

## exp_003 — (TODO — ta variante ou mission étoile ⭐ si tu y vas)

- ...

---

## 🏁 Évaluation finale sur holdout (modèle retenu)

> **À remplir une seule fois**, à la tâche 5 du brief, **après** avoir choisi
> ton modèle retenu parmi les `exp_NNN` ci-dessus. Le holdout n'est consulté
> qu'ici.

- **Date** : YYYY-MM-DD HH:MM
- **Expérience retenue** : exp_NNN
- **Modèle persisté** : `models/pyrenex_risk_v2.joblib`
- **Données holdout** : `data/lending_club_holdout.csv` (sha256 …, n=…)
- **Métriques** :
  - F1 macro : …
  - F1 défaut : …
  - ROC-AUC : …
  - Recall défaut : …
- **Matrice de confusion** :

|  | Pred Fully Paid | Pred Charged Off |
|---|---|---|
| **Vrai Fully Paid** | … | … |
| **Vrai Charged Off** | … | … |

- **Comparaison baseline 2017** : (cf. `verdict.md`)