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

- **Date** : 2026-06-02 12:28 UTC
- **Modèle** : RandomForestClassifier (sklearn 1.5.1)
- **Dataset** : lending_club_train.csv (sha256 d2da093bee40024b196e73a0d2d763193782f947e3d60552a3d7bbad0bd944e3), n=24000
- **Split** : test_size=0.2, stratify=y, random_state=42
- **Hyperparamètres** : tous par défaut, `n_jobs=-1`, `random_state=42`
- **Pré-traitement** : OneHotEncoder + StandardScaler (Pipeline scikit-learn)
- **Métriques (test interne)** :
  - F1 macro : 0.5131
  - F1 défaut : 0.1269
  - ROC-AUC : 0.7170
  - Recall défaut : 0.0713
  - Precision défaut : 0.5727
  - Balanced accuracy : 0.5297
  - Matrice de confusion : [[3870, 47], [820, 63]]
  - Commentaire matrice : modèle très conservateur, très peu de faux positifs (47) mais énormément de défauts ratés (820 FN) ; il protège la précision au détriment du rappel défaut.
- **Temps d'entraînement** : ~10s
- **Verdict** : ⛔ Écarté — modèle trop conservateur, recall défaut très faible (7.13%), trop de défauts manqués.

---

## exp_002 — RF balanced

- **Date** : 2026-06-02 12:35 UTC
- **Modèle** : RandomForestClassifier (sklearn 1.5.1)
- **Dataset** : lending_club_train.csv (sha256 d2da093bee40024b196e73a0d2d763193782f947e3d60552a3d7bbad0bd944e3), n=24000
- **Split** : test_size=0.2, stratify=y, random_state=42
- **Hyperparamètres** : `n_estimators=200`, `max_depth=10`, `min_samples_leaf=10`, `class_weight='balanced'`, `n_jobs=-1`, `random_state=42`
- **Pré-traitement** : OneHotEncoder + StandardScaler (Pipeline scikit-learn)
- **Métriques (test interne)** :
  - F1 macro : 0.6123 => Moyenne de F1 classe 0 et F1 classe 1 avec poids égaux, performance équilibrée entre les 2 classes
  - F1 défaut : 0.4326 => score global de qualité sur la classe défaut seulement
  - ROC-AUC : 0.7442 => Mesure la capacité de séparation des classes sur tous les seuils (0.5 = aléatoire, 1.0 = parfait)
  - Recall défaut : 0.6308 => vous récupérez 63.08% des vrais défauts
  - Precision défaut : 0.3292 => parmi les dossiers prédits “défaut”, seulement 32.92% sont vraiment en défaut => beaucoup de faux positifs
  - Balanced accuracy : 0.6705
  - Matrice de confusion : [[2782, 1135], [326, 557]]
  - Commentaire matrice : sur 883 vrais défauts, 557 sont bien détectés (TP) et 326 sont ratés (FN) ; en contrepartie, 1135 non-défauts sont classés à tort en défaut (FP).
- **Temps d'entraînement** : non tracé
- **Verdict** : ✅ Retenu (candidat final) — forte hausse du recall défaut (+55.95 points vs exp_001) et gain net sur F1 macro/ROC-AUC, avec un trade-off assumé sur la précision.

---

## exp_003 — Sweep hyperparamètres (balanced + trial_01 à trial_10)

- **Date** : 2026-06-02
- **Modèle** : RandomForestClassifier (sklearn 1.5.1)
- **Dataset** : lending_club_train.csv (sha256 d2da093bee40024b196e73a0d2d763193782f947e3d60552a3d7bbad0bd944e3), n=24000
- **Split** : test_size=0.2, stratify=y, random_state=42
- **Protocole** : comparaison de 12 runs (`default`, `balanced`, `trial_01..trial_10`) sur le même split interne

### Résultats clés (classement)

- **Meilleur F1 défaut** : `trial_05` = **0.4362**
- **Meilleur Recall défaut** : `trial_05` = **0.6365**
- **Meilleure Balanced accuracy** : `trial_05` = **0.6737**
- **Meilleur F1 macro** : `trial_06` = **0.6185**
- **Meilleur ROC-AUC** : `trial_07` = **0.7454**
- **Meilleure Precision défaut** : `default` = **0.5727** (mais recall très faible à 0.0713)

### Top 5 par F1 défaut

| Run | F1 défaut | Recall défaut | Precision défaut | F1 macro | ROC-AUC | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| trial_05 | 0.4362 | 0.6365 | 0.3318 | 0.6146 | 0.7440 | 0.6737 |
| trial_01 | 0.4351 | 0.6240 | 0.3339 | 0.6163 | 0.7437 | 0.6717 |
| trial_06 | 0.4343 | 0.6104 | 0.3371 | 0.6185 | 0.7381 | 0.6699 |
| balanced | 0.4326 | 0.6308 | 0.3292 | 0.6123 | 0.7442 | 0.6705 |
| trial_09 | 0.4326 | 0.6308 | 0.3292 | 0.6123 | 0.7442 | 0.6705 |

### Lecture comparative

- Les runs `default`, `trial_03`, `trial_04`, `trial_07`, `trial_08` gardent une précision défaut plus haute mais ratent trop de défauts (recall bas).
- Le groupe `balanced`, `trial_01`, `trial_05`, `trial_06`, `trial_09`, `trial_10` est plus cohérent avec l'objectif métier "ne pas rater les défauts".
- `trial_05` est le meilleur compromis sur les métriques orientées risque (F1 défaut, recall défaut, balanced accuracy), avec un ROC-AUC toujours élevé.

- **Verdict** : ✅ `trial_05` retenu comme meilleur candidat avant holdout.
- **Config retenue** : `n_estimators=300`, `max_depth=8`, `min_samples_leaf=20`, `class_weight='balanced_subsample'`, `max_features='sqrt'`, `random_state=42`, `n_jobs=-1`.

---

## 🏁 Évaluation finale sur holdout (modèle retenu)

> **À remplir une seule fois**, à la tâche 5 du brief, **après** avoir choisi
> ton modèle retenu parmi les `exp_NNN` ci-dessus. Le holdout n'est consulté
> qu'ici.

- **Date** : 2026-06-02
- **Expérience retenue** : exp_003 (trial_05)
- **Modèle persisté** : `models/pyrenex_risk_v2_trial_05.joblib`
- **Données holdout** : `data/lending_club_holdout.csv` (sha256 b5ca9339a6ddc4303b73e7b7529329de44e1bcfe72371639eb3d4a8a6209fc77, n=6000)
- **Métriques** :
  - F1 macro : 0.6108
  - F1 défaut : 0.4347
  - ROC-AUC : 0.7348
  - Recall défaut : 0.6473
  - Precision défaut : 0.3272
  - Accuracy : 0.6905
- **Matrice de confusion** :

|  | Pred Fully Paid | Pred Charged Off |
|---|---|---|
| **Vrai Fully Paid** | 3429 | 1468 |
| **Vrai Charged Off** | 389 | 714 |

- **Comparaison baseline 2017** : F1 macro supérieur (0.6108 vs 0.5018), recall défaut massivement amélioré (0.6473 vs 0.0500), F1 défaut en forte hausse (0.4347 vs 0.0859 calculé depuis la matrice 2017), ROC-AUC légèrement meilleur (0.7348 vs 0.7296). En contrepartie, accuracy plus faible (0.6905 vs 0.8492) et précision défaut plus basse (0.3272 vs 0.6100), ce qui reflète un modèle volontairement moins conservateur.