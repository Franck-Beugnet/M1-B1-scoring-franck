# Traçage d'expérimentations — `experiments.md` (et MLflow en bonus) — Mini-cours

> Brief associé : M1-B1
> Durée de lecture + pratique : ~30 min
> Pré-requis : avoir entraîné au moins 1 modèle.

## Pourquoi cette techno ?

Sans traçage, **tu ne peux pas répondre à la question fondamentale** :
*« Pourquoi je retiens ce modèle plutôt qu'un autre ? »*. Le client
(Pyrenex Crédit) le demande explicitement dans son mail.

Tracer une expérimentation = consigner par run :

- **Quoi** : nom de la config, modèle, dataset, version
- **Comment** : hyperparamètres, transformations, seed
- **Combien** : métriques (F1, ROC-AUC, temps, taille du modèle)
- **Verdict** : retenu / écarté + raison courte

Deux options pour le brief M1-B1 :

| Outil | Quand l'utiliser ? |
|---|---|
| **`experiments.md` versionné** | Notre standard M1-B1. Markdown, lisible, diffé sur GitHub. Aucune dépendance. |
| **MLflow** | Plus puissant (UI web, registry, comparaisons). **Optionnel** en M1, deviendra utile en M5/M6. Pas de tropisme outil — ne pas l'imposer ici. ⚠️ Sans discipline (runs inutiles, doublons, UI bruitée), MLflow se transforme vite en cimetière d'expériences inexploitable. |

Le principe : **mieux vaut un `experiments.md` discipliné qu'un MLflow
non maintenu**. On en reparle en M5 quand le besoin de gouvernance
multi-équipes apparaît.

> 🔑 **Trois règles d'or du traçage** (à garder en tête tout au long du brief)
>
> 1. **Reproductibilité** — *si un run n'est pas reproductible, il ne compte pas.*
> 2. **Décision** — *un run sans verdict (retenu / écarté + pourquoi) est un log inutile.*
> 3. **Comparabilité** — *deux runs ne se comparent que sur même split, même version du dataset, même métrique de référence.* Sinon : pommes vs poires.
>
> **Niveau M1-B1 attendu** : 2 à 5 runs **bien documentés** suffisent. On
> juge sur la **qualité de justification**, pas sur le volume.

## Concepts clés

- **Un run** = un entraînement complet avec une config précise. Toujours
  un **nom** (`exp_001_rf_default`), une **date**, des **métriques**, une
  **décision** (retenu / écarté).
- **Reproductible** = mêmes hyperparams + split + seed → score stable à
  10⁻³ près. Sans ça, le run ne compte pas (règle d'or n°1).
- **Comparable** = deux runs partagent le **même split**, la **même version
  du dataset** (sha256 ou tag git) et la **même métrique de référence**.
  Changer le split entre runs casse la comparaison.
- **Log vs verdict** : la section *Métriques* est **factuelle** (chiffres
  bruts, vérifiables). La section *Verdict* est **interprétative**
  (décision + raison courte). Ne pas mélanger — `experiments.md` n'est
  pas un journal d'opinions, c'est une trace d'audit.
- **Versionnement** : `experiments.md` vit **dans Git** — chaque commit
  raconte une étape de la démarche. Pour le DPO et le métier, c'est une
  preuve.
- **Pas de tableau de 100 colonnes** : 6-10 champs par run. Lisible > exhaustif.

### MLflow — quand ça devient pertinent

À partir du moment où :

- Plusieurs personnes lancent des expériences sur le même projet
- On veut comparer **plus de 10 runs** côte à côte
- On veut **enregistrer les modèles** dans un registry partagé
- On veut une **UI web** pour les non-techniciens

→ Pour Pyrenex M1, c'est **prématuré**. On reverra en M5 (CI/CD + MLOps).

## Exemple minimal qui tourne

### Squelette à copier (avant de remplir avec tes vraies valeurs)

````markdown
# Expériences — M<x>-B<y> <client>

## exp_XXX — <nom court explicite>

- **Date** : YYYY-MM-DD HH:MM
- **Modèle** : <Classe sklearn> (sklearn x.y.z)
- **Dataset** : <fichier> (sha256 <hash court>), n=<nb lignes>
- **Split** : test_size=..., stratify=..., random_state=42
- **Hyperparamètres** : <liste exhaustive et copiable>
- **Pré-traitement** : <transformations + où elles sont `fit`>
- **Métriques (test)** : F1 macro, F1 minoritaire, ROC-AUC, recall minoritaire
- **Métriques (holdout)** : à remplir uniquement à la toute fin
- **Temps d'entraînement** : X.X s
- **Verdict** : ✅ retenu / ⛔ écarté — <raison courte argumentée>
````

> 💡 **Les sections *Métriques* sont factuelles, la section *Verdict* est
> interprétative.** Cette séparation est essentielle : un collègue ou un
> auditeur doit pouvoir relire tes chiffres bruts sans avoir à démêler
> tes conclusions personnelles.

### Option A — `experiments.md` rempli (notre choix)

```markdown
# Expériences — M1-B1 Pyrenex Crédit (Lending Club)

## exp_001 — RF par défaut

- **Date** : 2026-05-19 11:20
- **Modèle** : RandomForestClassifier (sklearn 1.5.1)
- **Dataset** : lending_club_train.csv (sha256 a3f9...), n=24000
- **Split** : test_size=0.2, stratify=y, random_state=42
- **Hyperparamètres** : tous par défaut, `n_jobs=-1`, `random_state=42`
- **Pré-traitement** : OneHotEncoder, StandardScaler (pipeline scikit-learn)
- **Métriques (test interne)** :
  - F1 macro : 0.602
  - F1 défaut : 0.182
  - ROC-AUC : 0.703
  - Recall défaut : 0.142
- **Métriques (holdout)** : F1 macro 0.598, F1 défaut 0.175
- **Temps d'entraînement** : 12.4 s
- **Verdict** : ⛔ écarté — F1 défaut trop bas, modèle qui prédit majoritairement la classe 0.

## exp_002 — RF balanced

- **Date** : 2026-05-19 14:05
- **Modèle** : RandomForestClassifier
- **Hyperparamètres** : `n_estimators=200`, `max_depth=10`, `class_weight='balanced'`,
  `min_samples_leaf=10`, `n_jobs=-1`, `random_state=42`
- **Pré-traitement** : idem exp_001
- **Métriques (test interne)** :
  - F1 macro : 0.706
  - F1 défaut : 0.491
  - ROC-AUC : 0.741
  - Recall défaut : 0.612
- **Métriques (holdout)** : F1 macro 0.701, F1 défaut 0.485
- **Temps d'entraînement** : 38.1 s
- **Verdict** : ✅ retenu — recall défaut multiplié par 4 vs exp_001, F1 macro
  +10 pts, au prix de précision défaut 0.38 → 0.41. Trade-off explicité au client.
```

→ Te suffit. Diff GitHub propre, lisible en 30 secondes.

### Option B — MLflow (bonus, pour les apprenants à l'aise)

```python
# example_mlflow.py — versions testées : mlflow 2.18+
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score


mlflow.set_experiment("pyrenex-risk-v2")  # crée si absent

with mlflow.start_run(run_name="exp_002_rf_balanced"):
    params = {"n_estimators": 200, "max_depth": 10, "class_weight": "balanced"}
    mlflow.log_params(params)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("dataset_sha256", "a3f9...")

    model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    mlflow.log_metric("f1_macro", f1_score(y_test, model.predict(X_test), average="macro"))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))

    mlflow.sklearn.log_model(model, artifact_path="model")
```

Puis `mlflow ui` lance la UI web sur `http://localhost:5000`.

## Exercice guidé

À partir de tes 2 runs (jeux d'hyperparamètres) en M1-B1 :

1. Ouvre `experiments.md` et **copie-colle** le template de l'exemple
   A ci-dessus. Renomme `exp_001` et `exp_002` selon **tes** configs.
2. **Remplis chaque champ avec TES vraies valeurs** (métriques exactes,
   pas approximatives).
3. Pour chaque run, rédige **un verdict d'une ligne** : retenu ou écarté +
   pourquoi.
4. Commit dans Git avec un message du type
   `docs(experiments): add exp_001 and exp_002 with verdicts`.
5. **Mission bonus** : essaie MLflow sur un 3ᵉ run. Compare la lisibilité
   en équipe : tu préfères quoi pour ton contexte (solo vs équipe) ?

**Solution attendue (point 3)** : un verdict de la forme
*« retenu — meilleur F1 macro de la session, recall défaut acceptable
(>0.60), trade-off précision défaut justifié pour le métier crédit »*.

## Pièges fréquents

| Piège | Conséquence |
|---|---|
| Pas de date sur les runs | Impossible de reconstituer l'ordre de la démarche |
| Hyperparams listés sans valeurs (juste `n_estimators=...`) | Run non reproductible |
| Métriques arrondies à 1 décimale | Comparaisons fines impossibles |
| Pas de verdict par run | Le lecteur ne sait pas pourquoi tel modèle est retenu |
| Tableau Excel séparé | Pas dans Git, perdu à la prochaine machine |
| Oublier `sha256` du dataset | Si le dataset change, plus de référence stable |
| Tout passer en MLflow sans discipline | `mlruns/` qui explose, aucune narration retrouvable |

### Symptôme → cause probable

| Symptôme | Cause probable |
|---|---|
| Run 2 jours plus tard donne des chiffres différents | `random_state` oublié quelque part, OU dataset modifié OU env Python différent |
| `experiments.md` devient illisible (50+ runs en vrac) | Manque de sections — séparer par semaine ou par objectif |
| MLflow UI vide alors qu'on a run plusieurs fois | `mlflow.set_experiment` oublié ou pas dans le bon dossier |
| Diff Git énorme sur `experiments.md` | Tableau au lieu de sections — préférer markdown structuré |

## Pour aller plus loin

- Doc officielle : [MLflow tracking](https://mlflow.org/docs/latest/tracking.html)
  - Article : [Reproducibility in ML — Joelle Pineau](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)
  - Référence : *Hidden Technical Debt in Machine Learning Systems*, Sculley et al., NeurIPS 2015 — pour comprendre pourquoi le traçage compte.
  - À venir en M5 : *Versioning de datasets avec DVC* (en bonus si nécessaire).

## Vérification (checklist apprenant)

- [ ] Mon `experiments.md` contient **au moins 2 runs** documentés
  - [ ] Chaque run a : date, modèle, hyperparams, métriques, verdict
  - [ ] Mes hyperparams sont copiables-collables pour reproduire le run
  - [ ] J'ai commité `experiments.md` dans Git avec un message explicite
  - [ ] Je peux montrer ce fichier à un collègue qui décide en 2 minutes