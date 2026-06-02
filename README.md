# M1-B1 — Squelette repo (Pyrenex Crédit scoring)

> **Repo template GitHub.** Clique sur **« Use this template »** en haut à
> droite de cette page → **Create a new repository** → nomme-le
> `M1-B1-scoring-<prénom>` sur **ton** compte GitHub personnel.
> C'est ce nouveau repo que tu cloneras pour travailler.

---

## 🚀 Démarrage (4 commandes)

```bash
# 0. Clone ton repo perso fraîchement créé
git clone git@github.com:<ton-user>/M1-B1-scoring-<prenom>.git
cd M1-B1-scoring-<prenom>

# 1. Environnement virtuel
python -m venv .venv && source .venv/bin/activate     # Linux/macOS
# .venv\Scripts\activate                              # Windows

# 2. Dépendances
pip install -r requirements.txt

# 3. Vérification
python src/train.py --help     # → doit afficher l'usage du script
```

Si ces 4 commandes marchent, ton poste est prêt.

---

## 📁 Structure du repo

```
M1-B1-scoring-<prenom>/
├── data/
│   ├── lending_club_train.csv           # à télécharger (cf. ci-dessous)
│   └── lending_club_holdout.csv         # à télécharger
├── notebooks/
│   └── M1-B1_template.ipynb             # à dupliquer en M1-B1_<prenom>_scoring.ipynb
├── src/
│   ├── preprocess.py                    # transformations reproductibles
│   ├── train.py                         # script d'entraînement
│   └── evaluate.py                      # métriques sur holdout
├── models/                              # .joblib + .json produits ici
│   └── .gitkeep
├── ressources/                          # 📚 mini-cours d'appui (lecture juste-à-temps)
│   ├── 01_Pandas_Sklearn_split_essentiel.md
│   ├── 02_Metrics_classif_desequilibree_essentiel.md
│   ├── 03_RandomForest_hyperparams_essentiel.md
│   ├── 04_Tracage_experiments_md_essentiel.md
│   ├── 05_Persistance_modele_joblib_essentiel.md
│   ├── liens_officiels.md
│   └── README.md                        # ordre de mobilisation + objectifs
├── contract_test.py                     # à compléter — valide shapes/classes/probas du .joblib
├── experiments.md                       # à compléter run par run
├── verdict.md                           # à rédiger en fin de journée
├── requirements.txt
├── .gitignore
└── README.md (ce fichier — à compléter)
```

---

## 📚 Mini-cours d'appui

Les **5 mini-cours pédagogiques** du brief sont fournis dans
[`./ressources/`](./ressources/). Chacun se lit en ~15-20 min, **au moment où
tu en as besoin** pendant la journée :

| Tâche | Mini-cours |
|---|---|
| EDA + split stratifié | [`01_Pandas_Sklearn_split_essentiel.md`](./ressources/01_Pandas_Sklearn_split_essentiel.md) |
| Métriques pour classification déséquilibrée | [`02_Metrics_classif_desequilibree_essentiel.md`](./ressources/02_Metrics_classif_desequilibree_essentiel.md) |
| Hyperparamètres RandomForest | [`03_RandomForest_hyperparams_essentiel.md`](./ressources/03_RandomForest_hyperparams_essentiel.md) |
| Traçage des runs (`experiments.md`) | [`04_Tracage_experiments_md_essentiel.md`](./ressources/04_Tracage_experiments_md_essentiel.md) |
| Persistance modèle (joblib + JSON) | [`05_Persistance_modele_joblib_essentiel.md`](./ressources/05_Persistance_modele_joblib_essentiel.md) |

Cf. [`./ressources/README.md`](./ressources/README.md) pour l'ordre de mobilisation détaillé.

---

## 📥 Données

Le dataset Lending Club sous-échantillonné (~30 k lignes) t'est fourni par
la formatrice mardi 9h. Place les 2 fichiers dans `data/` :

- `data/lending_club_train.csv` (~24 k lignes)
- `data/lending_club_holdout.csv` (~6 k lignes — **à ne PAS toucher** pendant l'entraînement)

---

## 🧭 Démarche attendue

1. **Comprends la baseline** : clone le repo public
   [`Formation-SIMPLON-IA/pyrenex-risk-v1`](https://github.com/Formation-SIMPLON-IA/pyrenex-risk-v1),
   lis le code et les métriques rapportées.
2. **EDA** dans le notebook (cellules markdown structurées).
3. **Split stratifié** avec `random_state=42`. Le `holdout` reste intact
   jusqu'à l'étape 6 (cf. règle d'or *comparabilité*).
4. **Entraînement** d'au moins 2 jeux d'hyperparamètres dans `src/train.py`.
   Trace chaque run dans `experiments.md` avec score `test` interne (pas
   le holdout).
5. **Évaluation finale sur le holdout** avec `src/evaluate.py` —
   **une seule fois**.
6. **Persistance** du Pipeline complet : `models/pyrenex_risk_v2.joblib`
   + `pyrenex_risk_v2.json` avec les **5 clés obligatoires**
   (`model_version`, `created_at`, `sklearn_version`, `dataset_sha256`,
   `metrics_holdout`).
7. **Contract test** : complète `contract_test.py` et lance-le dans un
   script séparé — tous les `assert` doivent passer.
8. **Verdict** dans `verdict.md` (1 page max) + tag git `v2.0.0`.

Mini-cours d'appui : voir [`./ressources/`](./ressources/).

---

## ✅ Conventions de code

- Python 3.11+
- Type hints sur toutes les signatures publiques
- Pas de `print` — utiliser **Loguru**
- `random_state=42` partout où il y a de l'aléa
- `pathlib.Path` pour les chemins (pas de `os.path`)

---

## 🆘 Bloqué·e ?

1. Relis le mini-cours correspondant à ta tâche actuelle (cf.
   [`./ressources/README.md`](./ressources/README.md)).
2. Vérifie ton split avec **2 runs successifs** : mêmes shapes, mêmes
   contenus → reproductibilité OK.
3. Compare tes métriques à celles de la baseline `pyrenex-risk-v1` —
   un écart > 50% absolu = relire ton preprocessing.
4. Demande en direct mardi.
