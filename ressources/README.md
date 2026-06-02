# Ressources M1-B1 — Réentraîner et challenger un modèle de scoring crédit

> Brief associé : **M1-B1**.
> Mode : individuel, présentiel mardi, 5 heures synchrone.
> Le brief lui-même est diffusé sur **Simplonline** (énoncé + liens utiles).

Ce dossier rassemble **les 5 mini-cours pédagogiques** auxquels le brief M1-B1
fait référence + les liens officiels. Le squelette de code, ce sont les
fichiers à la **racine de ce repo** que tu as créé via « Use this template ».

---

## 📚 Ordre de mobilisation au fil de la journée

| Tâche du brief | Durée | Mini-cours associé |
|---|---|---|
| 1. Comprendre la baseline `pyrenex-risk-v1` | 30 min | (README du repo baseline) |
| 2. EDA du nouveau dataset Lending Club | 1 h | [`01_Pandas_Sklearn_split_essentiel.md`](./01_Pandas_Sklearn_split_essentiel.md) (partie EDA) |
| 3. Préparation et split stratifié | 30 min | [`01_Pandas_Sklearn_split_essentiel.md`](./01_Pandas_Sklearn_split_essentiel.md) |
| 4. Entraînement + benchmark hyperparamètres | 1 h 30 | [`03_RandomForest_hyperparams_essentiel.md`](./03_RandomForest_hyperparams_essentiel.md) |
| 5. Évaluation + verdict | 45 min | [`02_Metrics_classif_desequilibree_essentiel.md`](./02_Metrics_classif_desequilibree_essentiel.md) |
| 6. Traçage des expérimentations | en continu | [`04_Tracage_experiments_md_essentiel.md`](./04_Tracage_experiments_md_essentiel.md) |
| 7. Persistance et release du modèle | 30 min | [`05_Persistance_modele_joblib_essentiel.md`](./05_Persistance_modele_joblib_essentiel.md) |

> 💡 **Tu n'es pas obligé·e de lire les mini-cours en amont.** Chacun est conçu
> pour être consulté **au moment où tu en as besoin**, pendant la tâche
> correspondante. Lecture + exercice guidé en ~15-20 min chacun.

---

## 🛠️ Ton repo de travail

Ce dossier `ressources/` est livré dans le **repo template GitHub**
[`Formation-SIMPLON-IA/ia-atos-parcours-m1-b1`](https://github.com/Formation-SIMPLON-IA/ia-atos-parcours-m1-b1).
Si tu lis ce fichier, c'est que tu as déjà cliqué sur **« Use this template »**
et cloné ton repo perso `M1-B1-scoring-<prénom>`. Pour l'installation et le
démarrage en 4 commandes, cf. le [`README.md`](../README.md) à la racine du
repo.

> ⚠️ Le **repo baseline** est public et **séparé** :
> [`Formation-SIMPLON-IA/pyrenex-risk-v1`](https://github.com/Formation-SIMPLON-IA/pyrenex-risk-v1).
> Tu le clones en lecture seule (`git clone https://github.com/Formation-SIMPLON-IA/pyrenex-risk-v1.git`).
> Le repo perso que tu travailles est **ton** repo, où tu produiras le
> nouveau modèle `pyrenex_risk_v2`. Tous tes commits vont chez **toi**,
> pas chez Simplon — c'est ton historique que tu pourras montrer en évaluation.

---

## 🎯 Ce qu'on cherche à atteindre

À la fin de M1-B1, tu dois avoir :

- Un **notebook d'EDA + d'entraînement** propre, exécutable top-to-bottom
- **Au moins 2 jeux d'hyperparamètres** testés sur RandomForest, comparés
- Des **métriques pertinentes pour classification déséquilibrée** : F1 macro,
  ROC-AUC, matrice de confusion
- Un **`experiments.md`** (ou `mlruns/` si MLflow) qui trace tes runs
- Un **modèle persisté** `models/pyrenex_risk_v2.joblib` + métadonnées JSON
- Un **`verdict.md`** d'1 page : doit-on remplacer la baseline ? Pourquoi ?
- Un **repo GitHub** `M1-B1-scoring-<prénom>` avec ≥ 3 commits propres
  (`feat(eda): ...`, `feat(training): ...`, `docs(verdict): ...`)

→ Compétence visée : **C5 — imiter** (reproduire une démarche d'entraînement
sur un nouveau dataset).

→ ⭐ **Mission étoile optionnelle** : si tronc solide avant 14h, ajoute
soit un Gradient Boosting + SHAP, soit un fine-tuning léger d'un SLM sur
le champ `purpose`. **Non évaluée**, célébrée en restitution 16h.

---

## 🔗 Liens externes

Toutes les URLs externes utilisées dans les mini-cours sont consolidées dans
[`liens_officiels.md`](./liens_officiels.md), vérifiées avant chaque envoi
de brief par l'outillage formateur.

---

## 🆘 Bloqué·e ?

1. **Relire l'Exercice guidé** du mini-cours concerné (chacun a une solution
   attendue à la fin).
2. **Comparer aux performances de la baseline** : le repo `pyrenex-risk-v1`
   donne les chiffres de référence — un écart inattendu = relire ton split.
3. **Vérifier `random_state=42`** sur toutes les fonctions stochastiques
   (`train_test_split`, `RandomForestClassifier`, etc.) → reproductibilité.
4. **Demander en direct mardi** — tu es en présentiel-Discord, autant en
   profiter. N'attends pas d'être bloqué·e 30 min sur le même point.

**Garde-fou** : pas besoin de coder hors mardi 9h-17h. Si tu finis tronc
avant 14h, attaque la **mission étoile** ⭐. Si pas le temps, ne la lance
pas — le tronc passe en priorité absolue.