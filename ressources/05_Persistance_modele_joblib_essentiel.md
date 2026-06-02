# Persistance de modèle avec joblib + métadonnées — Mini-cours

> Brief associé : M1-B1
> Durée de lecture + pratique : ~35 min
> Pré-requis : modèle entraîné qui passe l'évaluation, prêt à être livré.

## Pourquoi cette techno ?

Quand le client Pyrenex demande *« livrez-nous le modèle persisté »*, il
attend :

1. Un **fichier qui se recharge** sans la moindre dépendance au notebook
   d'entraînement.
2. Une **traçabilité** : quelle version ? Quelles métriques au moment du
   serrage ? Quand a-t-il été entraîné ?
3. Une **stabilité prédictive** : *à données identiques, prédictions
   identiques*. Pas de dérive silencieuse entre notebook et API.
4. Un **environnement reproductible** : un `requirements.txt` (ou `pyproject.toml`)
   qui fige les versions des libs, idéalement consommé dans un venv ou un
   container — un `.joblib` sans son environnement, c'est une clé sans serrure.

Sans ces 4 garanties, **le modèle est inutilisable en production**. C'est
exactement ce qu'on prépare ici pour le passage en M1-B2 (API conteneurisée)
puis M5 (CI/CD).

**Alternatives à connaître :**

| Format | Quand l'utiliser ? |
|---|---|
| **joblib (.joblib)** | Standard scikit-learn. Notre choix M1. **`joblib` = wrapper de `pickle` optimisé pour les objets contenant de gros arrays NumPy** — plus rapide, compression native. |
| **pickle (.pkl)** | Standard Python plus large, mais joblib est plus rapide sur les objets contenant des arrays NumPy. À éviter pour scikit-learn. |
| **ONNX** | Format universel, déployable hors-Python. Pas pour M1 (M5 envisageable). |
| **PMML** | Anciens systèmes BI. Quasi-obsolète. |
| **`mlflow.sklearn.log_model`** | Si on utilise MLflow. Inclut versionning automatique. Bonus M1. |

> 🔑 **Règle d'or de la persistance** : *un modèle sans métadonnées n'est
> pas un modèle livrable*. Le `.joblib` seul ne dit ni qui l'a entraîné, ni
> sur quelles données, ni avec quelles performances — il est techniquement
> rechargeable mais professionnellement inexploitable.

> ⚠️ **Avertissement sécurité — à retenir tout de suite** : un fichier
> `.joblib` (comme `.pkl`) est **du code Python sérialisé**. Le charger
> exécute du code sur ta machine. **Ne jamais `joblib.load` un fichier
> dont tu ne connais pas la source.** Pour Pyrenex, le seul `.joblib`
> légitime est celui que **toi** as produit dans ce repo Git, ou celui
> qu'une pipeline CI/CD authentifiée a publié. Pas de `.joblib` reçu
> par mail, pas de `.joblib` téléchargé d'un site tiers.

### Persistance vs packaging — deux gestes distincts

On confond souvent les deux. Pour le brief M1, il faut distinguer :

| Geste | Ce qu'il produit | Outil |
|---|---|---|
| **Persistance** | Sérialiser l'objet Python entraîné en un fichier rechargeable | `joblib.dump` |
| **Packaging** | Assembler le fichier sérialisé + **métadonnées** + **version** + **artefacts adjacents** (requirements, README, hash dataset) en un livrable cohérent | convention `model/` + `.json` + tag git |

La **persistance** te donne un fichier. Le **packaging** te donne quelque
chose qu'une équipe tierce peut consommer. En M1-B2, c'est le packaging
complet que l'API consomme — pas juste le `.joblib`.

## Concepts clés

- **`joblib.dump(model, path, compress=3)`** : sérialise un objet Python
  (modèle, pipeline). `compress=3` : trade-off vitesse/taille (4 → plus
  compressé, 0 → pas compressé). Toujours **compresser** pour éviter
  les artefacts > 100 Mo qui bloquent Git.
- **`joblib.load(path)`** : recharge l'objet. Doit fonctionner sur une
  machine ne disposant **que** du fichier (et des bonnes versions de libs).
- **Versions** : un modèle `.joblib` produit avec `scikit-learn==1.5.1`
  doit être rechargé avec **la même version mineure** (ou très proche).
  Sinon : warnings, voire crash.
- **Métadonnées JSON adjacentes** : convention FastIA. À côté du
  `.joblib`, un `.json` qui décrit le modèle. Pour M5, ce JSON sera lu
  par `/info` de l'API.
- **Pipeline complet** : on persiste **le `Pipeline` scikit-learn entier**
  (préprocessing + modèle), **pas juste le classifieur**. Sinon le client
  doit ré-appliquer la préparation à la main → source d'erreurs.

### Métadonnées obligatoires en M1-B1 (contrat minimum)

Pour qu'un livrable soit accepté en M1-B1, **les 5 clés suivantes sont
obligatoires** dans le `.json` adjacent — pas plus, pas moins :

| Clé | Type | Pourquoi obligatoire ? |
|---|---|---|
| `model_version` | string (`v2.0.0`) | Sans version, impossible de tracer ce qui tourne en prod |
| `created_at` | ISO 8601 UTC | Pour reconstituer la chronologie de production |
| `sklearn_version` | string | Versions ≠ entre dump et load → crash silencieux |
| `dataset_sha256` | string | Identifie *les données exactes* sur lesquelles le modèle a été entraîné |
| `metrics_holdout` | dict | Performance officielle annoncée au client — ne plus jamais bouger |

Le reste des champs ci-dessous est **recommandé** mais pas bloquant — à
ajouter au fil de la rigueur exigée par le contexte.

### Convention métadonnées complète (recommandée en M1, exigée en M5)

```json
{
  "model_name": "pyrenex_risk_v2",
  "model_version": "v2.0.0",
  "created_at": "2026-05-19T16:42:00Z",
  "created_by": "<prenom> <nom>",
  "sklearn_version": "1.5.1",
  "python_version": "3.11.9",
  "dataset_sha256": "a3f9b8...",
  "dataset_n_rows": 24000,
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": 10,
    "class_weight": "balanced",
    "random_state": 42
  },
  "metrics_holdout": {
    "f1_macro": 0.701,
    "f1_default": 0.485,
    "roc_auc": 0.741,
    "recall_default": 0.612
  },
  "feature_columns": ["loan_amnt", "term", "int_rate", "..."],
  "target_column": "loan_status",
  "target_mapping": {"Fully Paid": 0, "Charged Off": 1}
}
```

Ce JSON est **le contrat entre M1 (toi) et M5 (CI/CD)**.

## Exemple minimal qui tourne

```python
# example_persist.py — versions testées : python 3.11+, scikit-learn 1.5+, joblib 1.4+
import json
import platform
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import joblib
import sklearn


def persist_model(
    pipeline,                # scikit-learn Pipeline complet (preprocess + classifier)
    metrics: dict[str, float],
    hyperparams: dict,
    feature_columns: list[str],
    dataset_path: Path,
    output_dir: Path,
    model_version: str = "v2.0.0",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "pyrenex_risk_v2.joblib"
    meta_path = output_dir / "pyrenex_risk_v2.json"

    # 1. Sérialiser le pipeline complet
    joblib.dump(pipeline, model_path, compress=3)

    # 2. Calculer le hash du dataset (traçabilité)
    dataset_sha256 = sha256(dataset_path.read_bytes()).hexdigest()

    # 3. Métadonnées
    metadata = {
        "model_name": "pyrenex_risk_v2",
        "model_version": model_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "dataset_sha256": dataset_sha256,
        "hyperparameters": hyperparams,
        "metrics_holdout": metrics,
        "feature_columns": feature_columns,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return model_path, meta_path


def contract_test_model(
    model_path: Path,
    X_sample,                       # noqa: N803 — convention ML pour la matrice de features
    expected_classes: set[int] | None = None,
    expected_first_proba: list[float] | None = None,
) -> None:
    """Contract test : valide schéma + stabilité prédictive d'un modèle rechargé.

    Plus exigeant qu'un simple `print OK` — on vérifie que le modèle respecte
    la signature attendue par l'API M1-B2 :
    - shape de `predict` et `predict_proba` cohérentes
    - classes prédites dans l'ensemble attendu
    - probabilités dans [0, 1]
    - (optionnel) stabilité bit-à-bit avec une référence notebook
    """
    pipeline = joblib.load(model_path)

    prediction = pipeline.predict(X_sample.head(3))
    proba = pipeline.predict_proba(X_sample.head(3))

    assert prediction.shape == (3,), f"shape predict={prediction.shape}, attendu (3,)"
    assert proba.shape == (3, 2), f"shape predict_proba={proba.shape}, attendu (3, 2)"
    assert (proba >= 0).all() and (proba <= 1).all(), "probabilités hors [0, 1]"

    if expected_classes is not None:
        observed = set(prediction.tolist())
        assert observed <= expected_classes, f"classes inattendues : {observed - expected_classes}"

    if expected_first_proba is not None:
        observed = proba[0].round(6).tolist()
        reference = [round(p, 6) for p in expected_first_proba]
        assert observed == reference, (
            f"dérive prédictive — observé {observed}, référence notebook {reference}"
        )

    print(f"Contract test OK — shapes valides, probas dans [0,1], stabilité confirmée.")
```

→ Tu obtiens un couple `(pyrenex_risk_v2.joblib, pyrenex_risk_v2.json)`
prêt pour M1-B2, **et** une fonction de test contractuel qui détecte la
moindre dérive entre notebook et fichier persisté.

### Le double geste : `metrics_test_internal` ≠ `metrics_holdout`

> 🔑 Le json se construit **en deux temps** — pas en un. La règle d'or
> *« le holdout n'apparaît PAS dans les runs intermédiaires »* (cf.
> mini-cours 04) impose cette séparation.

Tu vas écrire **deux clés de métriques** dans le `.json`, mais elles
n'arrivent **pas en même temps** :

| Clé | Quand l'écrire ? | Source | Outil |
|---|---|---|---|
| `metrics_test_internal` | À chaque entraînement (run de comparaison) | Split interne train/test du jeu d'entraînement (cross-validation ou hold-out 80/20 du `lending_club_train.csv`) | `python src/train.py` |
| `metrics_holdout` | **Une seule fois**, à la toute fin, sur le verdict | Le `lending_club_holdout.csv` scellé, **jamais vu pendant la sélection** | `python src/evaluate.py --update-meta` |

**Pourquoi cette gymnastique ?** Si tu écris `metrics_holdout` à chaque
itération, tu fais du *cherry-picking sur le holdout* — la métrique
publiée n'est plus une estimation honnête du modèle en production.
C'est exactement ce que Pyrenex devra défendre devant son contrôleur
interne en M9. Le holdout doit rester **scellé** jusqu'à la décision
finale.

**Workflow concret** (squelette M1-B1 fourni) :

```bash
# 1. À chaque run de comparaison (étapes 3-4 du brief — tâches "tronquer / tuner")
python src/train.py --config balanced   # produit .joblib + .json avec metrics_test_internal
python src/train.py --config tuned      # idem, écrit metrics_test_internal

# 2. UNE fois — quand tu as choisi le modèle final, étape 5 du brief
python src/evaluate.py \
    --model models/pyrenex_risk_v2.joblib \
    --data data/lending_club_holdout.csv \
    --update-meta                       # PATCH le .json en ajoutant metrics_holdout
```

`--update-meta` **ajoute** la clé `metrics_holdout` sans toucher au reste
du json (les autres clés survivent intactes). C'est ce json enrichi qui
sera servi par `/info` en M1-B2.

> ⚠️ **À ne pas oublier** : sans `--update-meta`, ton json final n'aura
> pas `metrics_holdout` et **M1-B2 cassera** au runtime (`KeyError` dans
> `/info`). Si tu vois ce symptôme en M1-B2, retourne lancer
> `evaluate.py --update-meta` sur ton repo M1-B1 et re-pousse le json
> mis à jour.

### Format de livraison final attendu

Voilà la structure standard qu'on remet au client (et qui sera consommée
telle quelle par l'API M1-B2) :

```text
models/
├── pyrenex_risk_v2.joblib        # modèle persisté (Pipeline complet, compress=3)
├── pyrenex_risk_v2.json          # métadonnées (5 clés obligatoires + recommandées,
│                                 # dont metrics_holdout patché par evaluate.py)
└── README.md                     # facultatif : comment recharger en 3 lignes
```

Le `requirements.txt` du repo fige les versions de libs — il vit à la
racine du repo, pas dans `models/`. Tag git `v2.0.0` sur le commit qui
publie ce dossier **après** le `--update-meta` (sinon tu tagges un json
incomplet). Voilà ce qui rentre dans la pipeline M5.

## Exercice guidé

À partir de ton meilleur modèle (`exp_002` ou suivant) :

1. **Persiste-le** avec `persist_model(...)` ci-dessus. Vérifie la taille
   du `.joblib` (`compress=3` doit te donner < 10 Mo pour un RF de 200
   arbres).
2. **Lance le contract test** dans un script séparé (`contract_test.py`) —
   **pas dans ton notebook**. Utilise `contract_test_model(...)` ci-dessus
   en lui passant `expected_first_proba` capturé depuis ton notebook
   d'entraînement. Tous les `assert` doivent passer — c'est ce qui prouve
   que ton `.joblib` est **livrable**, pas juste rechargeable.
3. **Ouvre le `.json`** et vérifie que toutes les clés sont remplies. Aucune
   valeur ne doit être `null` (sauf si justifié).
4. **Patche `metrics_holdout`** une fois ton verdict final décidé :
   ```bash
   python src/evaluate.py \
       --model models/pyrenex_risk_v2.joblib \
       --data data/lending_club_holdout.csv \
       --update-meta
   ```
   Ré-ouvre le `.json` : les **5 clés obligatoires** sont toutes là, dont
   `metrics_holdout` non vide. Si une clé manque, ton M1-B2 cassera.
5. **Commit dans Git** les 2 fichiers (`.joblib` + `.json` post-patch),
   avec un message `feat(model): persist pyrenex_risk_v2 with metadata`.
6. **Tag git** : `git tag -a v2.0.0 -m "Pyrenex risk model v2"` —
   préparation directe pour M1-B2. **Ne tagge qu'après** `--update-meta`
   (sinon tu figes un json incomplet).

**Solution attendue (point 2)** : si l'assert sur `expected_first_proba`
échoue, le coupable habituel est qu'on a persisté **le classifieur seul** au
lieu du **Pipeline complet** — vérifie que `pipeline.named_steps` contient
bien tes étapes de preprocessing. Autres causes possibles : version sklearn
différente entre dump et load, ou jeu de features réordonné silencieusement.

## Pièges fréquents

| Piège | Conséquence |
|---|---|
| Persister `RandomForestClassifier` seul (pas le Pipeline) | Le client doit réimplémenter la préparation, source d'erreurs |
| Oublier `compress=3` | Fichier 50-200 Mo, problèmes Git, lent à charger |
| Pas de métadonnées JSON | En M5, on ne sait pas quelle version est servie |
| Versions sklearn différentes train/load | Warnings, parfois crash silencieux |
| Charger un `.joblib` venant d'une source inconnue | **Risque sécurité** : pickle peut exécuter du code arbitraire. Toujours valider la source. |
| Commiter un `.joblib` > 100 Mo dans Git | Git refuse / clone lent — utiliser Git LFS, ou stocker hors Git (S3 en M5) |
| Métadonnées dans le notebook seulement | Information perdue dès qu'on quitte le notebook |
| Oublier de lancer `evaluate.py --update-meta` après le verdict | `metrics_holdout` absent du json → `KeyError` au démarrage de l'API M1-B2 |
| Écrire `metrics_holdout` dès `train.py` (avant le verdict final) | Cherry-picking sur le holdout : la métrique annoncée au client est biaisée — règle d'or *comparabilité* violée |

### Symptôme → cause probable

| Symptôme | Cause probable |
|---|---|
| `joblib.load` : `KeyError: 'sklearn.ensemble...'` | Version sklearn incompatible — recréer un venv avec la même version |
| Prédiction après reload différente de la prédiction notebook | Seul le classifieur a été persisté, pas le Pipeline — re-dump le Pipeline complet |
| `UserWarning: Trying to unpickle estimator from version X.Y when using version A.B` | Versions différentes — souvent OK si mineur identique, à vérifier en M5 |
| `.joblib` > 100 Mo | Probablement `compress=0` ou `n_estimators` énorme |
| `FileNotFoundError` en chargeant le `.joblib` dans le container Docker (M1-B2) | Path relatif vs absolu — utiliser `Path(__file__).parent / "model.joblib"` |
| `KeyError: 'metrics_holdout'` au démarrage de l'API M1-B2 | `evaluate.py --update-meta` n'a pas été lancé en M1-B1 — relance-le, re-commit le json, re-pull en M1-B2 |

## Pour aller plus loin

- Doc officielle : [joblib — persistence](https://joblib.readthedocs.io/en/latest/persistence.html)
- Doc officielle : [scikit-learn — model persistence](https://scikit-learn.org/stable/model_persistence.html)
- Article : [Pickling pitfalls](https://docs.python.org/3/library/pickle.html#restricting-globals) — sécurité.
- À venir en M5 : *Model Registry MLflow*, **DVC** pour les modèles, signatures cryptographiques.

## Vérification (checklist apprenant)

- [ ] J'ai persisté **le Pipeline complet** (pas juste le classifieur)
- [ ] `train.py` a produit le `.json` avec `metrics_test_internal`
- [ ] J'ai lancé `python src/evaluate.py --update-meta` **après** mon verdict final
- [ ] Mon `.json` métadonnées contient les **5 clés obligatoires** : `model_version`, `created_at`, `sklearn_version`, `dataset_sha256`, `metrics_holdout` (toutes non-nulles)
- [ ] Mon **contract test** (`contract_test.py`) passe tous les asserts dans un script séparé
- [ ] Mon `.joblib` fait **moins de 10 Mo** (sinon `compress=3` à vérifier)
- [ ] Mon `requirements.txt` est figé et co-localisé avec le repo
- [ ] J'ai commité + taggé `v2.0.0` dans Git **après** `--update-meta` — prêt à être tiré en M1-B2
