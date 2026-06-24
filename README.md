# Game Experience Analysis — Stress Classification

Predicting a player's stress level from gaming-session behavioral and psychological data using scikit-learn ensemble classifiers, multiple feature-selection strategies, and repeated stratified k-fold cross-validation.

> **Tech stack:** Python 3.8 · Jupyter · scikit-learn · pandas · NumPy · seaborn · matplotlib · plotly

---

## Overview

In a controlled study, participants played a computer game while behavioral and psychological data were recorded. Validated psychological tests (Depression / Anxiety / Stress scales) provided ground-truth labels, and the goal of this project is to **predict each player's stress level** (a 3-class target) from the recorded features.

This is framed as a supervised **multi-class classification** problem. The notebook runs end-to-end: from exploratory data analysis, through three independent feature-selection methods, to model comparison and final hyperparameter tuning.

> **Note on data:** The dataset is confidential and is **not** included in this repository. Only the analysis code is shared. The notebook expects a local Excel file (`Game data (imputed).xlsx`) with one row per participant.

---

## What's inside

| File | Description |
|------|-------------|
| `Game Experience Analysis (Stress).ipynb` | The full analysis: EDA, feature selection, model comparison, tuning, and a sample prediction. |
| `LICENSE` | MIT License. |
| `.gitignore` | Ignore rules. |

---

## The data (schema)

The target column is `Stress` (3 classes). The ~34 predictor columns observed in the notebook fall into three groups:

- **Demographics & background:** `age`, `sex`, handedness (`r or l`), `game experience`
- **Psychological scales:** `Depression`, `Anxiety`
- **In-game behavioral metrics:** phase durations, hit counts and scores across game phases, "eating health" counts, final grade / score / hits, and per-boss durations and hit counts (`Boss1–3 duration`, `Boss1–3 hitCount`, `increasing1/2 hitCount`, etc.)

---

## Methods / Approach

**1. Exploratory data analysis**
- Class-frequency distribution of the `Stress` target (bar and pie charts).
- Breakdown of stress classes by `sex` using plotly donut charts.

**2. Feature selection (three independent strategies)**
- **RFECV** — recursive feature elimination with cross-validation, using a `DecisionTreeClassifier` estimator, to automatically choose the number of features.
- **Tree-based importance** — feature importances from `ExtraTreesClassifier` and `RandomForestClassifier`.
- **Mutual information** — `SelectKBest` with `mutual_info_classif`, keeping features with information > 0 and ranking them.

**3. Model comparison**
Each feature subset is evaluated across a suite of classifiers:
`Perceptron`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `AdaBoostClassifier`, `ExtraTreesClassifier`, and `MLPClassifier`.

**4. Validation**
All models are scored with **`RepeatedStratifiedKFold`** (`n_splits=3`, `n_repeats=3`, `random_state=1`) on `accuracy`, reporting mean ± standard deviation.

**5. Final model & tuning**
`ExtraTreesClassifier` on the mutual-information feature subset is selected as the final model, then tuned over `n_estimators` and `min_samples_split` before fitting and demonstrating a single-sample prediction.

---

## How to run

This is a self-contained Jupyter notebook. You'll need Python 3.8+ and the dependencies below, plus a local copy of the (confidential) dataset.

```bash
# 1. Install dependencies
pip install numpy pandas scikit-learn seaborn matplotlib plotly openpyxl jupyter

# 2. Launch the notebook
jupyter notebook "Game Experience Analysis (Stress).ipynb"
```

Before running, update the hard-coded data paths in the notebook (e.g. the `pd.read_excel(...)` call and the figure-saving paths) to point at your local files.

---

## Results

Reported as mean accuracy (± std) under repeated stratified 3-fold cross-validation. These are the figures produced by the executed notebook:

- **Feature-set comparison** — across all three feature-selection strategies, `ExtraTreesClassifier` was consistently among the strongest classifiers. On the **mutual-information** feature subset it reached the best comparison accuracy (**≈ 0.71**), ahead of Random Forest, Decision Tree, AdaBoost, Gradient Boosting, the Perceptron, and the MLP.
- **Tuning** — grid exploration of `ExtraTreesClassifier` hyperparameters (`n_estimators`, `min_samples_split`) on the mutual-information subset pushed cross-validated accuracy up to roughly **0.73–0.77**, the highest scores observed in the notebook.

> Stress labels are imbalanced and some classes are small, so per-fold accuracy varies meaningfully — results are reported with their standard deviation rather than as single point estimates.

---

## License

Released under the [MIT License](LICENSE). Copyright (c) 2021 Hamid.
