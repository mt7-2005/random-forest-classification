# Random Forest — Binary Classification

A Jupyter notebook that trains a **Random Forest** classifier on the Breast Cancer Wisconsin dataset and evaluates it with
**Accuracy, Precision, Recall, F1 and Loss** (log loss). The metrics are written from scratch and checked against `sklearn.metrics`.

Companion to [optimizers-logistic-regression](https://github.com/mt7-2005/optimizers-logistic-regression) (same dataset and split, so the results are directly comparable).

## Dataset — Breast Cancer Wisconsin

- 569 samples, 30 numeric features computed from cell-nucleus images
- Target: `1` = **Malignant** (positive class), `0` = Benign
- Built into scikit-learn (`load_breast_cancer`), so no download or internet is needed
- Stratified 80/20 train/test split (`random_state=42`); no scaling needed for trees

## Notebook contents

[Random_Forest_Classification.ipynb](Random_Forest_Classification.ipynb):

1. Dataset and class distribution
2. How Random Forest works (bagging, feature randomness, Gini impurity, soft voting)
3. Metric functions written from scratch
4. Baseline: a single Decision Tree (shows overfitting) and a tree visualization
5. Random Forest with 300 trees and out-of-bag (OOB) score
6. Results table, check against `sklearn.metrics`, confusion matrix, ROC curves
7. Effect of `n_estimators` (number of trees)
8. Effect of `max_depth`
9. Hyperparameter tuning with `GridSearchCV` (5-fold CV, F1)
10. Feature importance

## Results (test set, 114 samples)

| Model | Loss | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Decision Tree | 2.4238 | 0.9298 | 0.9048 | 0.9048 | 0.9048 |
| Logistic Regression | 0.0773 | 0.9649 | 0.9750 | 0.9286 | 0.9512 |
| **Random Forest** (300 trees) | 0.1135 | **0.9737** | **1.0000** | 0.9286 | **0.9630** |

OOB accuracy: 0.9604. The tuned model (`max_depth=5, max_features=0.5, min_samples_leaf=2, n_estimators=100`, CV F1 = 0.9615) scored the same on the test set as the defaults.

## How to run

**Google Colab:** File → Open notebook → GitHub tab → `mt7-2005/random-forest-classification`, then Runtime → Run all.

**Kaggle:** New Notebook → File → Import Notebook, then Run All (no internet needed).

The GridSearchCV cell tries 54 combinations × 5 folds; it takes about a minute locally and may take a few minutes on Colab.

**Locally:**

```bash
pip install -r requirements.txt
jupyter notebook
```
