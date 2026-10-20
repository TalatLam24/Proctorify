# Approach 03 Run Log

**Run timestamp:** 2026-08-30_13-32-21

## Configuration

- Training subjects: Bushra, Prova, Shahin, Zarif
- Test subjects: Lam, Ikra
- Test cap per subject: 40,000 frames
- Feature count: 53
- Random state: 42
- EMA alpha: 0.3
- Ensemble method: performance-weighted (Option C)

## Data Summary

- Training: 375,219 frames (131,399 positive, 243,820 negative)
- Test: 80,000 frames (29,362 positive, 50,638 negative)
- Train class balance: 35.0% positive
- Test class balance: 36.7% positive

## Individual Model Results

| Model | Overall F2 | Best Threshold | AUC-PR | Training Time (s) | Device |
|---|---|---|---|---|---|
| xgboost | 0.8829 | 0.06 | 0.9201 | 0.0 | loaded from disk |
| random_forest | 0.8691 | 0.18 | 0.8927 | 0.0 | loaded from disk |
| logistic_regression | 0.8084 | 0.34 | 0.8443 | 0.0 | loaded from disk |
| svm | 0.8302 | 0.09 | 0.8800 | 674.0 | CPU |

## Per-Subject F2 (at each model's best overall threshold)

| Model | Ikra | Lam |
|---|---|---|
| xgboost | 0.8801 | 0.8863 |
| random_forest | 0.8682 | 0.8701 |
| logistic_regression | 0.8082 | 0.8086 |
| svm | 0.8313 | 0.8288 |

## Ensemble (Performance-Weighted)

| Model | Weight |
|---|---|
| xgboost | 0.2604 |
| random_forest | 0.2563 |
| logistic_regression | 0.2384 |
| svm | 0.2449 |

- **Overall F2: 0.8635**
- Best threshold: 0.21
- AUC-PR: 0.8972

### Ensemble Per-Subject F2

| Subject | F2 | Precision | Recall |
|---|---|---|---|
| Ikra | 0.8634 | 0.6037 | 0.9674 |
| Lam | 0.8636 | 0.8411 | 0.8694 |

## SVM Hyperparameter Note

**No full grid search was performed in this run.** A partial grid search
(interrupted due to time) found that C=1, gamma=0.01 gave AUC-PR=0.8809,
the best observed. We use these hyperparameters directly.

The partial grid search results (for reference):

| C | gamma | AUC-PR |
|---|---|---|
| 0.1 | 0.001 | 0.8397 |
| 0.1 | 0.01 | 0.8799 |
| 0.1 | 0.1 | 0.7930 |
| 0.1 | 1 | 0.7569 |
| 1 | 0.001 | 0.8654 |
| 1 | 0.01 | **0.8809** (best) |
| 1 | 0.1 | 0.8109 |
| 1 | 1 | 0.5101 |
| 10 | 0.001 | 0.8793 |
| 10 | 0.01 | 0.8740 |
| 10 | 0.1 | 0.8087 |
| 10 | 1 | 0.7580 |
| 100 | 0.001 | 0.8650 |
| 100 | 0.01 | 0.8556 |
| 100 | 0.1 | 0.8043 |