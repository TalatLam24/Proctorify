"""
Proctorify Approach 03: Four-Model Hold-Out Evaluation (Modified)
=================================================================
Trains 4 models (XGBoost, Random Forest, Logistic Regression, SVM) on 4
training subjects and evaluates on 2 held-out test subjects (Lam + Ikra).

MODIFICATIONS FROM PREVIOUS VERSION:
  1. SVM grid search removed. Uses best-known hyperparameters from partial
     grid search: C=1, gamma=0.01 (AUC-PR=0.8809, the best observed).
  2. Added "skip if model exists" feature. If a model .joblib file is already
     present in saved_models/, it will be loaded instead of retrained.
  3. Added progress indicators throughout the script.
  4. Added time tracking for each step.

Methodology (unchanged):
  - Test set: 40,000 frames per test subject (Option B - physically balanced)
  - Ensemble: Performance-weighted by individual F2 (Option C)
  - Hyperparameters: Defaults for XGBoost/RF/LR, fixed for SVM
  - Evaluation: Per-subject F2 reporting + overall F2

Usage:
    python train_4models.py

Outputs:
    saved_models/xgboost.joblib
    saved_models/random_forest.joblib
    saved_models/logistic_regression.joblib
    saved_models/logistic_regression_scaler.joblib
    saved_models/svm.joblib
    saved_models/svm_scaler.joblib
    saved_models/ensemble_weights.json
    results/run_<timestamp>/run_log.md
    results/run_<timestamp>/metrics.json
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import yaml
import joblib
from datetime import datetime

# Models
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import fbeta_score, precision_score, recall_score, average_precision_score

# Shared utilities
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'shared'))
sys.path.insert(0, SHARED_DIR)
from losocv import apply_ema_per_session


# ============================================================================
# PROGRESS INDICATOR
# ============================================================================

def log_step(step_num, total_steps, message):
    """Print a clearly visible step header."""
    print(f"\n{'='*60}")
    print(f"  STEP {step_num}/{total_steps}: {message}")
    print(f"{'='*60}")


def log_progress(message):
    """Print a progress message with timestamp."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"  [{timestamp}] {message}")


def log_elapsed(start_time, label):
    """Print elapsed time for an operation."""
    elapsed = time.time() - start_time
    if elapsed < 60:
        print(f"  -> {label} completed in {elapsed:.1f}s")
    else:
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        print(f"  -> {label} completed in {minutes}m {seconds:.1f}s")


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config():
    config_path = os.path.join(SCRIPT_DIR, 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================================
# DATA LOADING AND SPLITTING
# ============================================================================

def load_data(config):
    csv_path = config['data']['grand_csv']
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(SCRIPT_DIR, csv_path)
    log_progress(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    log_progress(f"Total: {len(df):,} rows, {len(df.columns)} columns")
    log_progress(f"Subjects in data: {sorted(df['Subject_ID'].unique())}")
    return df


def prepare_labels(df, config):
    label_col = config['label']['column']
    positive_class = config['label']['positive_class']
    df = df.copy()
    df['y_binary'] = (df[label_col] == positive_class).astype(int)
    n_pos = df['y_binary'].sum()
    n_neg = len(df) - n_pos
    log_progress(f"Binary labels: {n_pos:,} positive ({positive_class}), {n_neg:,} negative")
    return df


def split_train_test(df, config):
    train_subjects = config['subjects']['training']
    test_subjects = config['subjects']['testing']
    cap = config['test_cap_frames_per_subject']

    # Validate test subjects exist in data
    available_subjects = sorted(df['Subject_ID'].unique())
    missing_test = [s for s in test_subjects if s not in available_subjects]
    if missing_test:
        print(f"\nERROR: Test subjects not found in data: {missing_test}")
        print(f"Available subjects: {available_subjects}")
        sys.exit(1)

    # Validate training subjects exist
    missing_train = [s for s in train_subjects if s not in available_subjects]
    if missing_train:
        print(f"\nERROR: Training subjects not found in data: {missing_train}")
        sys.exit(1)

    # Split
    log_progress(f"Splitting data (train: {train_subjects}, test: {test_subjects})")
    train_df = df[df['Subject_ID'].isin(train_subjects)].copy()
    test_df_full = df[df['Subject_ID'].isin(test_subjects)].copy()

    # Cap each test subject to N frames
    capped_test_dfs = []
    for subject in test_subjects:
        subject_df = test_df_full[test_df_full['Subject_ID'] == subject].copy()
        subject_df = subject_df.sort_values('Timestamp').reset_index(drop=True)
        if len(subject_df) > cap:
            subject_df = subject_df.head(cap)
        capped_test_dfs.append(subject_df)
        log_progress(f"  Test subject {subject}: {len(subject_df):,} frames (cap: {cap:,})")
    test_df = pd.concat(capped_test_dfs, ignore_index=True)

    log_progress(f"Training set: {len(train_df):,} frames, {train_df['y_binary'].mean()*100:.1f}% positive")
    log_progress(f"Test set: {len(test_df):,} frames, {test_df['y_binary'].mean()*100:.1f}% positive")

    return train_df, test_df


def get_xy(df, config):
    feature_cols = config['feature_columns']
    X = df[feature_cols].values
    y = df['y_binary'].values
    return X, y


# ============================================================================
# MODEL TRAINING (with skip-if-exists feature)
# ============================================================================

def load_or_train_xgboost(X_train, y_train, config, models_dir):
    model_path = os.path.join(models_dir, 'xgboost.joblib')
    if os.path.exists(model_path):
        log_progress(f"XGBoost model found at {model_path}, loading instead of retraining...")
        model = joblib.load(model_path)
        return model, 0, "loaded from disk"

    log_progress("Training XGBoost on GPU...")
    cfg = config['xgboost']
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = {
        'n_estimators': cfg['n_estimators'],
        'max_depth': cfg['max_depth'],
        'learning_rate': cfg['learning_rate'],
        'subsample': cfg['subsample'],
        'colsample_bytree': cfg['colsample_bytree'],
        'min_child_weight': cfg['min_child_weight'],
        'gamma': cfg['gamma'],
        'reg_alpha': cfg['reg_alpha'],
        'reg_lambda': cfg['reg_lambda'],
        'random_state': config['random_state'],
        'verbosity': cfg.get('verbosity', 0),
        'n_jobs': cfg.get('n_jobs', -1),
        'scale_pos_weight': scale_pos_weight,
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'tree_method': cfg['tree_method'],
        'device': cfg['device'],
    }

    start = time.time()
    try:
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        device = "GPU (cuda)"
    except Exception as e:
        log_progress(f"XGBoost GPU failed: {e}")
        log_progress("Falling back to CPU...")
        params['device'] = 'cpu'
        params['tree_method'] = 'hist'
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        device = "CPU (fallback)"
    elapsed = time.time() - start
    log_elapsed(start, f"XGBoost training on {device}")
    joblib.dump(model, model_path)
    log_progress(f"Saved to {model_path}")
    return model, elapsed, device


def load_or_train_random_forest(X_train, y_train, config, models_dir):
    model_path = os.path.join(models_dir, 'random_forest.joblib')
    if os.path.exists(model_path):
        log_progress(f"Random Forest model found at {model_path}, loading instead of retraining...")
        model = joblib.load(model_path)
        return model, 0, "loaded from disk"

    log_progress("Training Random Forest on CPU (this takes ~3-4 minutes)...")
    cfg = config['random_forest']
    params = {
        'n_estimators': cfg['n_estimators'],
        'min_samples_split': cfg['min_samples_split'],
        'min_samples_leaf': cfg['min_samples_leaf'],
        'max_features': cfg['max_features'],
        'class_weight': cfg['class_weight'],
        'random_state': config['random_state'],
        'n_jobs': cfg.get('n_jobs', -1),
    }
    if cfg.get('max_depth') is not None:
        params['max_depth'] = cfg['max_depth']

    start = time.time()
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    log_elapsed(start, "Random Forest training")
    joblib.dump(model, model_path)
    log_progress(f"Saved to {model_path}")
    return model, time.time() - start, "CPU"


def load_or_train_logistic_regression(X_train, y_train, config, models_dir):
    model_path = os.path.join(models_dir, 'logistic_regression.joblib')
    scaler_path = os.path.join(models_dir, 'logistic_regression_scaler.joblib')
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        log_progress(f"Logistic Regression model found, loading instead of retraining...")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler, 0, "loaded from disk"

    log_progress("Training Logistic Regression on CPU...")
    cfg = config['logistic_regression']

    log_progress("  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    params = {
        'C': cfg['C'],
        'max_iter': cfg['max_iter'],
        'class_weight': cfg['class_weight'],
        'random_state': config['random_state'],
        'solver': cfg.get('solver', 'lbfgs'),
    }

    start = time.time()
    model = LogisticRegression(**params)
    model.fit(X_train_scaled, y_train)
    log_elapsed(start, "Logistic Regression training")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    log_progress(f"Saved to {model_path}")
    return model, scaler, time.time() - start, "CPU"


def load_or_train_svm(X_train, y_train, X_test, y_test, config, models_dir):
    """Train SVM with FIXED best hyperparameters (no grid search).

    The previous grid search (partial) found these best params:
      C=1, gamma=0.01 -> AUC-PR=0.8809 (best observed)

    We use these directly. No grid search.
    """
    model_path = os.path.join(models_dir, 'svm.joblib')
    scaler_path = os.path.join(models_dir, 'svm_scaler.joblib')
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        log_progress(f"SVM model found at {model_path}, loading instead of retraining...")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler, {'C': 1, 'gamma': 0.01}, [], 0, "loaded from disk"

    log_progress("Training SVM with fixed hyperparameters (C=1, gamma=0.01)...")
    log_progress("  NOTE: SVM is CPU-only and may take 10-15 minutes.")
    log_progress("  NOTE: Training on 50,000 sub-sampled examples for tractability.")

    # Sub-sample training data (class-balanced)
    max_samples = config['svm'].get('max_samples', 50000)
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    n_per_class = max_samples // 2
    n_pos_take = min(n_per_class, n_pos)
    n_neg_take = min(n_per_class, n_neg)
    rng = np.random.RandomState(config['random_state'])
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]
    pos_selected = rng.choice(pos_indices, size=n_pos_take, replace=False)
    neg_selected = rng.choice(neg_indices, size=n_neg_take, replace=False)
    selected = np.concatenate([pos_selected, neg_selected])
    rng.shuffle(selected)
    X_sub = X_train[selected]
    y_sub = y_train[selected]
    log_progress(f"  Sub-sampled: {len(selected):,} examples ({n_pos_take:,} pos, {n_neg_take:,} neg)")

    # Scale features
    log_progress("  Scaling features...")
    scaler = StandardScaler()
    X_sub_scaled = scaler.fit_transform(X_sub)

    # Train SVM with fixed hyperparameters
    log_progress("  Training SVM (please wait, this is the slowest model)...")
    start = time.time()
    model = SVC(
        C=1,
        gamma=0.01,
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=config['random_state'],
    )
    model.fit(X_sub_scaled, y_sub)
    elapsed = time.time() - start
    log_elapsed(start, "SVM training")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    log_progress(f"Saved to {model_path}")
    return model, scaler, {'C': 1, 'gamma': 0.01}, [], elapsed, "CPU"


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model_predictions(probs, y_test, test_df, config, model_name):
    """Evaluate model probabilities on test set with progress indicators."""
    log_progress(f"Evaluating {model_name}...")

    # Apply EMA per session
    log_progress("  Applying EMA smoothing per session...")
    session_ids = test_df['Session_ID'].values
    timestamps = test_df['Timestamp'].values
    alpha = config['post_processing']['ema_alpha']
    probs_ema = apply_ema_per_session(probs, session_ids, timestamps, alpha)

    # Threshold sweep
    log_progress("  Sweeping thresholds (0.01 to 0.90, 90 values)...")
    thr_min = config['threshold']['sweep_min']
    thr_max = config['threshold']['sweep_max']
    thr_step = config['threshold']['sweep_step']
    beta = config['threshold']['beta']
    thresholds = np.arange(thr_min, thr_max + thr_step/2, thr_step)

    subjects_in_test = sorted(test_df['Subject_ID'].unique())
    subject_indices = {s: (test_df['Subject_ID'].values == s) for s in subjects_in_test}

    best_overall_f2 = -1
    best_threshold = None
    sweep_results = []

    n_thresholds = len(thresholds)
    for i, thr in enumerate(thresholds):
        if (i + 1) % 30 == 0:
            log_progress(f"    Threshold sweep progress: {i+1}/{n_thresholds}")

        binary_pred = (probs_ema >= thr).astype(int)

        overall_f2 = fbeta_score(y_test, binary_pred, beta=beta, zero_division=0)
        overall_precision = precision_score(y_test, binary_pred, zero_division=0)
        overall_recall = recall_score(y_test, binary_pred, zero_division=0)

        per_subject = {}
        for s in subjects_in_test:
            idx = subject_indices[s]
            if len(np.unique(y_test[idx])) < 2:
                per_subject[s] = {'f2': 0.0, 'precision': 0.0, 'recall': 0.0}
            else:
                per_subject[s] = {
                    'f2': float(fbeta_score(y_test[idx], binary_pred[idx], beta=beta, zero_division=0)),
                    'precision': float(precision_score(y_test[idx], binary_pred[idx], zero_division=0)),
                    'recall': float(recall_score(y_test[idx], binary_pred[idx], zero_division=0)),
                }

        sweep_results.append({
            'threshold': float(thr),
            'overall_f2': float(overall_f2),
            'overall_precision': float(overall_precision),
            'overall_recall': float(overall_recall),
            'per_subject': per_subject,
        })

        if overall_f2 > best_overall_f2:
            best_overall_f2 = overall_f2
            best_threshold = float(thr)

    best_result = next(r for r in sweep_results if r['threshold'] == best_threshold)
    auc_pr = float(average_precision_score(y_test, probs_ema))

    log_progress(f"  Best F2: {best_overall_f2:.4f} at threshold {best_threshold:.2f}")
    log_progress(f"  AUC-PR: {auc_pr:.4f}")

    return {
        'best_threshold': best_threshold,
        'best_overall_f2': float(best_overall_f2),
        'best_overall_precision': float(best_result['overall_precision']),
        'best_overall_recall': float(best_result['overall_recall']),
        'per_subject_at_best_threshold': best_result['per_subject'],
        'auc_pr': auc_pr,
    }


def compute_ensemble_weights(individual_results):
    """Performance-weighted ensemble (Option C)."""
    weights = {}
    total_f2 = sum(r['best_overall_f2'] for r in individual_results.values())
    for model_name, result in individual_results.items():
        f2 = result['best_overall_f2']
        weight = f2 / total_f2 if total_f2 > 0 else 1.0 / len(individual_results)
        weights[model_name] = float(weight)
    return weights


# ============================================================================
# DOCUMENTATION
# ============================================================================

def save_documentation(config, individual_results, ensemble_result, ensemble_weights,
                        training_times, run_timestamp, run_dir, train_df, test_df,
                        svm_best_params, devices_used):
    log_progress("Saving documentation...")

    metrics = {
        'run_timestamp': run_timestamp,
        'configuration': {
            'training_subjects': config['subjects']['training'],
            'test_subjects': config['subjects']['testing'],
            'test_cap_frames_per_subject': config['test_cap_frames_per_subject'],
            'n_features': len(config['feature_columns']),
            'random_state': config['random_state'],
            'ema_alpha': config['post_processing']['ema_alpha'],
        },
        'data': {
            'n_train_frames': len(train_df),
            'n_test_frames': len(test_df),
            'train_class_balance': float(train_df['y_binary'].mean()),
            'test_class_balance': float(test_df['y_binary'].mean()),
        },
        'individual_models': {
            name: {
                'best_threshold': r['best_threshold'],
                'best_overall_f2': r['best_overall_f2'],
                'best_overall_precision': r['best_overall_precision'],
                'best_overall_recall': r['best_overall_recall'],
                'per_subject_at_best_threshold': r['per_subject_at_best_threshold'],
                'auc_pr': r['auc_pr'],
                'training_time_sec': training_times.get(name, 0),
                'device': devices_used.get(name, 'unknown'),
            }
            for name, r in individual_results.items()
        },
        'ensemble': {
            'method': 'performance_weighted',
            'weights': ensemble_weights,
            'best_threshold': ensemble_result['best_threshold'],
            'best_overall_f2': ensemble_result['best_overall_f2'],
            'best_overall_precision': ensemble_result['best_overall_precision'],
            'best_overall_recall': ensemble_result['best_overall_recall'],
            'per_subject_at_best_threshold': ensemble_result['per_subject_at_best_threshold'],
            'auc_pr': ensemble_result['auc_pr'],
        },
        'svm': {
            'best_params': svm_best_params,
            'note': 'Used fixed hyperparameters from partial grid search results. '
                    'C=1, gamma=0.01 gave AUC-PR=0.8809, the best observed in the partial grid.',
        },
    }

    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # run_log.md (human-readable)
    test_subjects = sorted(test_df['Subject_ID'].unique())
    log_lines = [
        f"# Approach 03 Run Log",
        f"",
        f"**Run timestamp:** {run_timestamp}",
        f"",
        f"## Configuration",
        f"",
        f"- Training subjects: {', '.join(config['subjects']['training'])}",
        f"- Test subjects: {', '.join(config['subjects']['testing'])}",
        f"- Test cap per subject: {config['test_cap_frames_per_subject']:,} frames",
        f"- Feature count: {len(config['feature_columns'])}",
        f"- Random state: {config['random_state']}",
        f"- EMA alpha: {config['post_processing']['ema_alpha']}",
        f"- Ensemble method: performance-weighted (Option C)",
        f"",
        f"## Data Summary",
        f"",
        f"- Training: {len(train_df):,} frames ({train_df['y_binary'].sum():,} positive, {(len(train_df)-train_df['y_binary'].sum()):,} negative)",
        f"- Test: {len(test_df):,} frames ({test_df['y_binary'].sum():,} positive, {(len(test_df)-test_df['y_binary'].sum()):,} negative)",
        f"- Train class balance: {train_df['y_binary'].mean()*100:.1f}% positive",
        f"- Test class balance: {test_df['y_binary'].mean()*100:.1f}% positive",
        f"",
        f"## Individual Model Results",
        f"",
        f"| Model | Overall F2 | Best Threshold | AUC-PR | Training Time (s) | Device |",
        f"|---|---|---|---|---|---|",
    ]
    for name, r in individual_results.items():
        log_lines.append(
            f"| {name} | {r['best_overall_f2']:.4f} | {r['best_threshold']:.2f} | "
            f"{r['auc_pr']:.4f} | {training_times.get(name, 0):.1f} | {devices_used.get(name, 'unknown')} |"
        )

    log_lines.append("")
    log_lines.append("## Per-Subject F2 (at each model's best overall threshold)")
    log_lines.append("")
    header = "| Model | " + " | ".join(test_subjects) + " |"
    separator = "|---|" + "---|" * len(test_subjects)
    log_lines.append(header)
    log_lines.append(separator)
    for name, r in individual_results.items():
        row = f"| {name} "
        for s in test_subjects:
            row += f"| {r['per_subject_at_best_threshold'][s]['f2']:.4f} "
        row += "|"
        log_lines.append(row)

    log_lines.append("")
    log_lines.append("## Ensemble (Performance-Weighted)")
    log_lines.append("")
    log_lines.append("| Model | Weight |")
    log_lines.append("|---|---|")
    for name, w in ensemble_weights.items():
        log_lines.append(f"| {name} | {w:.4f} |")
    log_lines.append("")
    log_lines.append(f"- **Overall F2: {ensemble_result['best_overall_f2']:.4f}**")
    log_lines.append(f"- Best threshold: {ensemble_result['best_threshold']:.2f}")
    log_lines.append(f"- AUC-PR: {ensemble_result['auc_pr']:.4f}")
    log_lines.append("")
    log_lines.append("### Ensemble Per-Subject F2")
    log_lines.append("")
    log_lines.append("| Subject | F2 | Precision | Recall |")
    log_lines.append("|---|---|---|---|")
    for s, m in ensemble_result['per_subject_at_best_threshold'].items():
        log_lines.append(f"| {s} | {m['f2']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} |")

    log_lines.append("")
    log_lines.append("## SVM Hyperparameter Note")
    log_lines.append("")
    log_lines.append("**No full grid search was performed in this run.** A partial grid search")
    log_lines.append("(interrupted due to time) found that C=1, gamma=0.01 gave AUC-PR=0.8809,")
    log_lines.append("the best observed. We use these hyperparameters directly.")
    log_lines.append("")
    log_lines.append("The partial grid search results (for reference):")
    log_lines.append("")
    log_lines.append("| C | gamma | AUC-PR |")
    log_lines.append("|---|---|---|")
    log_lines.append("| 0.1 | 0.001 | 0.8397 |")
    log_lines.append("| 0.1 | 0.01 | 0.8799 |")
    log_lines.append("| 0.1 | 0.1 | 0.7930 |")
    log_lines.append("| 0.1 | 1 | 0.7569 |")
    log_lines.append("| 1 | 0.001 | 0.8654 |")
    log_lines.append("| 1 | 0.01 | **0.8809** (best) |")
    log_lines.append("| 1 | 0.1 | 0.8109 |")
    log_lines.append("| 1 | 1 | 0.5101 |")
    log_lines.append("| 10 | 0.001 | 0.8793 |")
    log_lines.append("| 10 | 0.01 | 0.8740 |")
    log_lines.append("| 10 | 0.1 | 0.8087 |")
    log_lines.append("| 10 | 1 | 0.7580 |")
    log_lines.append("| 100 | 0.001 | 0.8650 |")
    log_lines.append("| 100 | 0.01 | 0.8556 |")
    log_lines.append("| 100 | 0.1 | 0.8043 |")

    with open(os.path.join(run_dir, 'run_log.md'), 'w') as f:
        f.write('\n'.join(log_lines))


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PROCTORIFY - APPROACH 03: FOUR-MODEL HOLD-OUT EVALUATION")
    print("=" * 70)

    config = load_config()
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_progress(f"Run timestamp: {run_timestamp}")

    # Setup directories
    results_dir = os.path.join(SCRIPT_DIR, config['data']['results_dir'])
    models_dir = os.path.join(SCRIPT_DIR, config['data']['models_dir'])
    run_dir = os.path.join(results_dir, f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # STEP 1: Load data
    log_step(1, 6, "LOAD AND SPLIT DATA")
    df = load_data(config)
    df = prepare_labels(df, config)
    train_df, test_df = split_train_test(df, config)
    X_train, y_train = get_xy(train_df, config)
    X_test, y_test = get_xy(test_df, config)

    individual_results = {}
    model_probs = {}
    training_times = {}
    devices_used = {}
    svm_best_params = None

    # STEP 2: Train XGBoost
    log_step(2, 6, "XGBOOST")
    xgb_model, xgb_time, xgb_device = load_or_train_xgboost(X_train, y_train, config, models_dir)
    log_progress("Computing XGBoost predictions on test set...")
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_result = evaluate_model_predictions(xgb_probs, y_test, test_df, config, "XGBoost")
    individual_results['xgboost'] = xgb_result
    model_probs['xgboost'] = xgb_probs
    training_times['xgboost'] = xgb_time
    devices_used['xgboost'] = xgb_device

    # STEP 3: Train Random Forest
    log_step(3, 6, "RANDOM FOREST")
    rf_model, rf_time, rf_device = load_or_train_random_forest(X_train, y_train, config, models_dir)
    log_progress("Computing Random Forest predictions on test set...")
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_result = evaluate_model_predictions(rf_probs, y_test, test_df, config, "Random Forest")
    individual_results['random_forest'] = rf_result
    model_probs['random_forest'] = rf_probs
    training_times['random_forest'] = rf_time
    devices_used['random_forest'] = rf_device

    # STEP 4: Train Logistic Regression
    log_step(4, 6, "LOGISTIC REGRESSION")
    lr_model, lr_scaler, lr_time, lr_device = load_or_train_logistic_regression(X_train, y_train, config, models_dir)
    log_progress("Computing Logistic Regression predictions on test set...")
    lr_probs = lr_model.predict_proba(lr_scaler.transform(X_test))[:, 1]
    lr_result = evaluate_model_predictions(lr_probs, y_test, test_df, config, "Logistic Regression")
    individual_results['logistic_regression'] = lr_result
    model_probs['logistic_regression'] = lr_probs
    training_times['logistic_regression'] = lr_time
    devices_used['logistic_regression'] = lr_device

    # STEP 5: Train SVM (with fixed hyperparameters)
    log_step(5, 6, "SVM (FIXED HYPERPARAMETERS)")
    svm_model, svm_scaler, svm_best_params, _, svm_time, svm_device = load_or_train_svm(
        X_train, y_train, X_test, y_test, config, models_dir
    )
    log_progress("Computing SVM predictions on test set...")
    svm_probs = svm_model.predict_proba(svm_scaler.transform(X_test))[:, 1]
    svm_result = evaluate_model_predictions(svm_probs, y_test, test_df, config, "SVM")
    individual_results['svm'] = svm_result
    model_probs['svm'] = svm_probs
    training_times['svm'] = svm_time
    devices_used['svm'] = svm_device

    # STEP 6: Ensemble
    log_step(6, 6, "ENSEMBLE (PERFORMANCE-WEIGHTED)")
    log_progress("Computing ensemble weights (Option C: performance-weighted)...")
    ensemble_weights = compute_ensemble_weights(individual_results)
    log_progress("Ensemble weights:")
    for name, w in ensemble_weights.items():
        log_progress(f"  {name}: {w:.4f}")

    with open(os.path.join(models_dir, 'ensemble_weights.json'), 'w') as f:
        json.dump(ensemble_weights, f, indent=2)
    log_progress(f"Saved weights to {os.path.join(models_dir, 'ensemble_weights.json')}")

    log_progress("Computing ensemble probabilities...")
    ensemble_probs = np.zeros(len(y_test))
    for name, probs in model_probs.items():
        ensemble_probs += ensemble_weights[name] * probs

    log_progress("Evaluating ensemble...")
    ensemble_result = evaluate_model_predictions(ensemble_probs, y_test, test_df, config, "Ensemble")

    # Save documentation
    log_progress("Saving documentation...")
    save_documentation(
        config, individual_results, ensemble_result, ensemble_weights,
        training_times, run_timestamp, run_dir, train_df, test_df,
        svm_best_params, devices_used
    )

    # FINAL SUMMARY
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\nIndividual model F2 scores:")
    for name, result in individual_results.items():
        print(f"  {name:25s}: overall F2={result['best_overall_f2']:.4f}, "
              f"threshold={result['best_threshold']:.2f}")
        for s, metrics in result['per_subject_at_best_threshold'].items():
            print(f"    {s:15s}: F2={metrics['f2']:.4f}, "
                  f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")

    print(f"\nEnsemble (performance-weighted):")
    print(f"  Weights: {ensemble_weights}")
    print(f"  Overall F2={ensemble_result['best_overall_f2']:.4f}, "
          f"threshold={ensemble_result['best_threshold']:.2f}")
    for s, metrics in ensemble_result['per_subject_at_best_threshold'].items():
        print(f"    {s:15s}: F2={metrics['f2']:.4f}, "
              f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")

    print(f"\nResults saved to: {run_dir}")
    print(f"Models saved to: {models_dir}")


if __name__ == "__main__":
    main()