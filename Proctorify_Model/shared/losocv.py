"""
Proctorify LOSOCV Utilities
============================
Shared utility functions for Leave-One-Subject-Out Cross-Validation
evaluation of proctoring models.

Used by training scripts in approaches/ folders.

Functions:
  - apply_ema_per_session: Smooth probability time series per session
  - apply_persistence_gate_per_session: Require sustained suspicion before alerting
  - compute_f2: F2 score (recall-weighted F-beta)
  - compute_auc_pr: Area under precision-recall curve
  - sweep_thresholds: Find best threshold for given probabilities
  - sweep_weights_and_thresholds: Joint weight + threshold optimization

All per-session functions respect session boundaries — EMA and persistence
gate are computed within a session, never across sessions. This is critical
because mixing data across sessions would leak information between them.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, average_precision_score


def apply_ema_per_session(probabilities, session_ids, timestamps, alpha):
    """
    Apply Exponential Moving Average to probabilities within each session.
    
    EMA formula: ema_t = alpha * prob_t + (1 - alpha) * ema_{t-1}
    
    The first frame of each session initializes the EMA (no smoothing).
    Sessions are sorted by timestamp before applying EMA to ensure correct
    temporal order, regardless of how the dataframe was ordered.
    
    Args:
        probabilities: np.array of model probabilities (one per frame)
        session_ids: np.array of session identifiers (one per frame)
        timestamps: np.array of timestamps (one per frame)
        alpha: float, EMA smoothing factor (0 = no update, 1 = no smoothing)
    
    Returns:
        np.array of EMA-smoothed probabilities (same shape as input)
    """
    probs = probabilities.copy().astype(float)
    ema = np.full_like(probs, np.nan)
    
    unique_sessions = pd.Series(session_ids).unique()
    
    for session in unique_sessions:
        mask = session_ids == session
        session_indices = np.where(mask)[0]
        
        # Sort by timestamp to ensure correct temporal order
        session_timestamps = timestamps[session_indices]
        sort_order = np.argsort(session_timestamps)
        sorted_indices = session_indices[sort_order]
        
        # Apply EMA in temporal order
        prev_ema = None
        for idx in sorted_indices:
            if prev_ema is None:
                # First frame of session: initialize with raw probability
                ema[idx] = probs[idx]
                prev_ema = probs[idx]
            else:
                new_ema = alpha * probs[idx] + (1 - alpha) * prev_ema
                ema[idx] = new_ema
                prev_ema = new_ema
    
    return ema


def apply_persistence_gate_per_session(binary_predictions, session_ids, timestamps, gate_seconds):
    """
    Apply persistence gate: an alert fires only if the binary prediction
    has been 1 for at least gate_seconds continuously within a session.
    
    This prevents momentary flickering above threshold from triggering alerts.
    A student must exhibit sustained suspicious behavior (default 2 seconds)
    before the system flags them.
    
    Args:
        binary_predictions: np.array of 0/1 predictions per frame
        session_ids: np.array of session identifiers per frame
        timestamps: np.array of timestamps per frame
        gate_seconds: float, minimum sustained duration for an alert
    
    Returns:
        np.array of 0/1 gated predictions (same shape as input)
    """
    gated = np.zeros_like(binary_predictions)
    unique_sessions = pd.Series(session_ids).unique()
    
    for session in unique_sessions:
        mask = session_ids == session
        session_indices = np.where(mask)[0]
        
        # Sort by timestamp
        session_timestamps = timestamps[session_indices]
        sort_order = np.argsort(session_timestamps)
        sorted_indices = session_indices[sort_order]
        
        # Process in temporal order
        n = len(sorted_indices)
        i = 0
        while i < n:
            idx = sorted_indices[i]
            if binary_predictions[idx] == 1:
                # Start of a potential sustained run
                run_start = i
                run_start_time = timestamps[sorted_indices[i]]
                
                # Find end of this run
                j = i
                while j < n and binary_predictions[sorted_indices[j]] == 1:
                    j += 1
                
                # Run is sorted_indices[run_start:j]
                run_end_time = timestamps[sorted_indices[j-1]]
                run_duration = run_end_time - run_start_time
                
                if run_duration >= gate_seconds:
                    # Mark only frames where sustained condition is met
                    for k in range(run_start, j):
                        frame_time = timestamps[sorted_indices[k]]
                        if frame_time - run_start_time >= gate_seconds:
                            gated[sorted_indices[k:j]] = 1
                            break
                
                i = j
            else:
                i += 1
    
    return gated


def compute_f2(y_true, y_pred):
    """
    Compute F2 score (F-beta with beta=2).
    F2 weights recall higher than precision, suitable for proctoring
    where catching cheaters is more important than avoiding false alarms.
    """
    if len(np.unique(y_true)) < 2:
        # If only one class present in y_true, F2 is undefined
        return 0.0
    return fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)


def compute_auc_pr(y_true, y_prob):
    """
    Compute Area Under Precision-Recall Curve.
    AUC-PR is threshold-independent and more informative than AUC-ROC
    for imbalanced datasets.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0
    return average_precision_score(y_true, y_prob)


def sweep_thresholds(ensemble_probs, session_ids, timestamps, y_true,
                      threshold_min, threshold_max, threshold_step,
                      gate_seconds, beta=2.0):
    """
    Sweep thresholds and return the F2 score at each threshold.
    
    For each threshold:
      1. Convert probabilities to binary predictions
      2. Apply persistence gate per session
      3. Compute F2 across all frames
    
    Args:
        ensemble_probs: np.array of ensemble probabilities per frame
        session_ids, timestamps, y_true: per-frame arrays
        threshold_min, threshold_max, threshold_step: threshold sweep range
        gate_seconds: persistence gate duration
        beta: F-beta parameter (default 2.0 for F2)
    
    Returns:
        list of dicts: [{threshold, f2, precision, recall}, ...]
    """
    thresholds = np.arange(threshold_min, threshold_max + threshold_step/2, threshold_step)
    results = []
    
    for thr in thresholds:
        # Binary prediction: 1 if probability >= threshold
        binary_pred = (ensemble_probs >= thr).astype(int)
        
        # Apply persistence gate per session
        gated_pred = apply_persistence_gate_per_session(
            binary_pred, session_ids, timestamps, gate_seconds
        )
        
        # Compute metrics
        from sklearn.metrics import precision_score, recall_score
        precision = precision_score(y_true, gated_pred, zero_division=0)
        recall = recall_score(y_true, gated_pred, zero_division=0)
        f_beta = fbeta_score(y_true, gated_pred, beta=beta, zero_division=0)
        
        results.append({
            'threshold': float(thr),
            'f2': float(f_beta),
            'precision': float(precision),
            'recall': float(recall),
        })
    
    return results


def sweep_weights_and_thresholds(xgb_probs, rf_probs, session_ids, timestamps, y_true,
                                   weight_min, weight_max, weight_step,
                                   threshold_min, threshold_max, threshold_step,
                                   ema_alpha, gate_seconds, beta=2.0):
    """
    Joint sweep over ensemble weights and decision thresholds.
    
    For each (weight, threshold) combination:
      1. Compute ensemble probability: weight * xgb_prob + (1-weight) * rf_prob
      2. Apply EMA per session (EMA is linear, so equivalent to applying EMA on
         each model's probs first, but computing per weight is clearer here)
      3. Convert to binary using threshold
      4. Apply persistence gate per session
      5. Compute F2 across all frames
    
    Returns the full grid plus the best (weight, threshold) by F2.
    
    Note: For efficiency, EMA is applied to each model's probabilities once
    (outside this function), and the ensemble is computed on EMA-smoothed
    probabilities. This is mathematically equivalent because EMA is linear.
    
    Args:
        xgb_probs, rf_probs: per-frame probabilities from each model
        session_ids, timestamps, y_true: per-frame arrays
        weight_min/max/step: ensemble weight sweep (weight = XGB share)
        threshold_min/max/step: threshold sweep
        ema_alpha: EMA smoothing factor
        gate_seconds: persistence gate duration
        beta: F-beta parameter
    
    Returns:
        dict with:
          'grid': list of {weight, threshold, f2, precision, recall}
          'best': {weight, threshold, f2, precision, recall}
    """
    weights = np.arange(weight_min, weight_max + weight_step/2, weight_step)
    thresholds = np.arange(threshold_min, threshold_max + threshold_step/2, threshold_step)
    
    grid = []
    best = {'f2': -1.0}
    
    for w in weights:
        # Ensemble probability
        ensemble_prob = w * xgb_probs + (1.0 - w) * rf_probs
        
        for thr in thresholds:
            # Binary prediction
            binary_pred = (ensemble_prob >= thr).astype(int)
            
            # Apply persistence gate
            gated_pred = apply_persistence_gate_per_session(
                binary_pred, session_ids, timestamps, gate_seconds
            )
            
            # Compute metrics
            from sklearn.metrics import precision_score, recall_score
            precision = precision_score(y_true, gated_pred, zero_division=0)
            recall = recall_score(y_true, gated_pred, zero_division=0)
            f_beta = fbeta_score(y_true, gated_pred, beta=beta, zero_division=0)
            
            entry = {
                'weight': float(w),
                'threshold': float(thr),
                'f2': float(f_beta),
                'precision': float(precision),
                'recall': float(recall),
            }
            grid.append(entry)
            
            if f_beta > best['f2']:
                best = entry.copy()
    
    return {'grid': grid, 'best': best}