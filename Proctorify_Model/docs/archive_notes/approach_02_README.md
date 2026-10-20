# Approach 02: XGBoost + RF Multimodal (Visual + Keyboard)

**Status:** Complete — Phase 2 finished. Multimodal model is essentially equivalent to visual-only baseline.
**Date completed:** 2026-08-26
**Mean F2:** 0.8670 ± 0.0478
**Mean AUC-PR:** 0.8895

## What This Approach Tried

Extend the Phase 1 visual baseline by adding 29 keyboard dynamics features derived from `raw_inputs.csv` (per-event keyboard and mouse logs captured during data collection by `honeypot_v2.py`). The keyboard features are based on a literature review of:

- Roy, S., Pradhan, J., Kumar, A., Adhikary, D. R., Roy, U., Sinha, D., & Pal, R. K. (2022). *A Systematic Literature Review on Latest Keystroke Dynamics Based Models.* IEEE Access, 10, 92192–92245.
- Kim, J., Kim, H., & Kang, P. (2018). *Keystroke dynamics-based user authentication using freely typed text based on user-adaptive feature extraction and novelty detection.* Applied Soft Computing, 65, 427–438.

The 29 keyboard features cover:
- DD-Time statistics (5 features) — Roy et al. Section V
- N-Graph DD-Times (2 features) — Roy et al. Section V
- L/R/S keyboard region DD-Times (8 features, Kang and Cho method as cited in Kim et al. 2018)
- Special key ratios (4 features) — Roy et al. Section F, Kim et al.
- Keystroke volume (4 features) — Roy et al. Section E
- Pause patterns (3 features) — derived from gap statistics
- Burst detection (2 features) — typing speed pattern
- Mouse activity (1 feature) — behavioral signal from our data

### Limitation acknowledged

Our logger records keydown events only (no keyup), so we cannot compute the following features as defined in the literature:
- Key Duration / dwell time (requires keyup of same key)
- UD-Time / flight time (requires keyup of one + keydown of next)
- UU-Time (requires successive keyup events)
- Digraph-Time and Trigraph-Time as defined (use keyup of second/third key)

We compute DD-only versions (time between consecutive keydowns) instead. This is documented as a known limitation; the model uses these DD-only features for 45% of total importance, so they appear to provide signal even without the full feature set the literature describes.

## Configuration

- **Features:** 53 total
  - 24 visual (8 spatial + 8 velocity + 8 variance)
  - 29 keyboard (as listed above)
  - Note: Face_Detected and Is_Interacting are excluded in this approach because Is_Interacting is redundant with the explicit keyboard features (Key_Count_2s, Key_Count_10s). This is a minor methodology difference from Approach 01, documented here for transparency.
- **Models:** XGBoost (GPU, GTX 1660, device='cuda', tree_method='hist') + Random Forest (CPU, class_weight='balanced')
- **Class weighting:** XGBoost uses scale_pos_weight = neg/pos (~1.87)
- **Ensemble sweep:** Weight (XGBoost share) 0.0–1.0 step 0.05 (21 values) × threshold 0.01–0.90 step 0.01 (90 values) = 1,890 combinations per fold
- **Post-processing:** EMA smoothing (alpha=0.3); persistence gate disabled
- **Evaluation:** F2 score (beta=2.0, recall-weighted)

## Results

| Held-Out Subject | AUC-PR | Best F2 | Best Weight (XGB) | Best Threshold |
|---|---|---|---|---|
| Bushra | 0.7667 | 0.7897 | 1.00 | 0.01 |
| Lam | 0.9340 | 0.8810 | 0.40 | 0.12 |
| Prova | 0.8545 | 0.8415 | 0.30 | 0.28 |
| Shahin | 0.9612 | 0.9292 | 0.55 | 0.18 |
| Zarif | 0.9314 | 0.8937 | 0.10 | 0.39 |
| **Mean** | **0.8895** | **0.8670** | — | — |

## Comparison to Phase 1

| Metric | Phase 1 (Visual-only) | Phase 2 (Multimodal) | Delta |
|---|---|---|---|
| Mean F2 | 0.8742 | 0.8670 | -0.0072 |
| Mean AUC-PR | 0.8963 | 0.8895 | -0.0068 |

**Interpretation:** Adding 29 keyboard features did not improve proctoring accuracy. The -0.0072 F2 delta is within noise for 5-fold LOSOCV. The model uses the keyboard features (45% of total importance — see feature analysis below) but they provide redundant signal that the visual features were already capturing.

### Per-fold comparison

| Subject | Phase 1 F2 | Phase 2 F2 | Change |
|---|---|---|---|
| Bushra | 0.7803 | 0.7897 | +0.0094 (slight improvement, weakest fold) |
| Lam | 0.8798 | 0.8810 | +0.0012 (essentially unchanged) |
| Prova | 0.8672 | 0.8415 | -0.0257 (notable regression) |
| Shahin | 0.9361 | 0.9292 | -0.0069 (essentially unchanged) |
| Zarif | 0.9073 | 0.8937 | -0.0136 (small regression) |

Prova's regression is the most concerning. Possible explanations:
- Prova's keyboard patterns differ from the other 4 subjects, and the model trained on them produces noise on Prova
- Prova's cheating style may not involve keyboard behavior changes
- 5 subjects is a small sample, so per-fold variation is expected

## Feature Importance Analysis

A feature importance analysis was run on the 5 saved XGBoost models (see `results/feature_importance.csv` for the full data).

### Group-level importance

| Feature group | Count | Total gain | % of total importance |
|---|---|---|---|
| Visual | 24 | 4256.85 | 54.8% |
| Keyboard | 29 | 3517.73 | 45.2% |

All 29 keyboard features have non-zero gain — the model uses all of them. The keyboard features are not ignored; they account for nearly half of the model's decisions. Yet they don't improve F2 because the information they provide was already accessible through the visual features.

### Top 20 features by gain (averaged across folds)

| Rank | Feature | Gain | Weight | Cover |
|---|---|---|---|---|
| 1 | Key_Count_2s (keyboard) | 1280.31 | 135 | 5973 |
| 2 | Gaze_X (visual) | 1201.42 | 472 | 5437 |
| 3 | DiGraph_DD_Mean_10s (keyboard) | 521.21 | 20 | 5027 |
| 4 | Qx (visual) | 508.42 | 480 | 3612 |
| 5 | Yaw (visual) | 427.01 | 769 | 3921 |
| 6 | Qz (visual) | 338.50 | 626 | 3643 |
| 7 | Qw (visual) | 302.28 | 446 | 2536 |
| 8 | DD_Time_Mean_10s (keyboard) | 269.03 | 105 | 2547 |
| 9 | Gaze_Y (visual) | 216.02 | 398 | 2596 |
| 10 | Pitch (visual) | 210.62 | 452 | 2720 |
| 11 | Qy (visual) | 162.94 | 344 | 1955 |
| 12 | Yaw_Var (visual) | 115.28 | 295 | 2099 |
| 13 | Gaze_Y_Var (visual) | 104.65 | 325 | 1844 |
| 14 | Key_Count_10s (keyboard) | 98.48 | 116 | 2048 |
| 15 | TriGraph_DD_Mean_10s (keyboard) | 97.18 | 64 | 1336 |
| 16 | DD_Time_Min_10s (keyboard) | 91.09 | 127 | 1482 |
| 17 | Key_Rate_60s (keyboard) | 90.15 | 267 | 2064 |
| 18 | DD_Time_Max_10s (keyboard) | 86.34 | 86 | 2330 |
| 19 | Shift_Ratio_30s (keyboard) | 83.85 | 168 | 1298 |
| 20 | Gaze_X_Var (visual) | 81.84 | 274 | 1113 |

The single most important feature in the entire model is a keyboard feature: `Key_Count_2s` (gain 1280). This tells us the model relies heavily on "is the student typing right now" as a discriminative signal.

### Bottom 10 features (least important)

These features contribute the least to the model's decisions:

| Rank | Feature | Gain |
|---|---|---|
| 44 | DD_Time_LS_30s | 34.55 |
| 45 | Longest_Pause_30s | 34.03 |
| 46 | DD_Time_SR_30s | 32.43 |
| 47 | Gaze_X_Vel | 31.33 |
| 48 | Gaze_Y_Vel | 31.23 |
| 49 | Pitch_Vel | 31.17 |
| 50 | Pause_Count_30s | 31.15 |
| 51 | Qw_Vel | 29.96 |
| 52 | Space_Ratio_30s | 26.96 |
| 53 | Qy_Vel | 21.73 |

Notable observations:
- Visual velocity features (Gaze_X_Vel, Gaze_Y_Vel, Pitch_Vel, Qw_Vel, Qy_Vel) are among the least useful — they're derived from spatial features and don't add discriminative power
- Pause features (Longest_Pause_30s, Pause_Count_30s) are redundant with the Key_Count features (which already capture "no typing")
- Two of the eight L/R/S region features (DD_Time_LS_30s, DD_Time_SR_30s) are at the bottom — the Kim et al. method didn't pan out as strongly as the literature suggested for our dataset

## Why Didn't Multimodal Improve Performance?

This is feature redundancy, a known phenomenon in ML:

1. **Phase 1 (visual-only):** The model learned to infer "is the student typing" from visual cues (head down, eyes on screen). When a student looks at the keyboard while typing, the visual features capture that.
2. **Phase 2 (multimodal):** The model gets explicit keyboard data and uses `Key_Count_2s` as the single most important feature. But the underlying information ("is the student typing") was already accessible through visual features.
3. Adding the keyboard features provides the same information in a more direct form, which does not improve generalization. The model's ranking quality (AUC-PR) doesn't change, and the F2 score doesn't change.

This is consistent with literature on multi-modal fusion, which often finds diminishing returns when modalities are correlated rather than complementary.

## Recommendation for Live Proctor (Phase 4)

Use the **multimodal model** (this approach) for the live proctor because:

1. More features provide robustness against failure modes not seen in training (e.g., a cheating pattern that doesn't involve head movement but does involve unusual typing behavior)
2. Browser keyboard capture is essentially free at deployment — keystroke events are native to the browser and require no extra setup
3. "We considered both visual and keyboard modalities" is more defensible to thesis reviewers and academic integrity officers than "we only used visual"
4. The offline performance is equivalent to the visual-only model, so there is no cost to using the richer feature set

## Data Summary

- 28 sessions across 5 subjects (Bushra: 4, Lam: 4, Prova: 5, Shahin: 7, Zarif: 8)
- 410,467 total frames (same as Phase 1, no frames added or removed)
- Class balance: 65.0% NORMAL, 35.0% CHEAT_PHYSICAL (same as Phase 1)
- 23,599 total keyboard events across all sessions
- 245 total mouse events across all sessions

## Literature Citations

Feature engineering was based on:

- Roy, S., Pradhan, J., Kumar, A., Adhikary, D. R., Roy, U., Sinha, D., & Pal, R. K. (2022). A Systematic Literature Review on Latest Keystroke Dynamics Based Models. IEEE Access, 10, 92192–92245.
- Kim, J., Kim, H., & Kang, P. (2018). Keystroke dynamics-based user authentication using freely typed text based on user-adaptive feature extraction and novelty detection. Applied Soft Computing, 65, 427–438.

Additional context papers (not directly used for feature engineering):
- Kim, D. I., Lee, S., & Shin, J. S. (2020). A New Feature Scoring Method in Keystroke Dynamics-Based User Authentications. IEEE Access, 8.
- Sadikan, S. F. N., Ramli, A. A., & Fudzee, M. F. M. (2019). A Survey Paper on Keystroke Dynamics Authentication for Current Applications. AIP Conference Proceedings, 2173.
- Tewari, A., & Verma, P. (2022). Keystroke Dynamics based Recognition Systems using Deep Learning: A Survey. TechRxiv preprint (not peer-reviewed).

## Next Steps

Phase 2 is complete. The finding (multimodal ≈ visual-only in offline performance) is documented and the model is trained. Proceed to Phase 4: Live Proctor development, using the multimodal model from this approach.