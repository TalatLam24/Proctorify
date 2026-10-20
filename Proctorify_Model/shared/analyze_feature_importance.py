"""
Proctorify Feature Importance Analyzer
=======================================
Loads the XGBoost models saved by a training run and extracts feature
importances, averaged across the LOSOCV folds.

XGBoost has built-in feature importance based on:
  - "weight": number of times a feature is used in splits
  - "gain": average gain (improvement in objective) when feature is used
  - "cover": average coverage (number of samples affected) when used
  - "total_gain": total gain across all splits using the feature

We use "gain" as the primary metric (most informative for understanding
which features the model actually relies on).

Usage:
    python analyze_feature_importance.py --approach_dir ../approaches/02_xgboost_rf_multimodal_keyboard

Arguments:
    --approach_dir: Path to the approach folder (containing config.yaml
                    and saved_models/ subfolder)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import yaml
import joblib
import xgboost as xgb


def load_config(approach_dir):
    """Load config.yaml from the approach directory."""
    config_path = os.path.join(approach_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_xgboost_models(models_dir):
    """Find all XGBoost model files in the models directory."""
    xgb_files = sorted([f for f in os.listdir(models_dir) if f.endswith('_xgb.joblib')])
    return xgb_files


def extract_importances(model, feature_names):
    """
    Extract feature importances from an XGBoost model.
    
    Returns a dict with multiple importance types:
      - gain: average gain per split using this feature
      - weight: number of splits using this feature
      - cover: average coverage per split using this feature
      - total_gain: total gain across all splits using this feature
    """
    booster = model.get_booster()
    
    # Get importance scores for each type
    importances = {}
    for imp_type in ['gain', 'weight', 'cover', 'total_gain']:
        scores = booster.get_score(importance_type=imp_type)
        # Convert to vector aligned with feature_names
        # XGBoost scores are keyed by 'f0', 'f1', ... or by feature name
        # If by feature name, map directly; if by f-index, parse index
        vec = np.zeros(len(feature_names))
        for key, val in scores.items():
            if key.startswith('f') and key[1:].isdigit():
                idx = int(key[1:])
                if idx < len(feature_names):
                    vec[idx] = val
            else:
                # Key is a feature name
                if key in feature_names:
                    idx = feature_names.index(key)
                    vec[idx] = val
        importances[imp_type] = vec
    
    return importances


def main():
    parser = argparse.ArgumentParser(
        description="Analyze feature importances from saved XGBoost models"
    )
    parser.add_argument('--approach_dir', required=True,
                        help='Path to the approach folder (containing config.yaml and saved_models/)')
    args = parser.parse_args()
    
    approach_dir = os.path.abspath(args.approach_dir)
    config = load_config(approach_dir)
    models_dir = os.path.join(approach_dir, 'saved_models')
    
    print("=" * 70)
    print("PROCTORIFY FEATURE IMPORTANCE ANALYZER")
    print("=" * 70)
    print(f"Approach: {approach_dir}")
    print()
    
    if not os.path.isdir(models_dir):
        print(f"ERROR: saved_models/ folder not found: {models_dir}")
        sys.exit(1)
    
    # Get feature names from config
    feature_names = config['feature_columns']
    print(f"Number of features in config: {len(feature_names)}")
    print()
    
    # Find XGBoost models
    xgb_files = find_xgboost_models(models_dir)
    if not xgb_files:
        print(f"ERROR: No XGBoost models found in {models_dir}")
        sys.exit(1)
    
    print(f"Found {len(xgb_files)} XGBoost models (one per LOSOCV fold)")
    print()
    
    # Load each model and extract importances
    all_importances = {'gain': [], 'weight': [], 'cover': [], 'total_gain': []}
    fold_subjects = []
    
    for xgb_file in xgb_files:
        # Extract subject name from filename (e.g., "Bushra_xgb.joblib" -> "Bushra")
        subject = xgb_file.replace('_xgb.joblib', '')
        fold_subjects.append(subject)
        
        model_path = os.path.join(models_dir, xgb_file)
        model = joblib.load(model_path)
        
        importances = extract_importances(model, feature_names)
        for imp_type in all_importances:
            all_importances[imp_type].append(importances[imp_type])
        
        print(f"  Loaded: {xgb_file}")
    
    print()
    
    # Average across folds
    mean_importances = {}
    for imp_type, vecs in all_importances.items():
        mean_importances[imp_type] = np.mean(vecs, axis=0)
    
    # Build a DataFrame for easy analysis
    df_imp = pd.DataFrame({
        'feature': feature_names,
        'gain_mean': mean_importances['gain'],
        'weight_mean': mean_importances['weight'],
        'cover_mean': mean_importances['cover'],
        'total_gain_mean': mean_importances['total_gain'],
    })
    
    # Add per-fold gains
    for i, subject in enumerate(fold_subjects):
        df_imp[f'gain_{subject}'] = all_importances['gain'][i]
    
    # Sort by gain (most informative)
    df_imp_sorted = df_imp.sort_values('gain_mean', ascending=False).reset_index(drop=True)
    
    # Print top 20 features by gain
    print("=" * 70)
    print("TOP 20 FEATURES BY GAIN (averaged across folds)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Feature':<30} {'Gain':>12} {'Weight':>8} {'Cover':>12}")
    print("-" * 70)
    for i, row in df_imp_sorted.head(20).iterrows():
        print(f"{i+1:<5} {row['feature']:<30} {row['gain_mean']:>12.2f} "
              f"{row['weight_mean']:>8.0f} {row['cover_mean']:>12.0f}")
    
    print()
    print("=" * 70)
    print("BOTTOM 10 FEATURES BY GAIN (least important)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Feature':<30} {'Gain':>12} {'Weight':>8}")
    print("-" * 70)
    for i, row in df_imp_sorted.tail(10).iterrows():
        print(f"{i+1:<5} {row['feature']:<30} {row['gain_mean']:>12.2f} "
              f"{row['weight_mean']:>8.0f}")
    
    # Separate visual vs keyboard features
    visual_features = [
        'Pitch', 'Yaw', 'Qx', 'Qy', 'Qz', 'Qw', 'Gaze_X', 'Gaze_Y',
        'Pitch_Vel', 'Yaw_Vel', 'Qx_Vel', 'Qy_Vel', 'Qz_Vel', 'Qw_Vel',
        'Gaze_X_Vel', 'Gaze_Y_Vel',
        'Pitch_Var', 'Yaw_Var', 'Qx_Var', 'Qy_Var', 'Qz_Var', 'Qw_Var',
        'Gaze_X_Var', 'Gaze_Y_Var'
    ]
    keyboard_features = [f for f in feature_names if f not in visual_features]
    
    # Compute aggregate statistics
    print()
    print("=" * 70)
    print("VISUAL vs KEYBOARD FEATURE GROUP COMPARISON")
    print("=" * 70)
    
    visual_gains = df_imp[df_imp['feature'].isin(visual_features)]['gain_mean']
    keyboard_gains = df_imp[df_imp['feature'].isin(keyboard_features)]['gain_mean']
    
    print(f"Visual features ({len(visual_features)} total):")
    print(f"  Mean gain:   {visual_gains.mean():.2f}")
    print(f"  Median gain: {visual_gains.median():.2f}")
    print(f"  Min gain:    {visual_gains.min():.2f}")
    print(f"  Max gain:    {visual_gains.max():.2f}")
    print(f"  Total gain:  {visual_gains.sum():.2f} ({visual_gains.sum() / df_imp['gain_mean'].sum() * 100:.1f}% of all gain)")
    print()
    print(f"Keyboard features ({len(keyboard_features)} total):")
    print(f"  Mean gain:   {keyboard_gains.mean():.2f}")
    print(f"  Median gain: {keyboard_gains.median():.2f}")
    print(f"  Min gain:    {keyboard_gains.min():.2f}")
    print(f"  Max gain:    {keyboard_gains.max():.2f}")
    print(f"  Total gain:  {keyboard_gains.sum():.2f} ({keyboard_gains.sum() / df_imp['gain_mean'].sum() * 100:.1f}% of all gain)")
    
    # Top keyboard features
    print()
    print("=" * 70)
    print("TOP 10 KEYBOARD FEATURES BY GAIN")
    print("=" * 70)
    df_keyboard = df_imp[df_imp['feature'].isin(keyboard_features)].sort_values('gain_mean', ascending=False)
    print(f"{'Rank':<5} {'Feature':<30} {'Gain':>12} {'Weight':>8}")
    print("-" * 70)
    for i, (_, row) in enumerate(df_keyboard.head(10).iterrows()):
        print(f"{i+1:<5} {row['feature']:<30} {row['gain_mean']:>12.2f} {row['weight_mean']:>8.0f}")
    
    # Save full analysis to CSV
    output_path = os.path.join(approach_dir, 'results', 'feature_importance.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_imp_sorted.to_csv(output_path, index=False)
    print()
    print(f"Full analysis saved to: {output_path}")
    
    # Save per-fold gains for top features
    per_fold_path = os.path.join(approach_dir, 'results', 'feature_importance_per_fold.csv')
    df_imp_sorted.to_csv(per_fold_path, index=False)
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    visual_pct = visual_gains.sum() / df_imp['gain_mean'].sum() * 100
    keyboard_pct = keyboard_gains.sum() / df_imp['gain_mean'].sum() * 100
    print(f"Visual features account for:    {visual_pct:.1f}% of total importance")
    print(f"Keyboard features account for:  {keyboard_pct:.1f}% of total importance")
    print(f"Number of keyboard features with 0 gain: {(keyboard_gains == 0).sum()} out of {len(keyboard_features)}")


if __name__ == "__main__":
    main()