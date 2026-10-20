"""
Proctorify Multimodal Feature Merger
=====================================
Combines per-session engineered_features.csv (26 visual features)
and keyboard_features.csv (29 keyboard features) into a single
multimodal_features.csv per session, then copies to the
features/multimodal_keyboard/ folder for use in Approach 02.

The merger:
  1. Verifies Frame_ID alignment between the two files
  2. Drops the duplicate Timestamp column from keyboard_features.csv
  3. Merges on Frame_ID
  4. Saves the combined file in the session folder
  5. Copies to features/multimodal_keyboard/<session_name>.csv

Output columns (57 total):
  Frame_ID, Timestamp,
  [26 visual features from engineered_features.csv],
  [29 keyboard features from keyboard_features.csv],
  Face_Detected, Is_Interacting, Target_Label

Usage:
    python build_multimodal_features.py
"""

import os
import sys
import shutil
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths relative to shared/ folder
SESSIONS_DIR = '../../Honeypot_Sessions'
FEATURES_DIR = '../features/multimodal_keyboard'


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PROCTORIFY MULTIMODAL FEATURE MERGER")
    print("=" * 70)
    print(f"Sessions directory: {SESSIONS_DIR}")
    print(f"Output directory:   {FEATURES_DIR}")
    print()

    if not os.path.isdir(SESSIONS_DIR):
        print(f"ERROR: Sessions directory not found: {SESSIONS_DIR}")
        sys.exit(1)

    # Create output directory
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # Find all session directories
    sessions = sorted([
        d for d in os.listdir(SESSIONS_DIR)
        if os.path.isdir(os.path.join(SESSIONS_DIR, d))
    ])

    print(f"Found {len(sessions)} session directories")
    print()

    n_processed = 0
    n_skipped_missing_visual = 0
    n_skipped_missing_keyboard = 0
    n_skipped_mismatch = 0
    skipped_sessions = []

    for session in sessions:
        session_path = os.path.join(SESSIONS_DIR, session)
        visual_path = os.path.join(session_path, 'engineered_features.csv')
        keyboard_path = os.path.join(session_path, 'keyboard_features.csv')

        # Verify both files exist
        if not os.path.exists(visual_path):
            print(f"  SKIP {session}: missing engineered_features.csv")
            n_skipped_missing_visual += 1
            skipped_sessions.append((session, 'missing visual'))
            continue
        if not os.path.exists(keyboard_path):
            print(f"  SKIP {session}: missing keyboard_features.csv")
            n_skipped_missing_keyboard += 1
            skipped_sessions.append((session, 'missing keyboard'))
            continue

        # Load both files
        df_visual = pd.read_csv(visual_path)
        df_keyboard = pd.read_csv(keyboard_path)

        # Verify row counts match
        if len(df_visual) != len(df_keyboard):
            print(f"  SKIP {session}: row count mismatch "
                  f"(visual={len(df_visual)}, keyboard={len(df_keyboard)})")
            n_skipped_mismatch += 1
            skipped_sessions.append((session, f'row count mismatch'))
            continue

        # Verify Frame_IDs align
        if not (df_visual['Frame_ID'].values == df_keyboard['Frame_ID'].values).all():
            print(f"  SKIP {session}: Frame_ID mismatch (frames out of order)")
            n_skipped_mismatch += 1
            skipped_sessions.append((session, 'Frame_ID mismatch'))
            continue

        # Merge: drop duplicate Timestamp from keyboard, then merge on Frame_ID
        df_keyboard_to_merge = df_keyboard.drop(columns=['Timestamp'])
        df_merged = pd.merge(df_visual, df_keyboard_to_merge, on='Frame_ID', how='inner')

        # Save merged file in the session folder
        merged_path = os.path.join(session_path, 'multimodal_features.csv')
        df_merged.to_csv(merged_path, index=False)

        # Copy to features/multimodal_keyboard/ with session name
        dest_path = os.path.join(FEATURES_DIR, f"{session}.csv")
        shutil.copy(merged_path, dest_path)

        n_processed += 1
        print(f"  OK   {session}: {len(df_merged):,} rows, {len(df_merged.columns)} columns")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Sessions processed:           {n_processed}")
    print(f"Skipped (missing visual):     {n_skipped_missing_visual}")
    print(f"Skipped (missing keyboard):   {n_skipped_missing_keyboard}")
    print(f"Skipped (row/Frame_ID mismatch): {n_skipped_mismatch}")
    print(f"Total skipped:                {n_skipped_missing_visual + n_skipped_missing_keyboard + n_skipped_mismatch}")

    if skipped_sessions:
        print()
        print("Skipped sessions:")
        for s, reason in skipped_sessions:
            print(f"  - {s}: {reason}")

    if n_processed > 0:
        print()
        print(f"Multimodal feature files saved to: {FEATURES_DIR}")
        print(f"Each file has 57 columns: Frame_ID, Timestamp, "
              f"26 visual + 29 keyboard features, "
              f"Face_Detected, Is_Interacting, Target_Label")
        print()
        print("Next steps:")
        print(f"  1. Run build_grand_csv.py with --features_dir {FEATURES_DIR}")
        print(f"     to combine all sessions into a grand dataset for Approach 02")


if __name__ == "__main__":
    main()