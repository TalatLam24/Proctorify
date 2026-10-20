"""
Proctorify Grand CSV Builder
============================
Combines per-session feature CSVs into a single training-ready CSV
for LOSOCV evaluation.

Usage:
    python build_grand_csv.py --features_dir <path> --output <path> [--sessions <list>]

Arguments:
    --features_dir  Directory containing per-session feature CSVs.
                    Each CSV should be named <SubjectID>_<timestamp>.csv
    --output        Path to write the combined grand CSV.
    --sessions      (Optional) Comma-separated list of session names to include.
                    If omitted, all .csv files in features_dir are included.

Example:
    python build_grand_csv.py \\
        --features_dir ../features/visual_only \\
        --output ../approaches/01_xgboost_rf_visual_baseline/grand_dataset.csv
"""

import os
import sys
import argparse
import pandas as pd


def parse_subject_id(filename):
    """
    Extract Subject_ID from a session filename.
    
    Naming convention: <SubjectName>_<SessionNumber>_<YYYYMMDD>_<HHMMSS>.csv
    where SessionNumber is typically a 2-3 digit number (01, 02, ... 10, etc.)
    
    Examples:
        'Bushra_03_20260610_141336.csv'  -> 'Bushra'
        'Lam_01_20250116_091245.csv'     -> 'Lam'
        'User_01_20250115_143022.csv'    -> 'User'
        'Bushra_20250115_143022.csv'     -> 'Bushra' (no session number)
        'Some_Name_01_20250115_143022'   -> 'Some_Name'
    """
    name = os.path.splitext(filename)[0]
    parts = name.split('_')
    
    # Find the date segment (8 consecutive digits)
    date_idx = None
    for i, part in enumerate(parts):
        if len(part) == 8 and part.isdigit():
            date_idx = i
            break
    
    # If no date segment found, return the full name (fallback)
    if date_idx is None:
        return name
    
    # Everything before the date is subject + optional session number
    pre_date = parts[:date_idx]
    
    if len(pre_date) == 0:
        return 'Unknown'
    
    # If the last segment before date is a short number (1-3 digits),
    # treat it as a session sequence number and strip it off
    if len(pre_date) >= 2 and len(pre_date[-1]) <= 3 and pre_date[-1].isdigit():
        return '_'.join(pre_date[:-1])
    
    # Otherwise, the subject is everything before the date
    return '_'.join(pre_date)


def build_grand_csv(features_dir, output_path, sessions_filter=None):
    """
    Combine all per-session CSVs in features_dir into one grand CSV.
    
    Adds Subject_ID and Session_ID columns. Verifies all CSVs have the
    same columns. Reports any issues.
    """
    if not os.path.isdir(features_dir):
        print(f"ERROR: features_dir not found: {features_dir}")
        sys.exit(1)
    
    csv_files = sorted([f for f in os.listdir(features_dir) if f.endswith('.csv')])
    
    if not csv_files:
        print(f"ERROR: No .csv files found in {features_dir}")
        sys.exit(1)
    
    if sessions_filter:
        filter_set = set(s.strip() for s in sessions_filter.split(','))
        csv_files = [f for f in csv_files if os.path.splitext(f)[0] in filter_set]
        if not csv_files:
            print(f"ERROR: No CSVs match the session filter.")
            sys.exit(1)
    
    print(f"Found {len(csv_files)} session CSV files in {features_dir}")
    print()
    
    all_dfs = []
    expected_columns = None
    column_mismatches = []
    
    for csv_file in csv_files:
        file_path = os.path.join(features_dir, csv_file)
        session_id = os.path.splitext(csv_file)[0]
        subject_id = parse_subject_id(csv_file)
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"  ERROR reading {csv_file}: {e}")
            continue
        
        # Check column consistency on first file
        if expected_columns is None:
            expected_columns = list(df.columns)
            print(f"  Reference columns ({len(expected_columns)}): {expected_columns[:5]}... ")
            print()
        elif list(df.columns) != expected_columns:
            column_mismatches.append({
                'file': csv_file,
                'expected': expected_columns,
                'got': list(df.columns)
            })
            print(f"  WARNING: {csv_file} has different columns! Skipped.")
            continue
        
        # Insert Subject_ID and Session_ID as the first two columns
        df.insert(0, 'Subject_ID', subject_id)
        df.insert(1, 'Session_ID', session_id)
        
        all_dfs.append(df)
        print(f"  Loaded {csv_file}: {len(df)} frames, Subject={subject_id}")
    
    if column_mismatches:
        print()
        print(f"WARNING: {len(column_mismatches)} file(s) had column mismatches and were skipped.")
        print("These files need to be re-extracted with the latest feature_engineer.py")
        print("Skipped files:")
        for m in column_mismatches:
            print(f"  - {m['file']}")
    
    if not all_dfs:
        print()
        print("ERROR: No valid CSVs to combine.")
        sys.exit(1)
    
    # Concatenate all dataframes
    grand_df = pd.concat(all_dfs, ignore_index=True)
    
    # Report label distribution
    if 'Target_Label' in grand_df.columns:
        print()
        print(f"Label distribution across {len(grand_df)} total frames:")
        label_counts = grand_df['Target_Label'].value_counts()
        for label, count in label_counts.items():
            pct = count / len(grand_df) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Report per-subject frame counts and label breakdown
    print()
    print("Frames per subject (with label breakdown):")
    for subject in sorted(grand_df['Subject_ID'].unique()):
        subj_df = grand_df[grand_df['Subject_ID'] == subject]
        n_total = len(subj_df)
        n_sessions = subj_df['Session_ID'].nunique()
        if 'Target_Label' in subj_df.columns:
            n_normal = (subj_df['Target_Label'] == 'NORMAL').sum()
            n_cheat = ((subj_df['Target_Label'] == 'CHEAT_PHYSICAL') | 
                       (subj_df['Target_Label'] == 'CHEAT_DIGITAL')).sum()
            n_other = n_total - n_normal - n_cheat
            cheat_pct = n_cheat / n_total * 100 if n_total > 0 else 0
            print(f"  {subject:12s}: {n_total:6d} frames, {n_sessions} sessions, "
                  f"NORMAL={n_normal}, CHEAT={n_cheat} ({cheat_pct:.1f}%), OTHER={n_other}")
        else:
            print(f"  {subject:12s}: {n_total:6d} frames, {n_sessions} sessions, (no labels)")
    
    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    grand_df.to_csv(output_path, index=False)
    print()
    print(f"Grand CSV saved: {output_path}")
    print(f"Total: {len(grand_df)} rows, {len(grand_df.columns)} columns")


def main():
    parser = argparse.ArgumentParser(
        description="Build grand training CSV from per-session feature files."
    )
    parser.add_argument('--features_dir', required=True,
                        help='Directory containing per-session feature CSVs')
    parser.add_argument('--output', required=True,
                        help='Path to write the combined grand CSV')
    parser.add_argument('--sessions', default=None,
                        help='Comma-separated list of session names to include (default: all)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PROCTORIFY GRAND CSV BUILDER")
    print("=" * 60)
    print(f"Features dir:   {args.features_dir}")
    print(f"Output:         {args.output}")
    print(f"Sessions filter: {args.sessions if args.sessions else '(all)'}")
    print()
    
    build_grand_csv(args.features_dir, args.output, args.sessions)


if __name__ == "__main__":
    main()