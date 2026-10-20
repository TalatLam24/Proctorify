"""
Proctorify Keyboard Feature Engineer
====================================
Computes 29 keyboard dynamics features per frame, aligned with frame_telemetry.csv
timestamps. Reads raw_inputs.csv (keyboard + mouse events) and outputs
keyboard_features.csv per session.

All features are directly supported by the keystroke dynamics literature:
  - Roy et al. (2022), "A Systematic Literature Review on Latest Keystroke
    Dynamics Based Models", IEEE Access VOLUME 10.
  - Kim et al. (2018), "Keystroke dynamics-based user authentication using
    freely typed text based on user-adaptive feature extraction and novelty
    detection", Applied Soft Computing.

Features (29 total):
  Group A: DD-Time Statistics (5 features) — Roy et al. Section V
  Group B: N-Graph DD-Times (2 features) — Roy et al. Section V
  Group C: L/R/S Region DD-Times (8 features) — kim2018 (Kang and Cho method)
  Group D: Special Key Ratios (4 features) — Roy et al. Section F + kim2018
  Group E: Keystroke Volume (4 features) — Roy et al. Section E
  Group F: Pause Patterns (3 features) — derived from gap statistics
  Group G: Burst Detection (2 features) — typing speed pattern
  Group H: Mouse Activity (1 feature) — behavioral signal from our data

Limitations (acknowledged):
  - We have keydown events only (no keyup), so we cannot compute:
    * Key Duration / dwell time
    * UD-Time / flight time (as defined in literature)
    * UU-Time
    * Digraph-Time (using keyup of second key) — we use DD-only version
    * Trigraph-Time (using keyup of third key) — we use DD-only version

Usage:
    python keyboard_feature_engineer.py [SESSIONS_DIR] [--session SESSION_NAME]

    SESSIONS_DIR defaults to "../../Honeypot_Sessions" (relative to shared/ folder)

    If --session is provided, only that session is processed.
    Otherwise, all subdirectories of SESSIONS_DIR are processed.

Output per session:
    {session_dir}/keyboard_features.csv  — Frame_ID, Timestamp, 29 keyboard features

Example:
    python keyboard_feature_engineer.py
    python keyboard_feature_engineer.py ../../Honeypot_Sessions --session Bushra_03_20260610_141336
"""

import os
import sys
import argparse
import bisect
import numpy as np
import pandas as pd


# ============================================================================
# KEYBOARD LAYOUT — L/R/S REGION MAPPING
# ============================================================================
# Based on standard QWERTY layout, following kim2018 (Kang and Cho method).
# kim2018: "They divided 28 keys (26 letter keys and two functional keys, i.e.,
# the enter key and spacebar, into three groups: L group, R group, and S group.
# The L group comprises keys generally typed by the left hand, while the R group
# comprises keys generally typed by the right hand. Note that the enter key
# belongs to the R group in their study. The S group only comprises the spacebar."
#
# We extend this with number keys (not in kim2018's 28-key set, but present in
# our exam data — students type numbers in answers).
# Left hand numbers: 1-5 (top row, left side)
# Right hand numbers: 6-0 (top row, right side)

LEFT_HAND_KEYS = set('qwertasdfgzxcvb12345QWERTASDFGZXCVB')
RIGHT_HAND_KEYS = set('yuiophjklnm67890YUIOPHJKLNM')
SPACE_KEYS = {'Key.space', ' '}
ENTER_KEYS = {'Key.enter', '\r', '\n'}


def get_key_region(key):
    """
    Return 'L', 'R', or 'S' for the keyboard region of a key.

    Returns None for keys that don't have a clear L/R/S assignment
    (e.g., Key.shift, Key.backspace, Key.ctrl — modifier and function keys
    that aren't part of the L/R/S region scheme).

    Based on kim2018's Kang and Cho method.
    """
    if key in SPACE_KEYS:
        return 'S'
    if key in ENTER_KEYS:
        return 'R'  # Per kim2018: "the enter key belongs to the R group"
    if key in LEFT_HAND_KEYS:
        return 'L'
    if key in RIGHT_HAND_KEYS:
        return 'R'
    return None


# ============================================================================
# BURST DETECTION
# ============================================================================

def count_bursts(times, min_keys=6, max_duration=1.5):
    """
    Count non-overlapping typing bursts in a list of keydown timestamps.

    A burst is defined as a run of >= min_keys keystrokes where the time
    between the first and last keystroke in the run is <= max_duration seconds.

    This captures "transcription" patterns (rapid typing from an external source)
    which typically produce faster, more uniform keystroke sequences than
    normal "thinking-then-typing" exam behavior.

    Args:
        times: sorted list of keydown timestamps
        min_keys: minimum keystrokes to qualify as a burst (default 6)
        max_duration: maximum time span of the burst in seconds (default 1.5)

    Returns:
        int: number of non-overlapping bursts
    """
    if len(times) < min_keys:
        return 0

    burst_count = 0
    i = 0
    while i <= len(times) - min_keys:
        j = i + min_keys - 1
        if times[j] - times[i] <= max_duration:
            # Found a burst starting at i; extend it as far as possible
            burst_count += 1
            k = j
            while k + 1 < len(times) and times[k + 1] - times[i] <= max_duration:
                k += 1
            # Skip past this burst
            i = k + 1
        else:
            i += 1
    return burst_count


def count_burst_keys(times, min_keys=6, max_duration=1.5):
    """
    Count total keystrokes that were part of bursts (non-overlapping).

    Same burst definition as count_bursts(). Returns the total count of
    keystrokes within all detected bursts.

    Args:
        times: sorted list of keydown timestamps
        min_keys: minimum keystrokes to qualify as a burst (default 6)
        max_duration: maximum time span of the burst in seconds (default 1.5)

    Returns:
        int: total keystrokes that were part of bursts
    """
    if len(times) < min_keys:
        return 0

    total_burst_keys = 0
    i = 0
    while i <= len(times) - min_keys:
        j = i + min_keys - 1
        if times[j] - times[i] <= max_duration:
            # Found a burst; extend it as far as possible
            k = j
            while k + 1 < len(times) and times[k + 1] - times[i] <= max_duration:
                k += 1
            total_burst_keys += (k - i + 1)
            i = k + 1
        else:
            i += 1
    return total_burst_keys


# ============================================================================
# FEATURE COMPUTATION
# ============================================================================

def compute_features_for_frame(kbd_times, kbd_keys, mouse_times, t_now):
    """
    Compute all 29 keyboard features for one frame at time t_now.

    Uses binary search (bisect) for efficient window queries since both
    kbd_times and mouse_times are sorted by timestamp.

    Args:
        kbd_times: sorted list of keyboard event timestamps (seconds)
        kbd_keys: list of key names, parallel to kbd_times
        mouse_times: sorted list of mouse event timestamps (seconds)
        t_now: current frame timestamp (seconds)

    Returns:
        dict: {feature_name: value} for all 29 features
    """
    features = {}

    # ------------------------------------------------------------------
    # 2-SECOND WINDOW
    # ------------------------------------------------------------------
    # Used for: Key_Count_2s, DD_Time_Mean_2s
    idx_start = bisect.bisect_left(kbd_times, t_now - 2.0)
    idx_end = bisect.bisect_right(kbd_times, t_now)
    times_2s = kbd_times[idx_start:idx_end]
    keys_2s = kbd_keys[idx_start:idx_end]

    features['Key_Count_2s'] = len(times_2s)

    # DD-Times in 2s window
    if len(times_2s) >= 2:
        dd_times_2s = [times_2s[i + 1] - times_2s[i] for i in range(len(times_2s) - 1)]
        features['DD_Time_Mean_2s'] = float(np.mean(dd_times_2s))
    else:
        features['DD_Time_Mean_2s'] = 0.0

    # ------------------------------------------------------------------
    # 10-SECOND WINDOW
    # ------------------------------------------------------------------
    # Used for: Key_Count_10s, DD-Time stats, N-graph, burst detection
    idx_start = bisect.bisect_left(kbd_times, t_now - 10.0)
    idx_end = bisect.bisect_right(kbd_times, t_now)
    times_10s = kbd_times[idx_start:idx_end]
    keys_10s = kbd_keys[idx_start:idx_end]

    features['Key_Count_10s'] = len(times_10s)

    # DD-Time statistics (Group A: Roy et al. Section V)
    if len(times_10s) >= 2:
        dd_times_10s = [times_10s[i + 1] - times_10s[i] for i in range(len(times_10s) - 1)]
        features['DD_Time_Mean_10s'] = float(np.mean(dd_times_10s))
        features['DD_Time_Std_10s'] = float(np.std(dd_times_10s, ddof=1)) if len(dd_times_10s) >= 2 else 0.0
        features['DD_Time_Min_10s'] = float(np.min(dd_times_10s))
        features['DD_Time_Max_10s'] = float(np.max(dd_times_10s))
    else:
        features['DD_Time_Mean_10s'] = 0.0
        features['DD_Time_Std_10s'] = 0.0
        features['DD_Time_Min_10s'] = 0.0
        features['DD_Time_Max_10s'] = 0.0

    # N-Graph features (Group B: Roy et al. Section V)
    # DiGraph DD-Time = time between keydown N and keydown N+1 (= DD-Time)
    # TriGraph DD-Time = time between keydown N and keydown N+2
    # (Literature defines these using keyup; we use DD-only since we have no keyup)
    features['DiGraph_DD_Mean_10s'] = features['DD_Time_Mean_10s']

    if len(times_10s) >= 3:
        trigraph_times = [times_10s[i + 2] - times_10s[i] for i in range(len(times_10s) - 2)]
        features['TriGraph_DD_Mean_10s'] = float(np.mean(trigraph_times))
    else:
        features['TriGraph_DD_Mean_10s'] = 0.0

    # Burst detection (Group G)
    features['Burst_Count_10s'] = count_bursts(times_10s, min_keys=6, max_duration=1.5)
    features['Burst_Key_Count_10s'] = count_burst_keys(times_10s, min_keys=6, max_duration=1.5)

    # ------------------------------------------------------------------
    # 30-SECOND WINDOW
    # ------------------------------------------------------------------
    # Used for: Key_Count_30s, L/R/S region features, special key ratios,
    # pause patterns, mouse clicks
    idx_start = bisect.bisect_left(kbd_times, t_now - 30.0)
    idx_end = bisect.bisect_right(kbd_times, t_now)
    times_30s = kbd_times[idx_start:idx_end]
    keys_30s = kbd_keys[idx_start:idx_end]

    features['Key_Count_30s'] = len(times_30s)

    # L/R/S Region DD-Times (Group C: kim2018 - Kang and Cho method)
    # kim2018: "categorized into one of eight possible pairs: L-L, L-R, L-S,
    # R-L, R-R, R-S, S-L, and S-R... average DD-time for the corresponding pair"
    region_dd_times = {
        'LL': [], 'LR': [], 'LS': [],
        'RL': [], 'RR': [], 'RS': [],
        'SL': [], 'SR': []
    }
    for i in range(len(keys_30s) - 1):
        r1 = get_key_region(keys_30s[i])
        r2 = get_key_region(keys_30s[i + 1])
        if r1 is None or r2 is None:
            continue  # Skip pairs with unknown regions (e.g., Key.shift)
        dd = times_30s[i + 1] - times_30s[i]
        key = f'{r1}{r2}'
        if key in region_dd_times:
            region_dd_times[key].append(dd)

    for pair, dd_list in region_dd_times.items():
        col_name = f'DD_Time_{pair}_30s'
        features[col_name] = float(np.mean(dd_list)) if dd_list else 0.0

    # Special Key Ratios (Group D: Roy et al. Section F + kim2018)
    # Roy et al. Section F: "the choice of shift and control keys, the frequency
    # of error, error-correcting methods"
    # kim2018: "the proportion of 'CapsLock' usage"
    total_keys_30s = len(keys_30s)
    if total_keys_30s > 0:
        features['Shift_Ratio_30s'] = sum(1 for k in keys_30s if k == 'Key.shift') / total_keys_30s
        features['Backspace_Ratio_30s'] = sum(1 for k in keys_30s if k == 'Key.backspace') / total_keys_30s
        features['Enter_Ratio_30s'] = sum(1 for k in keys_30s if k == 'Key.enter') / total_keys_30s
        features['Space_Ratio_30s'] = sum(1 for k in keys_30s if k == 'Key.space') / total_keys_30s
    else:
        features['Shift_Ratio_30s'] = 0.0
        features['Backspace_Ratio_30s'] = 0.0
        features['Enter_Ratio_30s'] = 0.0
        features['Space_Ratio_30s'] = 0.0

    # Pause Patterns (Group F)
    # Verified from our data: max gap was 31 seconds (thinking pause)
    if len(times_30s) >= 2:
        dd_times_30s = [times_30s[i + 1] - times_30s[i] for i in range(len(times_30s) - 1)]
        features['Pause_Count_30s'] = sum(1 for d in dd_times_30s if d > 2.0)
        features['Longest_Pause_30s'] = float(max(dd_times_30s))
        # Pause ratio: total time in pauses > 2s / total window time (30s)
        total_pause_time = sum(d for d in dd_times_30s if d > 2.0)
        features['Pause_Time_Ratio_30s'] = min(total_pause_time / 30.0, 1.0)
    else:
        features['Pause_Count_30s'] = 0
        features['Longest_Pause_30s'] = 0.0
        features['Pause_Time_Ratio_30s'] = 0.0

    # Mouse Clicks (Group H: 30s window)
    m_idx_start = bisect.bisect_left(mouse_times, t_now - 30.0)
    m_idx_end = bisect.bisect_right(mouse_times, t_now)
    features['Mouse_Click_Count_30s'] = m_idx_end - m_idx_start

    # ------------------------------------------------------------------
    # 60-SECOND WINDOW
    # ------------------------------------------------------------------
    # Used for: Key_Rate_60s (keystrokes per minute)
    # kim2018 mentions "number of words typed per minute (WPM)"; we use
    # keystrokes per minute since we don't reliably know word boundaries
    idx_start = bisect.bisect_left(kbd_times, t_now - 60.0)
    idx_end = bisect.bisect_right(kbd_times, t_now)
    features['Key_Rate_60s'] = idx_end - idx_start

    return features


# ============================================================================
# FEATURE COLUMN ORDER (canonical, used in output CSV)
# ============================================================================

FEATURE_COLUMNS = [
    # Group A: DD-Time Statistics (5)
    'DD_Time_Mean_2s',
    'DD_Time_Mean_10s',
    'DD_Time_Std_10s',
    'DD_Time_Min_10s',
    'DD_Time_Max_10s',
    # Group B: N-Graph DD-Times (2)
    'DiGraph_DD_Mean_10s',
    'TriGraph_DD_Mean_10s',
    # Group C: L/R/S Region DD-Times (8)
    'DD_Time_LL_30s',
    'DD_Time_LR_30s',
    'DD_Time_LS_30s',
    'DD_Time_RL_30s',
    'DD_Time_RR_30s',
    'DD_Time_RS_30s',
    'DD_Time_SL_30s',
    'DD_Time_SR_30s',
    # Group D: Special Key Ratios (4)
    'Shift_Ratio_30s',
    'Backspace_Ratio_30s',
    'Enter_Ratio_30s',
    'Space_Ratio_30s',
    # Group E: Keystroke Volume (4)
    'Key_Count_2s',
    'Key_Count_10s',
    'Key_Count_30s',
    'Key_Rate_60s',
    # Group F: Pause Patterns (3)
    'Pause_Count_30s',
    'Longest_Pause_30s',
    'Pause_Time_Ratio_30s',
    # Group G: Burst Detection (2)
    'Burst_Count_10s',
    'Burst_Key_Count_10s',
    # Group H: Mouse Activity (1)
    'Mouse_Click_Count_30s',
]


# ============================================================================
# SESSION PROCESSING
# ============================================================================

def process_session(session_path, session_name):
    """
    Process one session: read raw_inputs.csv and frame_telemetry.csv,
    compute keyboard features for each frame, write keyboard_features.csv.

    Args:
        session_path: full path to the session directory
        session_name: name of the session (for logging)

    Returns:
        dict: summary statistics for verification
    """
    raw_inputs_path = os.path.join(session_path, 'raw_inputs.csv')
    telemetry_path = os.path.join(session_path, 'frame_telemetry.csv')
    output_path = os.path.join(session_path, 'keyboard_features.csv')

    # Verify required files exist
    if not os.path.exists(raw_inputs_path):
        print(f"  SKIP: Missing raw_inputs.csv")
        return None
    if not os.path.exists(telemetry_path):
        print(f"  SKIP: Missing frame_telemetry.csv")
        return None

    # --- Load raw_inputs.csv ---
    df_raw = pd.read_csv(raw_inputs_path)

    # Separate keyboard and mouse events
    kbd_mask = df_raw['Event_Type'] == 'Keyboard'
    mouse_mask = df_raw['Event_Type'] == 'Mouse'

    kbd_df = df_raw[kbd_mask].copy()
    mouse_df = df_raw[mouse_mask].copy()

    # Sort by timestamp (should already be sorted, but verify)
    kbd_df = kbd_df.sort_values('Timestamp').reset_index(drop=True)
    mouse_df = mouse_df.sort_values('Timestamp').reset_index(drop=True)

    kbd_times = kbd_df['Timestamp'].values.tolist()
    kbd_keys = kbd_df['Key_Button'].values.tolist()
    mouse_times = mouse_df['Timestamp'].values.tolist()

    # --- Load frame_telemetry.csv (source of frame timestamps) ---
    df_tel = pd.read_csv(telemetry_path)
    frame_timestamps = df_tel['Timestamp'].values
    frame_ids = df_tel['Frame_ID'].values

    print(f"  Loaded: {len(kbd_times)} keyboard events, {len(mouse_times)} mouse events, {len(frame_timestamps)} frames")

    # --- Compute features for each frame ---
    results = []
    for i, t_now in enumerate(frame_timestamps):
        features = compute_features_for_frame(kbd_times, kbd_keys, mouse_times, float(t_now))
        features['Frame_ID'] = int(frame_ids[i])
        features['Timestamp'] = float(t_now)
        results.append(features)

    # --- Build output DataFrame ---
    df_out = pd.DataFrame(results)
    # Reorder columns: Frame_ID, Timestamp, then features in canonical order
    output_cols = ['Frame_ID', 'Timestamp'] + FEATURE_COLUMNS
    df_out = df_out[output_cols]

    # --- Save ---
    df_out.to_csv(output_path, index=False)

    # --- Verification summary ---
    summary = {
        'session': session_name,
        'n_frames': len(df_out),
        'n_keyboard_events': len(kbd_times),
        'n_mouse_events': len(mouse_times),
        'features_all_zero': [],
        'features_with_nan': [],
        'feature_means': {},
    }

    for col in FEATURE_COLUMNS:
        col_data = df_out[col]
        if col_data.isna().any():
            summary['features_with_nan'].append(col)
        # A feature is "all zero" if EVERY row is exactly 0
        # (this would indicate either no keyboard activity at all,
        # or a bug in feature computation)
        if (col_data == 0).all():
            summary['features_all_zero'].append(col)
        summary['feature_means'][col] = float(col_data.mean())

    return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Proctorify Keyboard Feature Engineer — computes 29 keyboard dynamics features per frame"
    )
    parser.add_argument('sessions_dir', nargs='?', default='../../Honeypot_Sessions',
                        help='Directory containing session subdirectories (default: ../../Honeypot_Sessions)')
    parser.add_argument('--session', default=None,
                        help='Process only this session (default: process all sessions)')
    args = parser.parse_args()

    print("=" * 70)
    print("PROCTORIFY KEYBOARD FEATURE ENGINEER")
    print("=" * 70)
    print(f"Sessions directory: {args.sessions_dir}")
    print(f"Session filter:    {args.session if args.session else '(all)'}")
    print()

    if not os.path.isdir(args.sessions_dir):
        print(f"ERROR: Sessions directory not found: {args.sessions_dir}")
        sys.exit(1)

    # Find session directories
    if args.session:
        session_path = os.path.join(args.sessions_dir, args.session)
        if not os.path.isdir(session_path):
            print(f"ERROR: Session directory not found: {session_path}")
            sys.exit(1)
        sessions = [args.session]
    else:
        sessions = sorted([
            d for d in os.listdir(args.sessions_dir)
            if os.path.isdir(os.path.join(args.sessions_dir, d))
        ])

    print(f"Found {len(sessions)} session directories")
    print()

    # Process each session
    all_summaries = []
    for i, session in enumerate(sessions, 1):
        session_path = os.path.join(args.sessions_dir, session)
        print(f"[{i}/{len(sessions)}] Processing: {session}")
        summary = process_session(session_path, session)
        if summary is not None:
            all_summaries.append(summary)
            # Print quick verification
            n_zero = len(summary['features_all_zero'])
            n_nan = len(summary['features_with_nan'])
            print(f"  -> {summary['n_frames']} frames saved to keyboard_features.csv")
            if n_zero > 0:
                print(f"  WARNING: {n_zero} features are all-zero: {summary['features_all_zero']}")
            if n_nan > 0:
                print(f"  WARNING: {n_nan} features have NaN values: {summary['features_with_nan']}")
        print()

    # --- Overall verification summary ---
    print("=" * 70)
    print("OVERALL VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Sessions processed: {len(all_summaries)}")
    print(f"Total frames:       {sum(s['n_frames'] for s in all_summaries)}")
    print(f"Total keyboard events: {sum(s['n_keyboard_events'] for s in all_summaries)}")
    print()

    # Aggregate feature means across sessions
    print("Feature means across all sessions:")
    print("-" * 70)
    for col in FEATURE_COLUMNS:
        means = [s['feature_means'][col] for s in all_summaries if col in s['feature_means']]
        if means:
            overall_mean = np.mean(means)
            min_mean = np.min(means)
            max_mean = np.max(means)
            print(f"  {col:30s}  mean={overall_mean:10.4f}  min={min_mean:10.4f}  max={max_mean:10.4f}")

    # Sessions with all-zero features (potential issues)
    print()
    print("Sessions with all-zero features (potential issues):")
    print("-" * 70)
    any_issues = False
    for s in all_summaries:
        if s['features_all_zero']:
            any_issues = True
            print(f"  {s['session']}: {len(s['features_all_zero'])} all-zero features")
            for f in s['features_all_zero']:
                print(f"    - {f}")
    if not any_issues:
        print("  None — all features have non-zero values in all sessions.")

    # Sessions with NaN features (definitely issues)
    print()
    print("Sessions with NaN features (definite issues):")
    print("-" * 70)
    any_nan_issues = False
    for s in all_summaries:
        if s['features_with_nan']:
            any_nan_issues = True
            print(f"  {s['session']}: {len(s['features_with_nan'])} features with NaN")
            for f in s['features_with_nan']:
                print(f"    - {f}")
    if not any_nan_issues:
        print("  None — no NaN values in any feature in any session.")

    print()
    print("Done.")
    print()
    print("Output files: {session_dir}/keyboard_features.csv")
    print(f"Feature columns ({len(FEATURE_COLUMNS)}):")
    for i, col in enumerate(FEATURE_COLUMNS, 1):
        print(f"  {i:2d}. {col}")


if __name__ == "__main__":
    main()