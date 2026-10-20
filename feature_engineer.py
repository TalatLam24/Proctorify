"""
Proctorify Feature Engineer
============================
Re-extracts features from raw session data with three critical fixes over
the old offline_feature_extractor.py:

  FIX 1 — Quaternion rotation (no gimbal lock):
    Replaces Euler angle Roll (degenerate at pitch ≈ ±90°, spanned -180° to +180°
    in audit) with quaternion components (Qx, Qy, Qz, Qw) which have no singularities.
    Keeps Pitch and Yaw as interpretable features (they are stable for near-frontal faces).

  FIX 2 — Gaze_Y outlier clipping:
    Clips Gaze_Y to [0.0, 1.0] before any downstream computation.
    Gaze_Y is a ratio of iris position within eye bounds — values >1.0 are physically
    impossible and indicate landmark misdetection (near-closed eyes, occlusions).
    Audit showed max Gaze_Y of 1117 with median 0.26.

  FIX 3 — Time-based variance window:
    Replaces fixed window=60 (frame-based, frame-rate-dependent) with a 2-second
    time-based window using the Timestamp column. This makes variance features
    consistent across sessions at different FPS (29.94 vs 30.19 etc).

Also:
  - Merges existing Target_Label from final_training_dataset.csv if present
    (no re-annotation needed)
  - Outputs a clean CSV ready for model training

Usage:
    python feature_engineer.py [SESSIONS_DIR]

    SESSIONS_DIR defaults to "Honeypot_Sessions"

Output per session:
    {session_dir}/engineered_features.csv   — all features + Target_Label (if annotated)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp

# --- CONFIGURATION ---
BASE_DIR = sys.argv[1] if len(sys.argv) > 1 else "Honeypot_Sessions"
FACE_MODEL_PATH = 'face_landmarker.task'

# Gaze_Y physical bounds (ratio of iris position within eye bounds)
GAZE_Y_MIN = 0.0
GAZE_Y_MAX = 1.0

# Time-based variance window (seconds)
VARIANCE_WINDOW_SEC = 2.0


# ============================================================================
# FIX 1: ROTATION MATRIX → QUATERNION (no gimbal lock)
# ============================================================================

def rotation_matrix_to_quaternion(rmat):
    """
    Convert a 3x3 rotation matrix to a quaternion (qx, qy, qz, qw).

    Uses Shepperd's method for numerical stability, which avoids
    division by near-zero values by choosing the largest diagonal element.

    Reference:
        Shepperd, S.W. (1978). "Quaternion from rotation matrix."
        Journal of Guidance and Control, 1(3), 223-224.

    The quaternion convention is (qx, qy, qz, qw) where:
        qw = cos(θ/2)  (scalar part)
        qx,qy,qz = sin(θ/2) * axis  (vector part)

    This convention matches scipy, PyTorch, and most ML literature.
    """
    trace = rmat[0, 0] + rmat[1, 1] + rmat[2, 2]

    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (rmat[2, 1] - rmat[1, 2]) / s
        qy = (rmat[0, 2] - rmat[2, 0]) / s
        qz = (rmat[1, 0] - rmat[0, 1]) / s
    elif rmat[0, 0] > rmat[1, 1] and rmat[0, 0] > rmat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rmat[0, 0] - rmat[1, 1] - rmat[2, 2])
        qw = (rmat[2, 1] - rmat[1, 2]) / s
        qx = 0.25 * s
        qy = (rmat[0, 1] + rmat[1, 0]) / s
        qz = (rmat[0, 2] + rmat[2, 0]) / s
    elif rmat[1, 1] > rmat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rmat[1, 1] - rmat[0, 0] - rmat[2, 2])
        qw = (rmat[0, 2] - rmat[2, 0]) / s
        qx = (rmat[0, 1] + rmat[1, 0]) / s
        qy = 0.25 * s
        qz = (rmat[1, 2] + rmat[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rmat[2, 2] - rmat[0, 0] - rmat[1, 1])
        qw = (rmat[1, 0] - rmat[0, 1]) / s
        qx = (rmat[0, 2] + rmat[2, 0]) / s
        qy = (rmat[1, 2] + rmat[2, 1]) / s
        qz = 0.25 * s

    # Normalize to unit quaternion (numerical drift correction)
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    if norm > 1e-10:
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

    # Ensure canonical form (qw >= 0) for consistency
    # (q and -q represent the same rotation; we pick the hemisphere with qw >= 0)
    if qw < 0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw

    return qx, qy, qz, qw


# ============================================================================
# HEAD POSE ESTIMATION
# ============================================================================

# 3D facial model points (standard anthropometric model used in head pose literature)
# Reference: Umeyama (1991), adapted for MediaPipe landmark indices
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Right eye outer corner
    (225.0, 170.0, -135.0),      # Left eye outer corner
    (-150.0, -150.0, -125.0),    # Right mouth corner
    (150.0, -150.0, -125.0)      # Left mouth corner
], dtype="double")

# MediaPipe landmark indices for the corresponding 3D model points
LANDMARK_INDICES = [1, 152, 263, 33, 287, 57]


def calculate_head_pose(face_landmarks, img_w, img_h):
    """
    Estimate head pose from facial landmarks using solvePnP.

    Returns:
        pitch, yaw (degrees) — interpretable Euler angles (stable for near-frontal faces)
        qx, qy, qz, qw      — quaternion rotation (no gimbal lock, no angle wrapping)

    The Euler angle extraction uses the YXZ intrinsic rotation convention:
        pitch = arcsin(R[2,1])
        yaw   = atan2(-R[2,0], R[2,2])
    Roll is NOT returned as an Euler angle because it suffers from gimbal lock
    degeneracy (audit showed Roll spanning -180° to +180° with std=141°).
    The quaternion captures the full rotation without singularities.
    """
    image_points = np.array([
        (face_landmarks[i].x * img_w, face_landmarks[i].y * img_h)
        for i in LANDMARK_INDICES
    ], dtype="double")

    camera_matrix = np.array([
        [img_w, 0,      img_w / 2],
        [0,     img_w,  img_h / 2],
        [0,     0,      1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Rotation vector → rotation matrix
    rmat, _ = cv2.Rodrigues(rotation_vector)

    # Extract Pitch and Yaw (safe for near-frontal faces where |pitch| < 85°)
    pitch = np.degrees(np.arcsin(np.clip(rmat[2, 1], -1, 1)))
    yaw = np.degrees(np.arctan2(-rmat[2, 0], rmat[2, 2]))

    # FIX 1: Extract quaternion (no gimbal lock, no angle wrapping)
    qx, qy, qz, qw = rotation_matrix_to_quaternion(rmat)

    return pitch, yaw, qx, qy, qz, qw


# ============================================================================
# FIX 2: IRIS GAZE WITH OUTLIER CLIPPING
# ============================================================================

def calculate_iris_gaze(landmarks):
    """
    Compute iris gaze direction from MediaPipe face landmarks.

    Gaze_X: horizontal iris position ratio within eye width [0, 1]
    Gaze_Y: vertical iris position ratio within eye height [0, 1]

    FIX 2: Gaze_Y is clipped to [GAZE_Y_MIN, GAZE_Y_MAX] = [0, 1].
    Values outside this range are physically impossible (iris outside eye bounds)
    and indicate landmark misdetection during blinks, occlusions, or partial faces.

    Audit evidence:
      NORMAL:  median=0.40, max=344.2  (860x median)
      CHEAT:   median=0.26, max=1117.5 (4300x median)
    After clipping, the median difference (0.40 vs 0.26) should become discriminative.
    """
    try:
        # --- Gaze_X (horizontal) ---
        r_inner = landmarks[362].x
        r_outer = landmarks[263].x
        r_iris = landmarks[473].x
        l_inner = landmarks[133].x
        l_outer = landmarks[33].x
        l_iris = landmarks[468].x

        r_width = abs(r_inner - r_outer) + 1e-6
        r_ratio = abs(r_iris - r_outer) / r_width
        l_width = abs(l_inner - l_outer) + 1e-6
        l_ratio = abs(l_iris - l_outer) / l_width

        gaze_x = (r_ratio + l_ratio) / 2.0

        # --- Gaze_Y (vertical) ---
        r_top = landmarks[386].y
        r_bottom = landmarks[374].y
        r_iris_y = landmarks[473].y
        r_height = abs(r_bottom - r_top) + 1e-6
        r_y_ratio = abs(r_iris_y - r_top) / r_height

        l_top = landmarks[159].y
        l_bottom = landmarks[145].y
        l_iris_y = landmarks[468].y
        l_height = abs(l_bottom - l_top) + 1e-6
        l_y_ratio = abs(l_iris_y - l_top) / l_height

        gaze_y = (r_y_ratio + l_y_ratio) / 2.0

        # FIX 2: Clip Gaze_Y to physical bounds
        gaze_y = np.clip(gaze_y, GAZE_Y_MIN, GAZE_Y_MAX)

        # Gaze_X should also be in [0, 1] by construction, but clip for safety
        gaze_x = np.clip(gaze_x, 0.0, 1.0)

        return gaze_x, gaze_y

    except IndexError:
        return np.nan, np.nan


# ============================================================================
# FIX 3: TIME-BASED VARIANCE WINDOW
# ============================================================================

def compute_time_based_variance(series, timestamps, window_sec):
    """
    Compute rolling variance using a time-based window instead of a fixed
    frame count.

    For each frame at time t, variance is computed over all frames in the
    interval [t - window_sec, t]. This makes the feature frame-rate-independent:
    a 2-second window at 30fps covers ~60 frames, while at 25fps it covers ~50
    frames, but both represent the same physical time span.

    Args:
        series:      pd.Series of feature values (one per frame)
        timestamps:  pd.Series of timestamps in seconds (same length as series)
        window_sec:  window duration in seconds

    Returns:
        pd.Series of variance values (same length as series)

    Complexity: O(n * w) where w is the average number of frames in the window.
    For 18K frames at 30fps with a 2s window, w ≈ 60, so ~1M operations — fast.
    """
    vals = series.values
    ts = timestamps.values
    n = len(vals)
    result = np.full(n, np.nan)

    for i in range(n):
        t_start = ts[i] - window_sec
        # Find the leftmost index within the window
        # (timestamps are monotonically increasing)
        j = i
        while j > 0 and ts[j - 1] >= t_start:
            j -= 1
        # Compute variance over window [j, i]
        window_vals = vals[j:i + 1]
        # Need at least 2 values for meaningful variance
        if len(window_vals) >= 2:
            result[i] = np.var(window_vals, ddof=1)
        elif len(window_vals) == 1:
            result[i] = 0.0

    return pd.Series(result, index=series.index)


# ============================================================================
# FEATURE COLUMNS
# ============================================================================

# Primary spatial features (after fixes)
SPATIAL_COLS = ['Pitch', 'Yaw', 'Qx', 'Qy', 'Qz', 'Qw', 'Gaze_X', 'Gaze_Y']

# Velocity features (derived from spatial)
VELOCITY_COLS = [f'{c}_Vel' for c in SPATIAL_COLS]

# Variance features (derived from spatial, time-based)
VARIANCE_COLS = [f'{c}_Var' for c in SPATIAL_COLS]

# Binary features
BINARY_COLS = ['Face_Detected', 'Is_Interacting']

# All modelable features
ALL_FEATURE_COLS = SPATIAL_COLS + VELOCITY_COLS + VARIANCE_COLS + BINARY_COLS

# Identifier columns
ID_COLS = ['Frame_ID', 'Timestamp']


# ============================================================================
# SESSION PROCESSING
# ============================================================================

def process_session(session_folder, options):
    """Process one session: extract features with all fixes, merge labels, save."""
    session_path = os.path.join(BASE_DIR, session_folder)
    video_path = os.path.join(session_path, "webcam_video.mp4")
    telemetry_path = os.path.join(session_path, "frame_telemetry.csv")
    annotated_path = os.path.join(session_path, "final_training_dataset.csv")
    output_csv = os.path.join(session_path, "engineered_features.csv")

    if not os.path.exists(video_path) or not os.path.exists(telemetry_path):
        print(f"  SKIP: Missing video or telemetry in {session_folder}")
        return

    print(f"Processing: {session_folder}")

    # --- Step 1: Load telemetry (ground truth timestamps) ---
    df_telemetry = pd.read_csv(telemetry_path)

    # --- Step 2: Extract vision features from video ---
    cap = cv2.VideoCapture(video_path)
    spatial_data = []
    gimbal_lock_count = 0
    gaze_y_clipped_count = 0

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx >= len(df_telemetry):
                break

            timestamp_ms = int(df_telemetry.loc[frame_idx, 'Timestamp'] * 1000)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                img_h, img_w, _ = frame.shape

                # Head pose: pitch, yaw + quaternion
                pitch, yaw, qx, qy, qz, qw = calculate_head_pose(
                    landmarks, img_w, img_h
                )

                # Check for gimbal lock condition (|pitch| > 85°)
                if abs(pitch) > 85:
                    gimbal_lock_count += 1

                # Iris gaze with clipping
                gaze_x, gaze_y = calculate_iris_gaze(landmarks)

                spatial_data.append({
                    'Frame_ID': frame_idx,
                    'Face_Detected': 1,
                    'Pitch': pitch, 'Yaw': yaw,
                    'Qx': qx, 'Qy': qy, 'Qz': qz, 'Qw': qw,
                    'Gaze_X': gaze_x, 'Gaze_Y': gaze_y
                })
            else:
                spatial_data.append({
                    'Frame_ID': frame_idx,
                    'Face_Detected': 0,
                    'Pitch': np.nan, 'Yaw': np.nan,
                    'Qx': np.nan, 'Qy': np.nan, 'Qz': np.nan, 'Qw': np.nan,
                    'Gaze_X': np.nan, 'Gaze_Y': np.nan
                })

            frame_idx += 1
            if frame_idx % 500 == 0:
                print(f"  -> Frame {frame_idx}...", end='\r')

    cap.release()

    if not spatial_data:
        print(f"  SKIP: No frames extracted from {session_folder}")
        return

    # --- Step 3: Merge vision + telemetry ---
    df_spatial = pd.DataFrame(spatial_data)
    df_master = pd.merge(df_telemetry, df_spatial, on='Frame_ID')

    # --- Step 4: Handle missing values (blinks/occlusions) ---
    # FIX A: Forward-fill ONLY (live-compatible). bfill is impossible in streaming
    # inference because it copies from FUTURE frames — this causes train/serve skew.
    # For initial frames where ffill has no prior value, use mean imputation
    # (computed from all face-detected frames in this session).
    #
    # Rationale:
    #   - ffill: carry forward last known good value → ✅ causal, live-compatible
    #   - bfill: copy from a future frame → ❌ impossible in live proctor
    #   - mean imputation for initial NaN: uses dataset statistics, not future data
    df_master[SPATIAL_COLS] = df_master[SPATIAL_COLS].ffill()

    # For any remaining NaN (initial frames before first face detection),
    # impute with the session-level mean from face-detected frames
    face_mask = df_master['Face_Detected'] == 1
    for col in SPATIAL_COLS:
        remaining_nan = df_master[col].isna().sum()
        if remaining_nan > 0:
            session_mean = df_master.loc[face_mask, col].mean()
            if pd.notna(session_mean):
                df_master[col] = df_master[col].fillna(session_mean)
            else:
                # No face detected at all in session — fill with 0 as last resort
                df_master[col] = df_master[col].fillna(0.0)

    # --- Step 5: Kinematic derivatives (true time-based) ---
    delta_t = df_master['Timestamp'].diff().fillna(0.033)
    delta_t = delta_t.replace(0, 0.001)  # avoid division by zero

    for col in SPATIAL_COLS:
        # Velocity = Δspatial / Δtime
        df_master[f'{col}_Vel'] = df_master[col].diff().fillna(0) / delta_t

        # FIX 3: Time-based variance (2-second window)
        df_master[f'{col}_Var'] = compute_time_based_variance(
            df_master[col], df_master['Timestamp'], VARIANCE_WINDOW_SEC
        )

    # Fill any remaining NaN in variance (first frame)
    for col in VARIANCE_COLS:
        df_master[col] = df_master[col].fillna(0)

    # --- Step 6: Binary flags ---
    df_master['Is_Interacting'] = (df_master['Interactions_Last_2s'] > 0).astype(float)

    # --- Step 7: Merge existing annotations (if present) ---
    if os.path.exists(annotated_path):
        df_annotated = pd.read_csv(annotated_path)
        if 'Target_Label' in df_annotated.columns:
            df_master['Target_Label'] = df_annotated['Target_Label'].values[:len(df_master)]
            # If lengths differ, pad with 'UNLABELED'
            if len(df_annotated) < len(df_master):
                pad_len = len(df_master) - len(df_annotated)
                df_master.loc[len(df_annotated):, 'Target_Label'] = 'UNLABELED'
            print(f"  -> Merged Target_Label from final_training_dataset.csv")
    else:
        print(f"  -> No annotations found (final_training_dataset.csv not present)")

    # --- Step 8: Select and order output columns ---
    output_cols = ID_COLS + ALL_FEATURE_COLS
    if 'Target_Label' in df_master.columns:
        output_cols.append('Target_Label')

    df_output = df_master[output_cols]
    df_output.to_csv(output_csv, index=False)

    # --- Summary ---
    total_frames = len(df_output)
    face_det_pct = df_output['Face_Detected'].mean() * 100
    n_labeled = (df_output['Target_Label'] != 'UNLABELED').sum() if 'Target_Label' in df_output.columns else 0

    print(f"  -> {total_frames} frames | Face: {face_det_pct:.1f}% | "
          f"Gimbal lock: {gimbal_lock_count} ({gimbal_lock_count/total_frames*100:.2f}%) | "
          f"Labels: {n_labeled}")
    print(f"  -> Saved: {output_csv}")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("PROCTORIFY FEATURE ENGINEER (v2.1)")
    print("Fixes: Quaternion rotation | Gaze_Y clipping | Time-based variance | ffill-only (no bfill)")
    print("=" * 60)
    print()

    if not os.path.exists(FACE_MODEL_PATH):
        print(f"ERROR: {FACE_MODEL_PATH} not found.")
        print("  Download from: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker")
        return

    if not os.path.isdir(BASE_DIR):
        print(f"ERROR: Sessions directory '{BASE_DIR}' not found.")
        return

    # Initialize MediaPipe FaceLandmarker
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1
    )

    # Find and process all session directories
    sessions = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ])

    print(f"Found {len(sessions)} session directories in {BASE_DIR}")
    print()

    for session in sessions:
        process_session(session, options)

    print("=" * 60)
    print("All sessions processed!")
    print()
    print("Output files: {session_dir}/engineered_features.csv")
    print("Feature columns: " + ", ".join(ALL_FEATURE_COLS))
    print(f"Total features: {len(ALL_FEATURE_COLS)}")


if __name__ == "__main__":
    main()
