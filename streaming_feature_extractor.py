"""
Proctorify Streaming Feature Extractor
=======================================
Single-frame, real-time feature extraction with rolling buffer for variance.
This is the SAME code path used for both training and live inference,
eliminating train/serve skew.

Key design decisions:
  1. Processes ONE frame at a time (no batch, no future frames)
  2. Maintains a rolling time-based buffer for variance computation
  3. Uses ffill ONLY for missing faces (never bfill — can't use future frames)
  4. For initial frames with no prior value, uses learned mean imputation
  5. Velocity computed from current and previous frame only (causal)

This class can be used in two modes:
  - TRAINING: Feed pre-recorded video frame-by-frame, collect features for CSV
  - LIVE: Feed webcam frames in real-time, get features for immediate inference

Usage (Training Mode):
    extractor = StreamingFeatureExtractor()
    for frame in video:
        features = extractor.process_frame(frame, timestamp)
        # features is a dict of all feature columns

Usage (Live Mode):
    extractor = StreamingFeatureExtractor(mean_impute_values={...})
    for frame in webcam_stream:
        features = extractor.process_frame(frame, timestamp)
        prediction = model.predict([features.values()])
        smoothed = extractor.smooth_prediction(prediction)

The mean_impute_values dict should be populated from training data statistics
before deploying to live inference. This ensures initial "no face" frames get
reasonable values instead of NaN.
"""

import time
import numpy as np
import cv2
import mediapipe as mp
from collections import deque

# --- CONFIGURATION ---
GAZE_Y_MIN = 0.0
GAZE_Y_MAX = 1.0
VARIANCE_WINDOW_SEC = 2.0

# 3D facial model points (same as feature_engineer.py)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype="double")

LANDMARK_INDICES = [1, 152, 263, 33, 287, 57]

# Feature column names
SPATIAL_COLS = ['Pitch', 'Yaw', 'Qx', 'Qy', 'Qz', 'Qw', 'Gaze_X', 'Gaze_Y']
VELOCITY_COLS = [f'{c}_Vel' for c in SPATIAL_COLS]
VARIANCE_COLS = [f'{c}_Var' for c in SPATIAL_COLS]
BINARY_COLS = ['Face_Detected', 'Is_Interacting']
ALL_FEATURE_COLS = SPATIAL_COLS + VELOCITY_COLS + VARIANCE_COLS + BINARY_COLS


# ============================================================================
# QUATERNION (same as feature_engineer.py — shared code)
# ============================================================================

def rotation_matrix_to_quaternion(rmat):
    """Convert 3x3 rotation matrix to quaternion (qx, qy, qz, qw)."""
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

    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    if norm > 1e-10:
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    if qw < 0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw

    return qx, qy, qz, qw


def calculate_head_pose(face_landmarks, img_w, img_h):
    """Estimate head pose from facial landmarks."""
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

    rmat, _ = cv2.Rodrigues(rotation_vector)
    pitch = np.degrees(np.arcsin(np.clip(rmat[2, 1], -1, 1)))
    yaw = np.degrees(np.arctan2(-rmat[2, 0], rmat[2, 2]))
    qx, qy, qz, qw = rotation_matrix_to_quaternion(rmat)
    return pitch, yaw, qx, qy, qz, qw


def calculate_iris_gaze(landmarks):
    """Compute iris gaze direction with Gaze_Y clipping."""
    try:
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

        gaze_y = np.clip(gaze_y, GAZE_Y_MIN, GAZE_Y_MAX)
        gaze_x = np.clip(gaze_x, 0.0, 1.0)
        return gaze_x, gaze_y
    except IndexError:
        return np.nan, np.nan


# ============================================================================
# ROLLING VARIANCE BUFFER
# ============================================================================

class RollingVarianceBuffer:
    """
    Maintains a time-based rolling buffer for computing variance
    over the last `window_sec` seconds.

    This is the streaming equivalent of compute_time_based_variance()
    in feature_engineer.py — same mathematical operation, but
    incremental (O(1) amortized per frame instead of O(n*w)).
    """

    def __init__(self, window_sec=VARIANCE_WINDOW_SEC):
        self.window_sec = window_sec
        # Store (timestamp, value) tuples
        self._buffer = deque()

    def update_and_compute(self, timestamp, value):
        """
        Add a new value and compute variance over [t - window_sec, t].

        Returns:
            float: variance over the window, or 0.0 if <2 values
        """
        # Add new value
        self._buffer.append((timestamp, value))

        # Evict values outside the window
        t_start = timestamp - self.window_sec
        while len(self._buffer) > 1 and self._buffer[0][0] < t_start:
            self._buffer.popleft()

        # Compute variance
        if len(self._buffer) < 2:
            return 0.0

        vals = np.array([v for _, v in self._buffer])
        return np.var(vals, ddof=1)

    def reset(self):
        self._buffer.clear()


# ============================================================================
# PREDICTION SMOOTHING
# ============================================================================

class TemporalSmoother:
    """
    Exponential Moving Average (EMA) for smoothing frame-level predictions.

    Prevents flickering (NORMAL-CHEAT-NORMAL-CHEAT on adjacent frames)
    without adding latency. The EMA is causal — uses only past predictions.

    Usage:
        smoother = TemporalSmoother(alpha=0.3)
        for frame in stream:
            raw_prob = model.predict_proba(frame)
            smooth_prob = smoother.update(raw_prob)
            prediction = 1 if smooth_prob > 0.5 else 0
    """

    def __init__(self, alpha=0.3):
        """
        Args:
            alpha: EMA smoothing factor. Higher = more responsive to new data.
                   0.3 = moderate smoothing (good default for 30fps video)
                   1.0 = no smoothing (raw frame-level)
                   0.1 = heavy smoothing (slow to react)
        """
        self.alpha = alpha
        self._ema = None

    def update(self, probability):
        """Update EMA with new probability and return smoothed value."""
        if self._ema is None:
            self._ema = probability
        else:
            self._ema = self.alpha * probability + (1 - self.alpha) * self._ema
        return self._ema

    def reset(self):
        self._ema = None


# ============================================================================
# STREAMING FEATURE EXTRACTOR
# ============================================================================

class StreamingFeatureExtractor:
    """
    Real-time, single-frame feature extractor for the live proctor.

    Processes one frame at a time, maintaining:
      - ffill state (last known good spatial values)
      - rolling variance buffers (one per spatial feature)
      - previous frame values (for velocity computation)

    This is the SAME code path for training and inference.
    No bfill, no batch operations, no future frame access.
    """

    def __init__(self, mean_impute_values=None, variance_window_sec=VARIANCE_WINDOW_SEC,
                 smoothing_alpha=0.3):
        """
        Args:
            mean_impute_values: dict mapping feature names to mean values
                for imputation when no prior value exists (initial frames).
                Populated from training data statistics.
                Example: {'Pitch': 17.5, 'Yaw': -5.2, 'Gaze_Y': 0.35, ...}
                If None, uses 0.0 for all features (not recommended).
            variance_window_sec: time window for rolling variance
            smoothing_alpha: EMA alpha for prediction smoothing
        """
        # Mean imputation values (from training data)
        if mean_impute_values is None:
            self.mean_impute = {col: 0.0 for col in SPATIAL_COLS}
        else:
            self.mean_impute = mean_impute_values

        # ffill state: last known good spatial values
        self._last_good = {col: None for col in SPATIAL_COLS}

        # Rolling variance buffers (one per spatial feature)
        self._var_buffers = {
            col: RollingVarianceBuffer(variance_window_sec)
            for col in SPATIAL_COLS
        }

        # Previous frame state (for velocity computation)
        self._prev_spatial = None
        self._prev_timestamp = None

        # Frame counter
        self._frame_count = 0

        # Prediction smoother
        self.smoother = TemporalSmoother(alpha=smoothing_alpha)

        # MediaPipe FaceLandmarker (lazy initialization)
        self._landmarker = None
        self._landmarker_options = None

    def init_mediapipe(self, model_path='face_landmarker.task'):
        """Initialize MediaPipe FaceLandmarker (call once before processing)."""
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1
        )
        self._landmarker_options = options
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr, timestamp_sec, interactions_last_2s=0):
        """
        Process a single frame and return all features.

        Args:
            frame_bgr: numpy array (BGR format from OpenCV)
            timestamp_sec: frame timestamp in seconds
            interactions_last_2s: count of keyboard/mouse interactions in last 2s

        Returns:
            dict with keys = ALL_FEATURE_COLS, values = float
            Ready for model.predict([feature_vector])
        """
        if self._landmarker is None:
            raise RuntimeError("Call init_mediapipe() before processing frames")

        # --- Detect face ---
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        )
        timestamp_ms = int(timestamp_sec * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        # --- Extract spatial features ---
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            img_h, img_w, _ = frame_bgr.shape

            pitch, yaw, qx, qy, qz, qw = calculate_head_pose(landmarks, img_w, img_h)
            gaze_x, gaze_y = calculate_iris_gaze(landmarks)

            current_spatial = {
                'Pitch': pitch, 'Yaw': yaw,
                'Qx': qx, 'Qy': qy, 'Qz': qz, 'Qw': qw,
                'Gaze_X': gaze_x, 'Gaze_Y': gaze_y
            }
            face_detected = 1

            # Update ffill state
            for col in SPATIAL_COLS:
                if np.notna(current_spatial[col]):
                    self._last_good[col] = current_spatial[col]

        else:
            # No face detected — use ffill (last known) or mean imputation
            current_spatial = {}
            for col in SPATIAL_COLS:
                if self._last_good[col] is not None:
                    current_spatial[col] = self._last_good[col]  # ffill
                else:
                    current_spatial[col] = self.mean_impute[col]  # mean impute
            face_detected = 0

        # --- Compute velocity (causal: current - previous) ---
        velocity = {}
        if self._prev_spatial is not None and self._prev_timestamp is not None:
            delta_t = timestamp_sec - self._prev_timestamp
            if delta_t < 1e-6:
                delta_t = 0.033  # fallback to ~30fps
            for col in SPATIAL_COLS:
                velocity[f'{col}_Vel'] = (current_spatial[col] - self._prev_spatial[col]) / delta_t
        else:
            # First frame — velocity = 0
            for col in SPATIAL_COLS:
                velocity[f'{col}_Vel'] = 0.0

        # --- Compute rolling variance (time-based, causal) ---
        variance = {}
        for col in SPATIAL_COLS:
            var_val = self._var_buffers[col].update_and_compute(
                timestamp_sec, current_spatial[col]
            )
            variance[f'{col}_Var'] = var_val

        # --- Binary features ---
        is_interacting = 1.0 if interactions_last_2s > 0 else 0.0

        # --- Update previous frame state ---
        self._prev_spatial = current_spatial.copy()
        self._prev_timestamp = timestamp_sec
        self._frame_count += 1

        # --- Assemble feature dict ---
        features = {}
        features.update(current_spatial)
        features.update(velocity)
        features.update(variance)
        features['Face_Detected'] = float(face_detected)
        features['Is_Interacting'] = is_interacting

        return features

    def get_feature_vector(self, features):
        """Convert features dict to ordered list matching ALL_FEATURE_COLS."""
        return [features[col] for col in ALL_FEATURE_COLS]

    def smooth_prediction(self, raw_probability):
        """Apply temporal smoothing to a raw prediction probability."""
        return self.smoother.update(raw_probability)

    def reset(self):
        """Reset all internal state (call between sessions)."""
        self._last_good = {col: None for col in SPATIAL_COLS}
        for buf in self._var_buffers.values():
            buf.reset()
        self._prev_spatial = None
        self._prev_timestamp = None
        self._frame_count = 0
        self.smoother.reset()

    @staticmethod
    def compute_training_means(df, feature_cols=SPATIAL_COLS):
        """
        Compute mean values from a training DataFrame for mean imputation.

        Args:
            df: DataFrame with ALL_FEATURE_COLS
            feature_cols: columns to compute means for

        Returns:
            dict mapping feature name → mean value

        Usage:
            means = StreamingFeatureExtractor.compute_training_means(train_df)
            extractor = StreamingFeatureExtractor(mean_impute_values=means)
        """
        face_mask = df['Face_Detected'] == 1
        means = {}
        for col in feature_cols:
            if face_mask.any():
                means[col] = df.loc[face_mask, col].mean()
            else:
                means[col] = df[col].mean()
            if np.isnan(means[col]):
                means[col] = 0.0
        return means


# ============================================================================
# TRAINING MODE: Process recorded video with streaming extractor
# ============================================================================

def process_session_streaming(session_folder, base_dir, extractor):
    """
    Process a recorded session using the streaming extractor.
    This is the TRAINING code path — same logic as live inference.

    Returns:
        list of dicts, each dict = one frame's features
    """
    import os
    import pandas as pd

    session_path = os.path.join(base_dir, session_folder)
    video_path = os.path.join(session_path, "webcam_video.mp4")
    telemetry_path = os.path.join(session_path, "frame_telemetry.csv")

    if not os.path.exists(video_path) or not os.path.exists(telemetry_path):
        return None

    df_telemetry = pd.read_csv(telemetry_path)
    cap = cv2.VideoCapture(video_path)

    all_features = []
    extractor.reset()  # Reset state between sessions

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(df_telemetry):
            break

        timestamp = df_telemetry.loc[frame_idx, 'Timestamp']
        interactions = df_telemetry.loc[frame_idx, 'Interactions_Last_2s']

        features = extractor.process_frame(frame, timestamp, interactions)
        features['Frame_ID'] = frame_idx
        features['Timestamp'] = timestamp
        all_features.append(features)

        frame_idx += 1

    cap.release()
    return all_features


# ============================================================================
# MAIN: Demonstrate training-mode usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PROCTORIFY STREAMING FEATURE EXTRACTOR")
    print("Single-frame processing | ffill-only | Rolling variance | EMA smoothing")
    print("=" * 60)
    print()
    print("This module provides the StreamingFeatureExtractor class.")
    print()
    print("Training usage:")
    print("  extractor = StreamingFeatureExtractor()")
    print("  extractor.init_mediapipe('face_landmarker.task')")
    print("  for frame in video:")
    print("      features = extractor.process_frame(frame, timestamp)")
    print()
    print("Live inference usage:")
    print("  means = StreamingFeatureExtractor.compute_training_means(train_df)")
    print("  extractor = StreamingFeatureExtractor(mean_impute_values=means)")
    print("  extractor.init_mediapipe('face_landmarker.task')")
    print("  for frame in webcam_stream:")
    print("      features = extractor.process_frame(frame, timestamp)")
    print("      prob = model.predict_proba([extractor.get_feature_vector(features)])[0][1]")
    print("      smooth_prob = extractor.smooth_prediction(prob)")
    print("      is_cheating = smooth_prob > 0.5")
    print()
    print("Feature count:", len(ALL_FEATURE_COLS))
    print("Feature columns:", ALL_FEATURE_COLS)
