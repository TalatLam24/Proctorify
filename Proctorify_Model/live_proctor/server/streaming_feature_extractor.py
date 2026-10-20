"""
Proctorify Streaming Feature Extractor
=======================================
Real-time feature extraction for the live proctor. Processes one webcam
frame at a time and maintains state across frames to compute velocity,
variance, and keyboard features.

This produces the same 53 features used during training:
  - 8 spatial features (Pitch, Yaw, Qx, Qy, Qz, Qw, Gaze_X, Gaze_Y)
  - 8 velocity features (time derivative of each spatial feature)
  - 8 variance features (2-second time-based rolling variance)
  - 29 keyboard features (computed from buffered keyboard/mouse events)

Key design decisions:
  - Visual feature computation reuses the exact same helper functions
    from feature_engineer.py (calculate_head_pose, calculate_iris_gaze)
    to eliminate train/serve skew.
  - Keyboard feature computation reuses compute_features_for_frame from
    keyboard_feature_engineer.py.
  - Face not detected: forward-fill from last known values (same as training).
  - First frame with no face: use zeros (same as training behavior).
  - Variance: time-based 2-second window (same as training).
  - Velocity: (current - previous) / delta_t (same as training).
"""

import os
import sys
import time
import numpy as np
import cv2
import mediapipe as mp
from collections import deque

# ============================================================================
# PATH SETUP — import helper functions from existing code
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add project root to path (for feature_engineer.py)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

# Add shared/ to path (for keyboard_feature_engineer.py)
SHARED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'shared'))
sys.path.insert(0, SHARED_DIR)

# Import helper functions from existing code
from feature_engineer import calculate_head_pose, calculate_iris_gaze
from keyboard_feature_engineer import compute_features_for_frame, FEATURE_COLUMNS as KBD_FEATURE_COLS


# ============================================================================
# SPATIAL FEATURE NAMES (must match training order)
# ============================================================================

SPATIAL_COLS = ['Pitch', 'Yaw', 'Qx', 'Qy', 'Qz', 'Qw', 'Gaze_X', 'Gaze_Y']
VELOCITY_COLS = [f'{c}_Vel' for c in SPATIAL_COLS]
VARIANCE_COLS = [f'{c}_Var' for c in SPATIAL_COLS]

# Full feature order (53 features, matching config.yaml)
ALL_FEATURE_NAMES = SPATIAL_COLS + VELOCITY_COLS + VARIANCE_COLS + KBD_FEATURE_COLS


# ============================================================================
# STREAMING FEATURE EXTRACTOR
# ============================================================================

class StreamingFeatureExtractor:
    """
    Real-time feature extractor for the live proctor.

    Maintains state across frames:
      - Previous spatial features (for velocity and ffill)
      - Per-feature timestamp buffer (for 2-second variance window)
      - Keyboard/mouse event buffers (for keyboard features)

    Usage:
        extractor = StreamingFeatureExtractor(face_model_path)
        extractor.add_keyboard_event(timestamp, key)
        extractor.add_mouse_event(timestamp, button)
        features = extractor.process_frame(frame_image, timestamp)
        # features is a numpy array of 53 values
    """

    def __init__(self, face_model_path, variance_window_sec=2.0, max_event_buffer_age=120):
        """
        Initialize the streaming feature extractor.

        Args:
            face_model_path: Path to the MediaPipe face_landmarker.task model
            variance_window_sec: Time window for variance computation (seconds)
            max_event_buffer_age: Maximum age of keyboard/mouse events to keep (seconds)
        """
        if not os.path.exists(face_model_path):
            raise FileNotFoundError(
                f"Face model not found at: {face_model_path}\n"
                f"Download from: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker"
            )

        # MediaPipe FaceLandmarker setup
        self.options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=face_model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(self.options)

        # Configuration
        self.variance_window_sec = variance_window_sec
        self.max_event_buffer_age = max_event_buffer_age

        # State: spatial features
        self.prev_spatial = None  # Previous frame's spatial features (for velocity)
        self.last_known_spatial = None  # Last face-detected spatial features (for ffill)
        self.prev_timestamp = None  # Previous frame's timestamp (for delta_t)

        # State: variance buffers (one deque per spatial feature)
        # Each deque stores (timestamp, value) pairs
        self.variance_buffers = {col: deque() for col in SPATIAL_COLS}

        # State: keyboard/mouse event buffers
        self.kbd_times = deque()
        self.kbd_keys = deque()
        self.mouse_times = deque()

        # Frame counter (for debugging)
        self.frame_count = 0

    def add_keyboard_event(self, timestamp, key):
        """Buffer a keyboard event."""
        self.kbd_times.append(timestamp)
        self.kbd_keys.append(key)

    def add_mouse_event(self, timestamp, button):
        """Buffer a mouse event."""
        self.mouse_times.append(timestamp)

    def _cleanup_old_events(self, current_time):
        """Remove keyboard/mouse events older than max_event_buffer_age."""
        cutoff = current_time - self.max_event_buffer_age
        while self.kbd_times and self.kbd_times[0] < cutoff:
            self.kbd_times.popleft()
            self.kbd_keys.popleft()
        while self.mouse_times and self.mouse_times[0] < cutoff:
            self.mouse_times.popleft()

    def _compute_spatial_features(self, frame, timestamp_ms):
        """
        Run MediaPipe face detection and compute 8 spatial features.
        Returns (spatial_values, face_detected).
        """
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            img_h, img_w, _ = frame.shape

            # Compute head pose: pitch, yaw + quaternion
            pitch, yaw, qx, qy, qz, qw = calculate_head_pose(landmarks, img_w, img_h)

            # Compute iris gaze with clipping
            gaze_x, gaze_y = calculate_iris_gaze(landmarks)

            spatial = {
                'Pitch': pitch, 'Yaw': yaw,
                'Qx': qx, 'Qy': qy, 'Qz': qz, 'Qw': qw,
                'Gaze_X': gaze_x, 'Gaze_Y': gaze_y
            }
            return spatial, True
        else:
            return None, False

    def _compute_velocity(self, current_spatial, prev_spatial, delta_t):
        """Compute velocity = (current - previous) / delta_t for each feature."""
        if delta_t == 0:
            delta_t = 0.001  # Avoid division by zero (same as training)

        velocity = {}
        for col in SPATIAL_COLS:
            velocity[col] = (current_spatial[col] - prev_spatial[col]) / delta_t
        return velocity

    def _compute_variance(self, timestamp, spatial_values):
        """
        Compute time-based variance for each spatial feature.
        Uses a 2-second rolling window (same as training).
        """
        variances = {}

        for col in SPATIAL_COLS:
            buf = self.variance_buffers[col]
            buf.append((timestamp, spatial_values[col]))

            # Remove entries older than variance_window_sec
            cutoff = timestamp - self.variance_window_sec
            while buf and buf[0][0] < cutoff:
                buf.popleft()

            # Compute variance of remaining values
            values = [v for _, v in buf]
            if len(values) >= 2:
                variances[col] = float(np.var(values, ddof=1))
            elif len(values) == 1:
                variances[col] = 0.0
            else:
                variances[col] = 0.0

        return variances

    def process_frame(self, frame, timestamp):
        """
        Process one webcam frame and return all 53 features.

        Args:
            frame: BGR image (numpy array, as captured by OpenCV/webcam)
            timestamp: Current timestamp in seconds (Unix epoch float)

        Returns:
            numpy array of 53 feature values, in the order matching the
            trained model's expected input.
        """
        self.frame_count += 1

        # Clean up old keyboard/mouse events
        self._cleanup_old_events(timestamp)

        # --- Step 1: Compute spatial features (with ffill) ---
        timestamp_ms = int(timestamp * 1000)
        spatial, face_detected = self._compute_spatial_features(frame, timestamp_ms)

        if face_detected:
            # Update last known spatial features for future ffill
            self.last_known_spatial = spatial.copy()
        else:
            # Face not detected: ffill from last known values
            if self.last_known_spatial is not None:
                spatial = self.last_known_spatial.copy()
            else:
                # First frame and no face: use zeros (same as training fallback)
                spatial = {col: 0.0 for col in SPATIAL_COLS}

        # --- Step 2: Compute velocity ---
        if self.prev_spatial is not None and self.prev_timestamp is not None:
            delta_t = timestamp - self.prev_timestamp
            velocity = self._compute_velocity(spatial, self.prev_spatial, delta_t)
        else:
            # First frame: velocity = 0 for all
            velocity = {col: 0.0 for col in SPATIAL_COLS}

        # --- Step 3: Compute variance (updates the buffer) ---
        variance = self._compute_variance(timestamp, spatial)

        # --- Step 4: Update state for next frame ---
        self.prev_spatial = spatial.copy()
        self.prev_timestamp = timestamp

        # --- Step 5: Compute keyboard features ---
        # Convert deques to lists for the keyboard feature function
        kbd_times_list = list(self.kbd_times)
        kbd_keys_list = list(self.kbd_keys)
        mouse_times_list = list(self.mouse_times)

        kbd_features = compute_features_for_frame(
            kbd_times_list, kbd_keys_list, mouse_times_list, timestamp
        )

        # --- Step 6: Build the full feature vector in correct order ---
        feature_vector = []

        # Spatial (8)
        for col in SPATIAL_COLS:
            feature_vector.append(float(spatial[col]))

        # Velocity (8)
        for col in SPATIAL_COLS:
            feature_vector.append(float(velocity[col]))

        # Variance (8)
        for col in SPATIAL_COLS:
            feature_vector.append(float(variance[col]))

        # Keyboard (29)
        for col in KBD_FEATURE_COLS:
            feature_vector.append(float(kbd_features[col]))

        return np.array(feature_vector), face_detected

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()