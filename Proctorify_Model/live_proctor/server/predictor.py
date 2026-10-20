"""
Proctorify Predictor
====================
Loads the trained XGBoost model and runs real-time predictions.
Handles EMA smoothing and alert tracking (sustained suspicious periods).

Alert logic (no persistence gate for now — Option B empirical tuning):
  - Probability >= threshold AND not in alert → alert STARTS
  - Probability >= threshold AND in alert → alert CONTINUES
  - Probability < threshold AND in alert → alert ENDS
  - Probability < threshold AND not in alert → no alert

When an alert ends, returns alert summary info (duration, peak, avg)
so the server can save it to the database.
"""

import joblib
import numpy as np


class Predictor:
    """
    Real-time predictor with EMA smoothing and alert tracking.

    Usage:
        predictor = Predictor(model_path, threshold=0.35, ema_alpha=0.15)
        prob, label, alert_event, alert_info = predictor.predict(features, timestamp)
    """

    def __init__(self, model_path, threshold=0.35, ema_alpha=0.15):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the saved XGBoost model (.joblib)
            threshold: Decision threshold (probability above this = suspicious)
            ema_alpha: EMA smoothing factor (0 = no update, 1 = no smoothing)
        """
        self.model = joblib.load(model_path)
        self.threshold = threshold
        self.ema_alpha = ema_alpha

        # EMA state
        self.ema_value = None

        # Alert state
        self.in_alert = False
        self.alert_start_time = None
        self.peak_probability = 0.0
        self.probability_sum = 0.0
        self.probability_count = 0

        # Session-level stats
        self.total_frames = 0
        self.flagged_frames = 0
        self.all_probability_sum = 0.0
        self.max_probability = 0.0

    def predict(self, features, timestamp, force_suspicious=False):
        """
        Run prediction on one frame's features.

        Args:
            features: numpy array of 53 feature values
            timestamp: Current timestamp (seconds, Unix epoch)

        Returns:
            tuple: (ema_probability, label, alert_event, alert_info)
              - ema_probability: EMA-smoothed probability (float 0-1)
              - label: 'NORMAL' or 'CHEAT_PHYSICAL'
              - alert_event: 'started', 'ongoing', 'ended', or 'none'
              - alert_info: dict with alert details if event='ended', else None
        """
        # --- 1. Get raw probability from model ---
        raw_prob = float(self.model.predict_proba(features.reshape(1, -1))[0, 1])

        # --- 2. Apply EMA smoothing ---
        if self.ema_value is None:
            self.ema_value = raw_prob
        else:
            self.ema_value = self.ema_alpha * raw_prob + (1 - self.ema_alpha) * self.ema_value

        prob = self.ema_value

        # --- 3. Determine label (with YOLO Override) ---
        if force_suspicious:
            prob = max(prob, 0.95)  # Force probability high
            is_suspicious = True
        else:
            is_suspicious = prob >= self.threshold
            
        label = 'CHEAT_PHYSICAL' if is_suspicious else 'NORMAL'

        # --- 4. Track alert state ---
        alert_event = 'none'
        alert_info = None

        if is_suspicious:
            if not self.in_alert:
                # Alert starting
                self.in_alert = True
                self.alert_start_time = timestamp
                self.peak_probability = prob
                self.probability_sum = prob
                self.probability_count = 1
                alert_event = 'started'
            else:
                # Alert continuing
                self.peak_probability = max(self.peak_probability, prob)
                self.probability_sum += prob
                self.probability_count += 1
                alert_event = 'ongoing'
        else:
            if self.in_alert:
                # Alert ending
                self.in_alert = False
                duration = timestamp - self.alert_start_time
                avg_prob = self.probability_sum / max(self.probability_count, 1)

                alert_info = {
                    'started_at': self.alert_start_time,
                    'ended_at': timestamp,
                    'duration_seconds': duration,
                    'peak_probability': float(self.peak_probability),
                    'avg_probability': float(avg_prob),
                    'label': 'CHEAT_PHYSICAL'
                }
                alert_event = 'ended'

                # Reset alert tracking
                self.alert_start_time = None
                self.peak_probability = 0.0
                self.probability_sum = 0.0
                self.probability_count = 0

        # --- 5. Update session-level stats ---
        self.total_frames += 1
        if is_suspicious:
            self.flagged_frames += 1
        self.all_probability_sum += prob
        self.max_probability = max(self.max_probability, prob)

        return prob, label, alert_event, alert_info

    def get_session_stats(self):
        """Return aggregate session statistics."""
        return {
            'total_frames': self.total_frames,
            'flagged_frames': self.flagged_frames,
            'flag_percentage': (self.flagged_frames / max(self.total_frames, 1)) * 100,
            'avg_probability': self.all_probability_sum / max(self.total_frames, 1),
            'max_probability': self.max_probability,
        }

    def reset(self):
        """Reset all state (for a new session)."""
        self.ema_value = None
        self.in_alert = False
        self.alert_start_time = None
        self.peak_probability = 0.0
        self.probability_sum = 0.0
        self.probability_count = 0
        self.total_frames = 0
        self.flagged_frames = 0
        self.all_probability_sum = 0.0
        self.max_probability = 0.0