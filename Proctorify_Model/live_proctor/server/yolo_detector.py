"""
Proctorify YOLO Phone Detector
==============================
Uses YOLOv8n to detect cell phones in webcam frames.
If a phone is detected, it boosts the ML model's probability
to ensure the alert fires.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class YoloPhoneDetector:
    """
    Real-time phone detector using YOLOv8n.
    
    Usage:
        detector = YoloPhoneDetector(model_path, confidence_threshold, target_class)
        phone_detected, confidence = detector.detect(frame)
    """

    def __init__(self, model_path, confidence_threshold=0.5, target_class="cell phone"):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to the YOLOv8n .pt model file
            confidence_threshold: Minimum confidence to consider a phone detected
            target_class: The COCO class name to detect (default: "cell phone")
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Find the class index for "cell phone" in the COCO names dictionary
        names = self.model.names
        self.class_index = None
        for idx, name in names.items():
            if name == target_class:
                self.class_index = idx
                break
        
        if self.class_index is None:
            raise ValueError(f"Class '{target_class}' not found in YOLO model classes")
        
        print(f"[YOLO] Loaded. Detecting class '{target_class}' (index {self.class_index}) at confidence >= {confidence_threshold}")

    def detect(self, frame):
        """
        Run YOLO detection on a single frame.
        
        Args:
            frame: BGR image (numpy array, as decoded from webcam)
        
        Returns:
            tuple: (phone_detected: bool, max_confidence: float)
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False)
        
        phone_detected = False
        max_confidence = 0.0
        
        # Check all detections in the frame
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # If this detection is a cell phone and confidence is high enough
                if cls == self.class_index and conf >= self.confidence_threshold:
                    phone_detected = True
                    max_confidence = max(max_confidence, conf)
        
        return phone_detected, max_confidence