"""
Vehicle Detector Module
=======================
YOLOv8m wrapper for vehicle detection and lane assignment.
Supports an optional custom emergency-vehicle YOLO model.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class VehicleDetector:
    """Detect vehicles with YOLO and assign them to lane polygons."""

    def __init__(self, model_path="yolov8m.pt", confidence=0.3,
                 emergency_model_path=None, simulate_ambulance=False):
        # Force GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  [Device] Using: {self.device.upper()}")
        
        print(f"  Loading YOLO from {model_path} ...")
        self.yolo = YOLO(model_path).to(self.device)
        self.confidence = confidence
        self.simulate_ambulance = simulate_ambulance

        # Auto-detect if using Open Images V7 model (which has a native Ambulance class!)
        self.is_oiv7 = "oiv7" in model_path.lower()
        if self.is_oiv7:
            print("  [OK] OpenImages V7 model detected! Native Ambulance support ENABLED.")
            # OIV7 often detects cars as generic "Vehicle" (567) or "Land vehicle" (302)
            self.VEHICLE_CLASSES = [6, 73, 90, 302, 342, 558, 567] 
            self.CLASS_NAMES = {
                6: "Ambulance", 73: "Bus", 90: "Car", 302: "Vehicle", 
                342: "Motorcycle", 558: "Truck", 567: "Vehicle"
            }
        else:
            self.VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
            self.CLASS_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

        # Smoothing history to prevent flickering when YOLO misses a car for 1 frame
        self.history = []
        self.history_frames = 10  # look back 10 frames

        # Optional dedicated emergency-vehicle detector
        self.emergency_detector = None
        if emergency_model_path:
            try:
                self.emergency_detector = YOLO(emergency_model_path).to(self.device)
                print(f"  [OK] Emergency detector loaded: {emergency_model_path}")
            except Exception as e:
                print(f"  [!!] Emergency detector failed to load: {e}")

    # ------------------------------------------------------------------
    def detect(self, frame, lane_polygons):
        """
        Run detection on *frame* and assign hits to *lane_polygons*.

        Returns dict:
            lane_counts      – {0: n, 1: n, 2: n, 3: n}
            emergency_lane   – lane index with emergency vehicle (-1 if none)
            detections       – list of per-vehicle dicts
            total_vehicles   – int
        """
        results = self.yolo(frame,
                            classes=self.VEHICLE_CLASSES,
                            conf=self.confidence,
                            verbose=False)[0]

        lane_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        emergency_lane = -1
        detections = []

        # --- dedicated emergency model (if available) -----------------
        if self.emergency_detector:
            emg_results = self.emergency_detector(frame,
                                                  conf=0.4,
                                                  verbose=False)[0]
            for box in emg_results.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                for idx, poly in lane_polygons.items():
                    if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                        emergency_lane = idx
                        detections.append(self._det_dict(
                            bx1, by1, bx2, by2, cls_id=-1,
                            cls_name="Ambulance", lane=idx,
                            is_emergency=True))
                        break

        # --- regular vehicle detections -------------------------------
        for box in results.boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2

            assigned_lane = -1
            for idx, poly in lane_polygons.items():
                if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                    assigned_lane = idx
                    lane_counts[idx] += 1
                    break

            # Normal vehicle detection
            is_emg = False
            
            # Temporary testing hack: If enabled, treat trucks (COCO class 7) as ambulances
            if self.simulate_ambulance and not self.is_oiv7 and cls == 7:
                is_emg = True
                
            # If using OIV7 model, class 6 is officially an Ambulance!
            if self.is_oiv7 and cls == 6:
                is_emg = True

            if is_emg and assigned_lane >= 0:
                emergency_lane = assigned_lane
                    
            detections.append(self._det_dict(
                bx1, by1, bx2, by2, cls_id=cls,
                cls_name=self.CLASS_NAMES.get(cls, "?"),
                lane=assigned_lane, is_emergency=is_emg))

        # --- smooth the counts ----------------------------------------
        self.history.append(lane_counts)
        if len(self.history) > self.history_frames:
            self.history.pop(0)

        smoothed_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for i in range(4):
            # Use the max count seen over the last N frames to ignore quick dropouts
            smoothed_counts[i] = max(h.get(i, 0) for h in self.history)

        return {
            'lane_counts': smoothed_counts,
            'emergency_lane': emergency_lane,
            'detections': detections,
            'total_vehicles': sum(smoothed_counts.values()),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _det_dict(x1, y1, x2, y2, cls_id, cls_name, lane, is_emergency):
        return {
            'bbox': (x1, y1, x2, y2),
            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
            'class_id': cls_id,
            'class_name': cls_name,
            'lane': lane,
            'is_emergency': is_emergency,
        }
