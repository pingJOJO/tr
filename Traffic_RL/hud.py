"""
Dashboard HUD Module
====================
Professional heads-up display drawn on every video frame:
  - Semi-transparent info panel
  - Queue bar-chart per lane
  - Phase / status / FPS / switch counter
  - Lane polygon overlays (green = active, red = inactive)
  - Emergency flash banner
"""

import cv2
import numpy as np
import time


class DashboardHUD:
    """Renders all visual overlays on the output frame."""

    LANE_COLORS = [
        (245, 135, 66),   # Blue
        (96, 245, 66),    # Green
        (66, 194, 245),   # Orange
        (215, 66, 245),   # Purple
    ]

    def __init__(self):
        self.start_time = time.time()
        self.frame_times = []
        self.fps = 0

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def draw(self, frame, lane_polygons, det_result, decision, stats):
        """Draw everything and return the annotated frame."""
        self._update_fps()
        
        is_yellow = decision.get('is_yellow', False)
        
        self._draw_lanes(frame, lane_polygons,
                         decision['phase'], det_result['emergency_lane'], is_yellow)
        self._draw_detections(frame, det_result['detections'])
        self._draw_panel(frame, det_result, decision, stats)
        if det_result['emergency_lane'] != -1:
            self._draw_emergency_banner(frame)
        return frame

    # ------------------------------------------------------------------
    # FPS
    # ------------------------------------------------------------------
    def _update_fps(self):
        now = time.time()
        self.frame_times.append(now)
        self.frame_times = [t for t in self.frame_times if now - t < 1.0]
        self.fps = len(self.frame_times)

    # ------------------------------------------------------------------
    # Lane overlays
    # ------------------------------------------------------------------
    def _draw_lanes(self, frame, polygons, green_phase, emg_lane, is_yellow):
        overlay = frame.copy()
        for idx, poly in polygons.items():
            if idx == emg_lane:
                color, alpha = (0, 0, 255), 0.4
            elif idx == green_phase:
                color = (0, 255, 255) if is_yellow else (0, 255, 0)
                alpha = 0.3
            else:
                color, alpha = (0, 0, 180), 0.15
            cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        for idx, poly in polygons.items():
            if idx == emg_lane:
                c = (0, 0, 255)
            elif idx == green_phase:
                c = (0, 255, 255) if is_yellow else (0, 255, 0)
            else:
                c = (0, 0, 180)
            cv2.polylines(frame, [poly], True, c, 3)
            cx, cy = poly.mean(axis=0).astype(int)
            label = f"Lane {idx}"
            if idx == green_phase:
                label += " [YELLOW]" if is_yellow else " [GREEN]"
            cv2.putText(frame, label, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ------------------------------------------------------------------
    # Detection markers
    # ------------------------------------------------------------------
    def _draw_detections(self, frame, detections):
        for d in detections:
            cx, cy = d['center']
            if d['is_emergency']:
                x1, y1, x2, y2 = d['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "EMERGENCY", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                c = (255, 200, 0) if d['lane'] >= 0 else (128, 128, 128)
                cv2.circle(frame, (cx, cy), 5, c, -1)
                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)

    # ------------------------------------------------------------------
    # Info panel
    # ------------------------------------------------------------------
    def _draw_panel(self, frame, det, decision, stats):
        pw, ph = 420, 220
        x0, y0 = 10, 10

        # background
        ov = frame.copy()
        cv2.rectangle(ov, (x0, y0), (x0 + pw, y0 + ph), (20, 20, 20), -1)
        cv2.addWeighted(ov, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (80, 80, 80), 2)

        # title
        cv2.putText(frame, "AI TRAFFIC CONTROLLER", (x0 + 15, y0 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.line(frame, (x0 + 10, y0 + 38),
                 (x0 + pw - 10, y0 + 38), (80, 80, 80), 1)

        # status
        cv2.putText(frame, f"Status: {decision['status']}",
                    (x0 + 15, y0 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, decision['color'], 2)

        # green lane
        cv2.putText(frame, f"Green: Lane {decision['phase']}",
                    (x0 + 15, y0 + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # queue bars
        cv2.putText(frame, "Queues:", (x0 + 15, y0 + 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        counts = det['lane_counts']
        mx = max(max(counts.values()), 1)
        bx = x0 + 95
        bw = 200
        for i in range(4):
            by = y0 + 108 + i * 22
            w = int((counts[i] / mx) * bw)
            c = (0, 255, 0) if i == decision['phase'] else self.LANE_COLORS[i]
            cv2.rectangle(frame, (bx, by), (bx + w, by + 14), c, -1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + 14), (80, 80, 80), 1)
            cv2.putText(frame, f"L{i}: {counts[i]}", (x0 + 15, by + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # bottom stats
        elapsed = time.time() - self.start_time
        txt = (f"FPS:{self.fps} | Switches:{stats['total_switches']} "
               f"| Time:{elapsed:.0f}s")
        cv2.putText(frame, txt, (x0 + 15, y0 + ph - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # ------------------------------------------------------------------
    # Emergency flash
    # ------------------------------------------------------------------
    def _draw_emergency_banner(self, frame):
        h, w = frame.shape[:2]
        if int(time.time() * 4) % 2 == 0:
            ov = frame.copy()
            cv2.rectangle(ov, (0, h - 60), (w, h), (0, 0, 200), -1)
            cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame,
                        "!! EMERGENCY VEHICLE DETECTED - PRIORITY OVERRIDE !!",
                        (w // 2 - 350, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
