"""
Lane Configuration Module
=========================
Interactive tool to draw and persist 4 traffic-lane polygons.
Draw once → save to lanes.json → never redraw again.

Controls:
    Left-click  : Place polygon corner
    SPACE       : Save current lane & move to next
    Z           : Undo last point
    R           : Reset current lane points
    U           : Undo last saved lane
    Q           : Quit without saving
"""

import cv2
import numpy as np
import json
import os


class LaneConfigurator:
    """Interactive tool to draw and save 4 traffic lane polygons."""

    COLORS = [
        (245, 135, 66),   # Blue   – Lane 0
        (96, 245, 66),    # Green  – Lane 1
        (66, 194, 245),   # Orange – Lane 2
        (215, 66, 245),   # Purple – Lane 3
    ]
    LANE_NAMES = ["North", "East", "South", "West"]

    def __init__(self, config_path="lanes.json"):
        self.config_path = config_path
        self.polygons = {}
        self.current_points = []
        self.current_lane = 0
        self.frame = None

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def setup(self, video_source):
        """Launch interactive lane setup on the first frame of the video."""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_source}")

        ret, self.frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Cannot read first frame from video")

        cv2.namedWindow("Lane Setup", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Lane Setup", self._mouse_callback)

        print("\n" + "=" * 50)
        print("       LANE CONFIGURATION SETUP")
        print("=" * 50)
        print("  Left-click : place polygon corner")
        print("  SPACE      : save current lane")
        print("  Z          : undo last point")
        print("  R          : reset current lane")
        print("  U          : undo last saved lane")
        print("  Q          : quit without saving")
        print("=" * 50 + "\n")

        while self.current_lane < 4:
            self._draw_setup_frame()
            key = cv2.waitKey(30) & 0xFF

            if key == ord(' '):
                if len(self.current_points) >= 3:
                    self.polygons[self.current_lane] = self.current_points.copy()
                    name = self.LANE_NAMES[self.current_lane]
                    n = len(self.current_points)
                    print(f"  [OK] Lane {self.current_lane} ({name}) saved – {n} pts")
                    self.current_lane += 1
                    self.current_points = []
                else:
                    print("  [!!] Need at least 3 points to form a polygon!")

            elif key == ord('z'):
                if self.current_points:
                    self.current_points.pop()

            elif key == ord('r'):
                self.current_points = []

            elif key == ord('u'):
                if self.current_lane > 0:
                    self.current_lane -= 1
                    if self.current_lane in self.polygons:
                        del self.polygons[self.current_lane]
                    self.current_points = []
                    print(f"  [<<] Lane {self.current_lane} removed")

            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False

        cv2.destroyAllWindows()
        self.save()
        return True

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _draw_setup_frame(self):
        display = self.frame.copy()

        # Top instruction bar
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], 55), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        name = self.LANE_NAMES[self.current_lane]
        text = f"Draw Lane {self.current_lane} ({name}) | SPACE=Save  Z=Undo  R=Reset"
        cv2.putText(display, text, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Already-saved polygons (filled + outlined)
        for idx, pts in self.polygons.items():
            pts_arr = np.array(pts, np.int32)
            over2 = display.copy()
            cv2.fillPoly(over2, [pts_arr], self.COLORS[idx])
            cv2.addWeighted(over2, 0.25, display, 0.75, 0, display)
            cv2.polylines(display, [pts_arr], True, self.COLORS[idx], 2)
            cx, cy = pts_arr.mean(axis=0).astype(int)
            cv2.putText(display, f"Lane {idx}",
                        (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS[idx], 2)

        # Current polygon being drawn
        color = self.COLORS[self.current_lane]
        for i, pt in enumerate(self.current_points):
            cv2.circle(display, pt, 6, color, -1)
            cv2.circle(display, pt, 8, (255, 255, 255), 2)
            if i > 0:
                cv2.line(display, self.current_points[i - 1], pt, color, 2)

        cv2.imshow("Lane Setup", display)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self):
        """Save lane polygons to JSON."""
        data = {}
        for idx, pts in self.polygons.items():
            data[str(idx)] = pts
        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n  [OK] Lanes saved to {self.config_path}")

    @staticmethod
    def load(config_path="lanes.json"):
        """Load lane polygons from JSON.  Returns dict {int: np.ndarray} or None."""
        if not os.path.exists(config_path):
            return None
        with open(config_path, 'r') as f:
            data = json.load(f)
        polygons = {}
        for idx_str, pts in data.items():
            polygons[int(idx_str)] = np.array(pts, np.int32)
        return polygons
