"""
Smart Traffic Controller – Main Entry Point
=============================================
Orchestrates YOLO detection, RL brain decisions, and HUD rendering.

Usage:
    python smart_traffic.py                        # saved lanes + default video
    python smart_traffic.py --video path.mp4       # specify video
    python smart_traffic.py --setup                # force lane re-setup
    python smart_traffic.py --webcam 0             # use webcam
    python smart_traffic.py --emergency-model e.pt # custom ambulance YOLO

Controls (while running):
    Q  – quit
    S  – re-run lane setup
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

from lane_config import LaneConfigurator
from detector import VehicleDetector
from brain_engine import BrainEngine
from hud import DashboardHUD

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO = os.path.join(SCRIPT_DIR, "watermarked_preview (1).mp4")
DEFAULT_LANES = os.path.join(SCRIPT_DIR, "lanes.json")
DEFAULT_BRAIN = "ppo_traffic_brain"


def parse_args():
    p = argparse.ArgumentParser(description="AI Smart Traffic Controller")
    p.add_argument("--video", default=DEFAULT_VIDEO,
                   help="Path to video file")
    p.add_argument("--webcam", type=int, default=None,
                   help="Webcam index (overrides --video)")
    p.add_argument("--setup", action="store_true",
                   help="Force lane re-configuration")
    p.add_argument("--lanes", default=DEFAULT_LANES,
                   help="Path to lanes.json config")
    p.add_argument("--brain", default=DEFAULT_BRAIN,
                   help="Path to PPO model (without .zip)")
    p.add_argument("--yolo", default="yolov8m.pt",
                   help="Path to YOLOv8 weights (default: yolov8m.pt)")
    p.add_argument("--emergency-model", default="ambulance_best.pt" if os.path.exists("ambulance_best.pt") else None,
                   help="Path to custom emergency vehicle YOLO model")
    p.add_argument("--simulate-ambulance", action="store_true",
                   help="Treat trucks (class 7) as ambulances for easy testing")
    p.add_argument("--max-phase", type=int, default=45,
                   help="Max seconds per green phase (default 45)")
    p.add_argument("--confidence", type=float, default=0.3,
                   help="YOLO detection confidence threshold")
    return p.parse_args()


def banner():
    print("\n" + "=" * 56)
    print("     AI SMART TRAFFIC CONTROLLER")
    print("     YOLOv8m Eyes  +  PPO RL Brain")
    print("=" * 56)


def ensure_lanes(args, video_source):
    """Return lane polygons, launching setup if needed."""
    polys = LaneConfigurator.load(args.lanes)
    
    if polys and len(polys) == 4 and not args.setup:
        # Prompt the user visually to confirm if they want to keep these lanes
        cap = cv2.VideoCapture(video_source)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            display = frame.copy()
            # Draw existing lanes
            for idx, pts in polys.items():
                pts_arr = np.array(pts, np.int32)
                cv2.polylines(display, [pts_arr], True, LaneConfigurator.COLORS[idx], 3)
                cx, cy = pts_arr.mean(axis=0).astype(int)
                cv2.putText(display, f"Lane {idx}", (cx - 30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, LaneConfigurator.COLORS[idx], 2)
            
            # Instruction banner
            cv2.rectangle(display, (0, 0), (display.shape[1], 55), (0, 0, 0), -1)
            cv2.putText(display, "Use existing lanes for this video? (Y/Enter = Yes, N = Redraw)", 
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Lane Confirmation", display)
            print("\n  [?] Waiting for user to confirm lanes (Y/N) in the video window...")
            
            keep_lanes = True
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key in [ord('y'), ord('Y'), 13, 32]: # Y, Enter, Space
                    keep_lanes = True
                    break
                elif key in [ord('n'), ord('N')]:
                    keep_lanes = False
                    break
                    
            cv2.destroyWindow("Lane Confirmation")
            
            if keep_lanes:
                print(f"  [OK] Using existing lanes from {args.lanes}")
                return polys
            else:
                print("  [i] User chose to redraw lanes.")

    print("  [!!] Starting lane setup ...")
    cfg = LaneConfigurator(config_path=args.lanes)
    ok = cfg.setup(video_source)
    if not ok:
        print("  Setup cancelled.")
        sys.exit(0)
    return LaneConfigurator.load(args.lanes)


def main():
    args = parse_args()
    banner()

    # --- video source -------------------------------------------------
    video_source = args.webcam if args.webcam is not None else args.video
    print(f"\n  Video source : {video_source}")

    # --- lanes --------------------------------------------------------
    lane_polygons = ensure_lanes(args, video_source)

    # --- modules ------------------------------------------------------
    print("\n  Initializing modules ...")
    detector = VehicleDetector(model_path=args.yolo,
                               confidence=args.confidence,
                               emergency_model_path=args.emergency_model,
                               simulate_ambulance=args.simulate_ambulance)
    brain = BrainEngine(model_path=args.brain,
                        max_phase_seconds=args.max_phase)
    hud_overlay = DashboardHUD()
    print("  [OK] All systems ready!\n")

    # --- main loop ----------------------------------------------------
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_source}")
        sys.exit(1)

    cv2.namedWindow("AI Smart Traffic Controller", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("AI Smart Traffic Controller", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    total_frames = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n  Video ended.")
            break

        # 1. Detect
        det = detector.detect(frame, lane_polygons)

        # 2. Decide
        decision = brain.decide(det['lane_counts'], det['emergency_lane'])

        # 3. Draw HUD
        hud_overlay.draw(frame, lane_polygons, det, decision,
                         brain.get_stats())

        # 4. Show
        cv2.imshow("AI Smart Traffic Controller", frame)
        total_frames += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cap.release()
            cv2.destroyAllWindows()
            lane_polygons = ensure_lanes(
                argparse.Namespace(setup=True, lanes=args.lanes),
                video_source)
            cap = cv2.VideoCapture(video_source)
            cv2.namedWindow("AI Smart Traffic Controller",
                            cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("AI Smart Traffic Controller", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # --- summary ------------------------------------------------------
    elapsed = time.time() - t0
    stats = brain.get_stats()
    print("\n" + "=" * 56)
    print("  SESSION SUMMARY")
    print("=" * 56)
    print(f"  Frames processed  : {total_frames}")
    print(f"  Total time        : {elapsed:.1f}s")
    print(f"  Avg FPS           : {total_frames / max(elapsed, 1):.1f}")
    print(f"  Phase switches    : {stats['total_switches']}")
    print(f"  Emergency events  : {stats['total_emergencies']}")
    print("=" * 56 + "\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
