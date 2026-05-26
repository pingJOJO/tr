"""
Train Emergency Vehicle Detector
=================================
Fine-tunes a YOLOv8n model to detect ambulances / emergency vehicles.

STEP 1 – Prepare your dataset
------------------------------
Organise images and YOLO-format labels like this:

    emergency_dataset/
      images/
        train/   ← training images (.jpg / .png)
        val/     ← validation images
      labels/
        train/   ← matching .txt label files (YOLO format)
        val/

Each .txt file has one line per object:
    <class_id> <cx> <cy> <w> <h>      (all normalised 0-1)

For a single-class detector (ambulance only), class_id = 0.

You can use Roboflow (https://roboflow.com) to:
  • find a public ambulance/emergency-vehicle dataset
  • annotate your own images
  • export in "YOLOv8" format

STEP 2 – Create data.yaml
--------------------------
Create a file called ``emergency_data.yaml`` next to this script:

    path: ./emergency_dataset
    train: images/train
    val:   images/val
    nc: 1
    names: ['ambulance']

STEP 3 – Run this script
-------------------------
    python train_emergency_detector.py

The trained model will be saved to:
    runs/detect/emergency_yolo/weights/best.pt

Copy it to the project root and use it with:
    python smart_traffic.py --emergency-model best.pt
"""

import os
import sys


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed.  pip install ultralytics")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(script_dir, "emergency_data.yaml")

    if not os.path.exists(data_yaml):
        print("=" * 56)
        print("  emergency_data.yaml not found!")
        print()
        print("  Please create it following the instructions at the")
        print("  top of this file, then re-run.")
        print("=" * 56)
        sys.exit(1)

    # Start from the small YOLOv8n for fast training
    print("Loading YOLOv8n base model ...")
    model = YOLO("yolov8n.pt")

    print("Starting fine-tuning for emergency vehicle detection ...")
    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        name="emergency_yolo",
        device=0,              # Use GPU
        patience=10,           # early stopping
        save=True,
        plots=True,
    )

    best = os.path.join("runs", "detect", "emergency_yolo", "weights", "best.pt")
    print("\n" + "=" * 56)
    print("  Training complete!")
    print(f"  Best weights saved to: {best}")
    print()
    print("  Usage:")
    print(f"    python smart_traffic.py --emergency-model {best}")
    print("=" * 56)


if __name__ == "__main__":
    main()
