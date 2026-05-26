# 🚦 AI Smart Traffic Controller: Comprehensive Technical Documentation

## 1. Executive Summary

The **AI Smart Traffic Controller** is an advanced, production-grade traffic management system designed to replace traditional, fixed-timer traffic lights. By combining **Computer Vision (YOLOv8)** for real-time perception and **Deep Reinforcement Learning (PPO)** for intelligent decision-making, the system dynamically routes traffic to minimize wait times, maximize throughput, and enforce rigorous safety standards.

Furthermore, it features a specialized, secondary AI model explicitly trained for **Emergency Vehicle (Ambulance) Detection**, allowing the system to execute automated "Emergency Overrides" to instantly clear traffic for first responders.

---

## 2. The Problem with Traditional Systems

Traditional traffic light systems operate on fixed time intervals (e.g., 30 seconds for Lane 1, 30 seconds for Lane 2). This leads to severe inefficiencies:
- **Empty Lane Starvation:** A green light remains active for a lane with zero cars, while dozens of cars wait unnecessarily in a red lane.
- **Inflexible Emergency Response:** First responders are often blocked by traffic and must slowly force their way through red lights.
- **Lack of Perception:** Traditional systems cannot "see" queue lengths and adapt their timings dynamically based on rush-hour loads vs. midnight lulls.

**The Solution:** This project introduces a system that "sees" the intersection, measures queue lengths in real-time, mathematically calculates the optimal time to switch lights using a trained neural network, and safely intercepts normal operations for ambulances.

---

## 3. System Architecture Overview

The system is highly modular, split into distinct subsystems that handle vision, logic, configuration, and visualization.

### Architecture Diagram (Mental Model)
```text
[ Camera / Video Feed ] 
         ↓ (Frames)
[ 1. VehicleDetector (Dual-YOLO) ]
   ├── yolov8m.pt (Standard Cars/Trucks/Buses)
   ├── ambulance_best.pt (Custom Emergency Detector)
   ├── Polygon intersection matching (Assigning cars to lanes)
   └── 10-Frame Smoothing History (Preventing flicker)
         ↓ (Lane Queues: [12, 0, 5, 2], EmergencyLane: -1)
[ 2. BrainEngine (RL State Machine) ]
   ├── PPO Neural Network (Predicts Switch vs Hold)
   ├── Timer Constraints (Min Green: 3s, Max Green: 60s)
   ├── Safety State (Yellow Light Transition: 1.5s)
   └── Smart Overrides (Skip empty lanes, Emergency Preemption)
         ↓ (Active Phase: Lane 0, Color: Green)
[ 3. DashboardHUD (Visualization) ]
   ├── OpenCV Drawing (Bounding boxes, Lane Polygons)
   ├── Traffic Light UI (Red, Yellow, Green indicators)
   └── Status Logging (Reason for AI decision)
         ↓
[ Final Display Output ]
```

---

## 4. Deep Dive: Component Breakdown

### 4.1. The Vision Engine (`detector.py`)
The system uses Ultralytics YOLOv8 for object detection. However, standard models have flaws when handling massive variance in traffic, so we employ a **Dual-Model Architecture**.
*   **Base Model (`yolov8m.pt`):** A medium-sized model pre-trained on the COCO dataset. It is highly optimized to detect classes like `Car (2)`, `Motorcycle (3)`, `Bus (5)`, and `Truck (7)`.
*   **Emergency Model (`ambulance_best.pt`):** A custom, fine-tuned model trained strictly on ambulance imagery. It runs parallel to the base model.
*   **Lane Assignment:** The user defines 4 polygons representing the 4 lanes. The center coordinate of every detected vehicle `(cx, cy)` is mathematically tested against these polygons (`cv2.pointPolygonTest`). If a vehicle falls inside a polygon, the queue count for that lane increments.
*   **Temporal Smoothing Buffer:** YOLO is not perfect; it may miss a car for a single frame. To prevent the RL brain from reacting to this "phantom drop," the detector maintains a **10-frame sliding window** (`self.history`). It reports the *maximum* number of cars seen in a lane over the last 10 frames, guaranteeing perfectly stable queue counts.

### 4.2. The RL Decision Engine (`brain_engine.py`)
The RL Brain dictates *when* a traffic light should change. It does not dictate *how* it changes (that is handled by the Safety State Machine).
*   **The Neural Network:** A pre-trained Proximal Policy Optimization (PPO) model (`ppo_traffic_brain`). 
*   **Observation Space (Input):** The brain is fed an array of 6 integers every frame:
    `[Lane0_Count, Lane1_Count, Lane2_Count, Lane3_Count, Current_Green_Lane, Time_Elapsed]`
*   **Action Space (Output):** The network outputs a binary decision:
    - `0`: Hold the current green light.
    - `1`: Switch to the next lane.

### 4.3. The Safety State Machine (`brain_engine.py`)
AI models can be erratic. If the RL Brain suddenly outputs `1` immediately after the light turned green, the light would instantly turn red, causing accidents. To prevent this, the RL brain is heavily restricted by a strict set of real-world safety rules:
1.  **Emergency Preemption:** If `detector.py` flags an ambulance in Lane `X`, the RL brain's output is completely ignored. The State Machine immediately forces a transition to Lane `X`.
2.  **Minimum Green Time (`min_phase_seconds=3`):** Once a lane turns green, it is locked green for at least 3 seconds. The AI cannot switch it, guaranteeing vehicles have time to physically accelerate.
3.  **Maximum Green Time (`max_phase_seconds=60`):** To prevent infinite starvation (e.g., if one lane has 100 cars and another has 1), the engine will forcefully switch lanes after 60 seconds regardless of the RL brain's decision.
4.  **The Yellow Phase Transition (`yellow_seconds=1.5`):** Lights never transition directly `Green -> Green`. When a switch is triggered, the system enters a locked `Yellow` state for 1.5 seconds to allow the intersection to clear.
5.  **Smart Skipping:** If the AI decides to switch lanes, the engine checks the queue of the *next* lane. If the next lane is empty, it bypasses it entirely and finds the next populated lane.

### 4.4. The Configurator (`lane_config.py`)
An interactive GUI built with OpenCV. Because camera angles differ per intersection, the user must define where the lanes actually are. The user clicks to draw points, creating 4 specific polygons. These coordinates are serialized to `lanes.json` and loaded automatically on startup.

---

## 5. Model Training Pipeline

### 5.1. Training the RL Traffic Brain (`train_agent.py` & `traffic_env.py`)
The PPO brain was trained using a custom OpenAI `gym.Env` simulation.
*   **Simulation Dynamics:** Vehicles spawn in random lanes over thousands of episodes. The environment simulates cars passing through green lights and stacking up at red lights.
*   **Reward Function:** The RL agent was trained to minimize the total wait time of all cars combined. It was penalized heavily for allowing queues to build up beyond a certain threshold, forcing it to learn a fair balancing mechanism.

### 5.2. Training the Ambulance YOLO Model (`train_emergency_detector.py`)
Because standard COCO models do not have an "Ambulance" class, a custom YOLOv8 model was fine-tuned.
1.  A dataset of thousands of ambulance images was acquired from Roboflow and formatted via `emergency_data.yaml`.
2.  A base `yolov8n.pt` model was loaded.
3.  The model was trained for 50 epochs using Early Stopping. The resultant weights were exported as `best.pt` and renamed to `ambulance_best.pt` for deployment.

---

## 6. Installation & Setup

### 6.1. Prerequisites
Ensure you have Python 3.8+ installed. 
Install the core AI and Vision libraries:
```bash
pip install ultralytics stable-baselines3 opencv-python numpy
```

### 6.2. Directory Structure Requirements
Ensure your project folder contains the following core files:
```text
Traffic_RL/
├── smart_traffic.py              # Main execution script
├── brain_engine.py               # RL Logic
├── detector.py                   # YOLO logic
├── hud.py                        # UI rendering
├── lane_config.py                # Polygon setup
├── ppo_traffic_brain.zip         # The trained RL weights
├── yolov8m.pt                    # Standard YOLO weights (auto-downloads)
└── ambulance_best.pt             # Custom Emergency YOLO weights
```

---

## 7. Execution & Usage

### 7.1. Running the System
The system is executed via the orchestrator script. It will automatically load your custom ambulance model if it exists in the folder.

**To run on a pre-recorded traffic video:**
```bash
python smart_traffic.py --video your_traffic_video.mp4
```

**To run on a live webcam/CCTV feed:**
```bash
python smart_traffic.py --webcam 0
```

### 7.2. First-Time Lane Configuration
When you run the script on a new video source for the first time, the system requires spatial calibration:
1. The video will pause on Frame 1. 
2. A prompt in your terminal will ask you to draw **Lane 0**.
3. Click inside the video window to trace a polygon around the physical lane on the road.
4. Press `ENTER` to complete the polygon.
5. Repeat for Lane 1, Lane 2, and Lane 3.
6. The polygons will be saved to `lanes.json`. The AI will now begin tracking vehicles.

If you ever move the camera and need to redraw the lanes, append the `--setup` flag:
```bash
python smart_traffic.py --video video.mp4 --setup
```

### 7.3. Real-Time Controls
While the traffic system is actively running, you have the following keyboard controls:
- **`Q`**: Terminate the program immediately.
- **`S`**: Pause the simulation and re-trigger the Lane Configuration setup tool to adjust polygons on the fly.

### 7.4. Testing Emergency Overrides (Without the Custom Model)
If you do not have the `ambulance_best.pt` model trained yet, but you wish to observe how the State Machine handles emergency priority, you can use the simulator flag.
```bash
python smart_traffic.py --simulate-ambulance
```
This forces the Computer Vision engine to temporarily categorize all generic **Trucks (Class 7)** as Emergency Vehicles, allowing you to watch the RL Brain forcefully switch to a Yellow state the moment a truck enters the frame.

---

## 8. Extending and Modifying the Project

The modular nature of the code makes it trivial to extend:
- **Adding Pedestrian Crossings:** You can modify `detector.py` to add `Person (0)` to `VEHICLE_CLASSES` and draw a 5th polygon over crosswalks. Modify `brain_engine.py` to trigger a global red light if the pedestrian count > 0.
- **Changing Speed vs Accuracy:** If the system is lagging on edge hardware (like a Raspberry Pi), open `smart_traffic.py` and modify the default model argument from `yolov8m.pt` (Medium) to `yolov8n.pt` (Nano). 
- **Modifying Timers:** Open `brain_engine.py` and adjust `min_phase_seconds`, `max_phase_seconds`, and `yellow_seconds` in the `__init__` function to fit the unique geometry of specific intersections.
