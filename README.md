# Autonomous Vehicle Simulation (CARLA)

## Overview

This project is an autonomous driving simulation built with the CARLA simulator.  
It is designed to replicate a real self-driving pipeline by connecting perception, decision-making, and control in a virtual environment.

**Pipeline:**  
`Camera -> Perception (YOLO) -> Decision System -> Vehicle Control`

---

## Current Progress

### Stage 1: Core Simulation System (Completed)

- Connected to CARLA server and initialized the simulation world
- Spawned vehicles dynamically using CARLA blueprint library
- Added collision-safe spawning with `try_spawn_actor()`
- Implemented vehicle control (throttle, steering, braking)
- Built a real-time spectator-follow camera system
- Added multiple camera perspectives (driver, dashcam, hood-like viewpoint)

### Stage 2: Perception + Learning Integration (In Progress)

- Attached RGB camera sensor to the ego vehicle
- Streaming real-time frames from CARLA to OpenCV
- Integrated YOLO (Ultralytics) object detection
- Added live telemetry overlay (speed, acceleration, steer, throttle, brake)
- Trained and used a small PyTorch classifier for driving direction probabilities
- Logged vehicle control data to `car_control_data.csv`

---

## Tech Stack

- Python
- CARLA Simulator (`0.9.16` recommended)
- NumPy
- OpenCV
- Ultralytics YOLO
- PyTorch
- Pandas

---

## Project Structure

```text
AutonomousVehicleSimulation/
|-- SelfDrivingVehicle.py
|-- car_control_data.csv
|-- yolo11m.pt
|-- yolo11n.pt
|-- README.md
```

---

## CARLA Setup and Installation

### 1) Download CARLA

Download from: [https://carla.org/](https://carla.org/)  
Recommended version: **CARLA 0.9.16**

### 2) Extract CARLA

Example path (Windows):

```text
C:\CARLA_0.9.16\
```

### 3) Start CARLA Server

**Windows**

```powershell
CarlaUE4.exe
```

**Linux**

```bash
./CarlaUE4.sh
```

Optional (lower graphics for better performance):

```bash
./CarlaUE4.sh -quality-level=Low
```

### 4) Install Python Dependencies

```bash
pip install numpy opencv-python ultralytics torch pandas
```

### 5) Add CARLA Python API Path

In your Python script (before importing `carla`), add your local CARLA API wheel/egg path:

```python
import sys
sys.path.append(r"C:\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl")
import carla
```

If you use a different Python version, use the matching CARLA package file for that version.

---

## Run the Simulation

1. Start CARLA server (`CarlaUE4.exe` or `./CarlaUE4.sh`)
2. Open this project directory
3. Run:

```bash
python SelfDrivingVehicle.py
```

Press `q` in the OpenCV window to close visualization.

---

## Next Steps

- Improve decision-making logic (brake / steer / accelerate)
- Add lane detection + traffic light detection
- Build a real-time dashboard (speed, detections, controls)
- Improve model training pipeline and dataset quality

---

## Future Goals

- Reinforcement learning-based driving agent
- Neural network-based end-to-end control system
- ROS2 integration
- SLAM-based localization
- Sim-to-real transition for robotics workflows

---

## Motivation

This project is built to bridge simulation and real-world autonomy by combining:

- Computer vision
- Machine learning
- Robotics control systems
- Real-time simulation environments

---

## 🚀 Demo

![Preview](https://img.youtube.com/vi/ueAmlV6UAzs/1.jpg)

▶️ Full Demo: https://www.youtube.com/watch?v=ueAmlV6UAzs

## Author

**Aurick Anwar**
