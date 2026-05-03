# Autonomous Vehicle Simulation (CARLA)

## Overview

This project is an autonomous driving simulation built using the CARLA simulator. The goal is to develop a full pipeline that mimics real-world self-driving systems:

**Camera → Perception (YOLO) → Decision Model → Vehicle Control**

This repository currently contains the foundational simulation setup, including vehicle spawning, movement, and camera perspective control.

---

## Current Progress (Stage 1: Simulation Setup)

* Connected to CARLA server and loaded environment
* Spawned vehicles dynamically using blueprint library
* Implemented vehicle control (throttle, steering, braking)
* Built real-time camera following using spectator transforms
* Configured multiple camera perspectives (driver view, dashcam, hood view)

---

## Key Concepts Implemented

* **Transforms**: Positioning and orienting objects in the simulation (location + rotation)
* **Spectator Camera System**: Simulated first-person and third-person perspectives
* **Vehicle Control API**: Real-time manipulation of throttle, steering, and braking
* **Spawn Logic**: Safe spawning using `try_spawn_actor()` and collision handling

---

## Tech Stack

* Python
* CARLA Simulator
* NumPy (planned)
* OpenCV (planned)
* YOLO (Ultralytics) (planned)

---

## Project Structure

```
AutonomousVehicleSimulation/
│── main.py
│── README.md
```

---

## Next Steps

* Attach RGB camera sensor to vehicle
* Stream real-time frames from CARLA
* Integrate YOLO for object detection
* Build decision-making logic (brake, steer, accelerate)
* Implement autonomous driving behavior

---

## Future Goals

* Real-time dashboard (speed, object distance, decisions)
* Lane detection and traffic light recognition
* Reinforcement learning / neural network decision model
* ROS2 + SLAM integration

---

## Motivation

This project aims to bridge simulation and real-world autonomous systems by combining computer vision, machine learning, and control systems in a realistic driving environment.

---

## Author

Aurick Anwar
