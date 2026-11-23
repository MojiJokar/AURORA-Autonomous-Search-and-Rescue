# AURORA — Autonomous Search & Rescue UAV

**AURORA** is an engineering project focused on developing an AI-powered autonomous UAV platform for Search & Rescue, with potential applications in infrastructure inspection.

The project aims to combine robotics, artificial intelligence, computer vision, autonomous navigation and intelligent mission planning.

## Project Vision

Develop a modular UAV software platform capable of:

- Detecting people and relevant objects using computer vision.
- Tracking detected targets across video frames.
- Planning and monitoring search missions.
- Communicating with an autonomous flight controller.
- Adapting mission plans based on detected events and mission conditions.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main development language |
| OpenCV | Image processing |
| YOLO | Object detection |
| ArduPilot | UAV flight control |
| Gazebo | Robotics simulation |
| ROS 2 | Robotics communication and integration |
| GitHub Actions | Automated testing |

Technologies will be introduced progressively as the project develops.

## Development Roadmap

- [ ] Project architecture and development environment
- [ ] Camera and image-processing pipeline
- [ ] AI-based object detection
- [ ] Object tracking
- [ ] Search-area generation and mission planning
- [ ] ArduPilot communication
- [ ] Gazebo integration
- [ ] Intelligent mission replanning
- [ ] Integrated Search & Rescue demonstration

## Project Status

**Early development — AURORA 2.0**

The project is being rebuilt with a clean, modular architecture. Roadmap items represent planned work and are not yet claimed as implemented.

## Engineering Principles

- Modular and maintainable software.
- Testable and reproducible components.
- Documented engineering decisions.
- Incremental development and integration.
- Honest reporting of capabilities and limitations.

## Safety Notice

AURORA is a research and development project. It is not currently certified or intended for real-world autonomous flight or operational rescue missions.

## License

License to be determined.


## Milestone 7 — YOLOv8 Person Detection

AURORA includes CPU-based YOLOv8 ONNX person detection using OpenCV DNN.

Features:
- OpenCV CPU inference.
- 640 x 480 model input.
- COCO person-class detection.
- Confidence filtering and non-maximum suppression.
- AURORA bounding-box results.
- Annotated image output.

Model setup:
Place a compatible model at models/yolov8n.onnx. Model weights are not included in this repository. Check the model provider licensing terms before redistribution.

Run detection:
PYTHONPATH=src python scripts/detect_person.py --model models/yolov8n.onnx --image datasets/bus.jpg --confidence 0.40 --output datasets/bus_detected.jpg

Verification:
The detector passed automated tests and detected three people in a recorded sample photograph. This milestone does not yet implement live UAV operation, object tracking, autonomous navigation, or complete search-and-rescue mission execution.
