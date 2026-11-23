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

## Milestone 8 — Multi-Object Tracking

AURORA includes a lightweight multi-object tracker based on
Intersection over Union (IoU) matching.

### Features

- Matches detections across consecutive frames.
- Assigns persistent track IDs.
- Uses greedy highest-IoU matching.
- Keeps tracks temporarily when detections are missed.
- Expires tracks after a configurable number of missed frames.
- Separates objects with different class names.
- Resets track state when required.
- Uses CPU-friendly Python operations without another AI model.

### Run tracking on a recorded video

Place a recorded video at `datasets/sample.mp4`.

Run:

`PYTHONPATH=src python scripts/track_people.py --model models/yolov8n.onnx --video datasets/sample.mp4 --output datasets/tracked_output.mp4`

Optional tracking parameters:

- `--confidence`: detector confidence threshold.
- `--min-iou`: minimum IoU for matching.
- `--max-missed`: number of missed frames allowed before a track expires.

The annotated output video contains bounding boxes, person IDs,
and detection confidence values.

### Limitations

This is a basic IoU tracker. It does not recognize a person's
identity, predict complex motion, or guarantee stable IDs when
people cross, overlap, or move substantially between frames.

This milestone does not implement autonomous flight,
obstacle avoidance, or search-and-rescue mission execution.

## Milestone 9 — Search-and-Rescue Target Assessment

AURORA now includes a lightweight target-assessment module for ranking
tracked person detections for operator attention.

The assessment considers:
- Person classification.
- Detection confidence.
- Track persistence across frames.
- Relative bounding-box visibility.
- Whether the track is currently observed.

Each assessment includes:
- Track ID.
- Candidate status.
- LOW, MEDIUM, or HIGH visual priority.
- A numerical score between 0 and 1.
- Individual assessment factors.
- Human-readable explanations.

The assessment is an operator-support feature. It does not determine
whether a person is injured, unconscious, or in need of rescue.

The current scoring system is a simple baseline and requires validation
with representative search-and-rescue footage before operational use.



## Milestone 11 — Mission Monitoring and Operator Dashboard

AURORA now includes a lightweight local dashboard for reviewing recorded
search-and-rescue perception runs.

### Dashboard features

- Mission frame and detection totals.
- Unique person tracking ID count.
- HIGH, MEDIUM, and LOW visual-priority track-frame counts.
- Per-track first and last observed frame.
- Per-track observation count, maximum confidence, and maximum score.
- Annotated MP4 playback in a local browser.
- JSON mission report for later review.

### Generate a mission report

Run the integrated perception pipeline:

`PYTHONPATH=src python scripts/run_perception_pipeline.py --model models/yolov8n.onnx --video datasets/sample.mp4 --output datasets/aurora_pipeline_output.mp4`

The command also generates:

`datasets/aurora_pipeline_output.json`

### Start the local dashboard

`python scripts/mission_dashboard.py`

Open `http://127.0.0.1:8765` in a browser on the same computer.

Stop the dashboard with Ctrl+C in its terminal.

### Limitations

This dashboard displays reports and video from recorded missions. It does
not provide live camera streaming, UAV telemetry, flight-controller
commands, autonomous navigation, or validated rescue decisions.

Priority scores are experimental visual indicators for operator attention.
They are not medical assessments or proof that a person needs rescue.

## Milestone 13 — Live Telemetry Health Monitoring

Milestone 13 adds telemetry freshness classification and dashboard health
indicators.

### Telemetry health states

- `LIVE`: a MAVLink telemetry snapshot was updated less than 3 seconds ago.
- `STALE`: the latest MAVLink snapshot is at least 3 seconds old.
- `DISCONNECTED`: the latest MAVLink snapshot is at least 10 seconds old,
  or no telemetry snapshot is available.
- `SIMULATED`: telemetry was generated by the simulator.
- `RECORDED`: telemetry comes from another saved source.

The dashboard exposes telemetry health through `/api/telemetry` and displays
the latest update age.

### Health monitor

The module `aurora.telemetry.health` provides:

- `TelemetryHealthMonitor` for monitoring telemetry receive times.
- `TelemetryStatus` for LIVE, STALE and DISCONNECTED states.
- `classify_telemetry_source` for source-aware telemetry classification.

### Dashboard

Start the dashboard with:

    PYTHONPATH=src python scripts/mission_dashboard.py \
      --video datasets/aurora_pipeline_output_h264.mp4

Open:

    http://127.0.0.1:8765

The dashboard is read-only. It does not arm, control or command the UAV.

### Important limitation

MAVLink freshness is currently inferred from the modification time of the
latest telemetry snapshot file. A LIVE status indicates recently updated
MAVLink data; it does not prove that the aircraft is airborne or that the
flight controller is safe to operate. Real flight and SITL telemetry
verification remain separate validation tasks.


## Milestone 15 — Autonomous Search-Grid Generation

AURORA generates rectangular lawnmower-style search routes from GPS boundaries, with configurable lane spacing, altitude and speed. It exports JSON and GeoJSON and displays the search boundary, route and waypoints on the mission map at `/api/search-grid`.

Example: `PYTHONPATH=src python scripts/plan_search_grid.py --south-lat 49.8723 --west-lon 8.6507 --north-lat 49.8733 --east-lon 8.6520 --spacing 20 --altitude 30 --speed 3`

Generated files: `datasets/search_grid.json` and `datasets/search_grid.geojson`.

**Safety:** The route status is `PLANNED_NOT_EXECUTED`. This is a simulation-first planning feature, not a certified flight plan. It does not arm or control a UAV. Review airspace, terrain, obstacles, weather and vehicle limits independently before any real flight.
