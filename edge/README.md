# Edge Deployment Subsystem - Raspberry Pi 5 (ARM64)

Hosts the edge visual inspection pipeline optimized for an ARM64-powered **Raspberry Pi 5 (8GB)**.

---

## Hardware Pipeline Architecture

```
Conveyor Belt
    ↓
[Photoelectric Proximity Sensor] (BCM Pin 24)
    ↓ (Rising Edge Trigger)
[Industrial USB / CSI Camera] (/dev/video0)
    ↓ (Acquire Frame)
[ResNet-50 Defect Inference Engine]
    ↓ (Compute Defect Probability)
Decision:
  - If DEFECT -> [Pneumatic Solenoid Rejection Actuator] (BCM Pin 18) fires -> Part diverted to quarantine bin
  - If OK     -> Part proceeds along conveyor
    ↓
[Async Telemetry Sync] -> Central FastAPI Backend & Prometheus
```

---

## Interfaces & Extension Points

- `edge/scripts/camera_interface.py`: Clean driver abstraction supporting OpenCV V4L2 USB cameras, PiCamera2, or simulation mock frames.
- `edge/scripts/sensor_trigger.py`: Proximity detection stub supporting hardware GPIO interrupts or simulation cycles.
- `edge/scripts/rejection_mechanism.py`: Actuator driver pulsing solenoid relays to eject defective items.
- `edge/scripts/edge_runner.py`: End-to-end edge pipeline.

---

## Running on Raspberry Pi 5

```bash
# Standalone run:
python edge/scripts/edge_runner.py --cycles 10

# Or via Docker ARM64 container:
docker compose -f edge/docker-compose.arm64.yml up --build -d
```
