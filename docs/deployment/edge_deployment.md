# Edge Deployment Guide - Raspberry Pi 5 (8GB)

## Target Platform Specs

- **SBC**: Raspberry Pi 5 (8GB LPDDR4X)
- **CPU**: Broadcom BCM2712 quad-core Arm Cortex-A76 @ 2.4GHz (64-bit ARMv8.2-A)
- **OS**: Raspberry Pi OS (64-bit) / Debian Bookworm ARM64
- **Camera Interface**: CSI-2 / USB 3.0 Industrial UVC Camera

## Hardware Wiring Schematic

| Sensor / Actuator | Function | Raspberry Pi 5 Pin (BCM) | Physical Header Pin |
|---|---|---|---|
| Photoelectric Proximity Sensor | Part Detection Trigger | GPIO 24 | Pin 18 |
| Solenoid Rejection Valve | Pneumatic Defect Ejection | GPIO 18 (PWM/Out) | Pin 12 |
| Camera | Real-time Frame Capture | USB 3.0 / CSI-2 | USB / CAM Port |
| Ground | Common Ground | GND | Pin 6 / Pin 14 |
| Power | 5V Logic / 24V Solenoid PSU | Relay isolated | External 24V PSU |

## Running the Edge Node

```bash
# 1. Build ARM64 container on Raspberry Pi
docker compose -f edge/docker-compose.arm64.yml build

# 2. Launch container with USB camera passthrough
docker compose -f edge/docker-compose.arm64.yml up -d
```
