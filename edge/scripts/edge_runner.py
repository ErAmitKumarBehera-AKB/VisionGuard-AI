#!/usr/bin/env python3

import argparse
import io
import signal
import sys
import time
from pathlib import Path
import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edge.scripts.camera_interface import create_camera
from edge.scripts.rejection_mechanism import SimulationRejectionMechanism
from edge.scripts.sensor_trigger import SimulationSensorTrigger
from training.src.inference.predictor import DefectPredictor
from training.src.utils.logging import get_logger, setup_logging

logger = get_logger("edge.runner")


class EdgeInspectionNode:

    def __init__(self, config_path: str = "edge/config/edge_config.yaml") -> None:
        self.running = True
        self.config = self._load_config(config_path)

        cam_cfg = self.config.get("camera", {})
        self.camera = create_camera(
            mode=cam_cfg.get("mode", "mock"),
            device_index=cam_cfg.get("device_index", 0),
        )

        trig_cfg = self.config.get("hardware_trigger", {})
        self.trigger = SimulationSensorTrigger(
            interval_seconds=trig_cfg.get("trigger_interval_seconds", 2.0)
        )

        act_cfg = self.config.get("rejection_mechanism", {})
        self.actuator = SimulationRejectionMechanism(
            delay_ms=act_cfg.get("ejection_delay_ms", 150)
        )

        self.predictor = DefectPredictor(
            checkpoint_path=self.config.get("model_service", {}).get("checkpoint_path", None),
            device="auto",
        )

        self.backend_url = self.config.get("backend_url", "http://localhost:8000")

    def _load_config(self, path: str) -> dict:
        p = Path(path)
        if p.is_file():
            with open(p) as f:
                return yaml.safe_load(f) or {}
        return {}

    def run(self, max_cycles: int = 10) -> None:
        logger.info("=" * 60)
        logger.info("EDGE INSPECTION NODE INITIALIZED ON RASPBERRY PI 5")
        logger.info(f"Target Line: {self.config.get('edge_device', {}).get('location', 'Conveyor-Line-A')}")
        logger.info("=" * 60)

        cycles = 0
        try:
            while self.running and (max_cycles <= 0 or cycles < max_cycles):
                self.trigger.wait_for_trigger()
                cycles += 1

                frame = self.camera.capture_frame()

                result = self.predictor.predict(
                    image_input=frame,
                    product_category="metal_nut",
                    source_dataset="edge_camera",
                )

                pred = result["prediction"]
                conf = result["confidence"]
                lat = result["latency_ms"]

                logger.info(
                    f"Cycle #{cycles:03d} -> Result: [{pred}] (Confidence: {conf * 100:.1f}%, Latency: {lat:.1f}ms)"
                )

                if pred == "DEFECT":
                    self.actuator.reject_part(
                        inspection_id=f"EDGE-{cycles}",
                        reason=f"Defect detected ({conf * 100:.1f}%)",
                    )
                else:
                    logger.info("Part PASSED quality check. Conveyor advancing.")

                self._sync_to_backend(frame, result)

        finally:
            self.shutdown()

    def _sync_to_backend(self, frame, result: dict) -> None:
        try:
            buf = io.BytesIO()
            frame.save(buf, format="PNG")
            files = {"image": ("edge_capture.png", buf.getvalue(), "image/png")}
            data = {"product_category": "metal_nut", "source_dataset": "edge_node"}
            requests.post(f"{self.backend_url}/api/v1/inspection/predict", files=files, data=data, timeout=1.0)
        except Exception:
            pass

    def shutdown(self) -> None:
        logger.info("Shutting down edge inspection station...")
        self.camera.release()
        self.trigger.cleanup()
        self.actuator.cleanup()
        logger.info("Edge cleanup complete.")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Run Raspberry Pi 5 edge inspection loop.")
    parser.add_argument("--cycles", type=int, default=5, help="Number of inspection cycles to simulate.")
    args = parser.parse_args()

    node = EdgeInspectionNode()

    def sig_handler(sig, frame):
        node.running = False

    signal.signal(signal.SIGINT, sig_handler)
    node.run(max_cycles=args.cycles)


if __name__ == "__main__":
    main()
