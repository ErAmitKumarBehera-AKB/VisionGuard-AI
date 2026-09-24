from abc import ABC, abstractmethod
import time
from ml.src.utils.logging import get_logger

logger = get_logger("edge.actuator")


class BaseRejectionMechanism(ABC):

    @abstractmethod
    def reject_part(self, inspection_id: str, reason: str = "DEFECT") -> None:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass


class SimulationRejectionMechanism(BaseRejectionMechanism):

    def __init__(self, delay_ms: int = 150) -> None:
        self.delay_ms = delay_ms

    def reject_part(self, inspection_id: str, reason: str = "DEFECT") -> None:
        time.sleep(self.delay_ms / 1000.0)
        logger.warning(
            f"[ACTUATOR ACTION] Pneumatic Solenoid FIRING! "
            f"Diverting defective part ({inspection_id}) into quarantine bin. Reason: {reason}"
        )

    def cleanup(self) -> None:
        pass


class GPIORejectionMechanism(BaseRejectionMechanism):

    def __init__(self, pin: int = 18, pulse_seconds: float = 0.2, delay_ms: int = 150) -> None:
        self.pin = pin
        self.pulse = pulse_seconds
        self.delay_ms = delay_ms
        try:
            import RPi.GPIO as GPIO
            self.gpio = GPIO
            self.gpio.setmode(GPIO.BCM)
            self.gpio.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
            logger.info(f"RPi GPIO rejection solenoid configured on BCM Pin {self.pin}")
        except Exception as e:
            logger.warning(f"RPi.GPIO unavailable ({e}). Fallback to simulation actuator.")
            self.gpio = None

    def reject_part(self, inspection_id: str, reason: str = "DEFECT") -> None:
        time.sleep(self.delay_ms / 1000.0)
        logger.warning(f"[HARDWARE ACTUATOR] Diverting defective item: {inspection_id}")
        if self.gpio is not None:
            self.gpio.output(self.pin, self.gpio.HIGH)
            time.sleep(self.pulse)
            self.gpio.output(self.pin, self.gpio.LOW)

    def cleanup(self) -> None:
        if self.gpio is not None:
            self.gpio.cleanup(self.pin)
