from abc import ABC, abstractmethod
import time
from typing import Callable, Optional

from ml.src.utils.logging import get_logger

logger = get_logger("edge.sensor")


class BaseSensorTrigger(ABC):

    @abstractmethod
    def wait_for_trigger(self) -> bool:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass


class SimulationSensorTrigger(BaseSensorTrigger):

    def __init__(self, interval_seconds: float = 3.0) -> None:
        self.interval = interval_seconds
        self._part_count = 0

    def wait_for_trigger(self) -> bool:
        time.sleep(self.interval)
        self._part_count += 1
        logger.info(f"[TRIGGER] Photoelectric sensor activated: Part #{self._part_count} arrived at inspection gate.")
        return True

    def cleanup(self) -> None:
        logger.info("Simulation sensor trigger stopped.")


class GPIOSensorTrigger(BaseSensorTrigger):

    def __init__(self, pin: int = 24) -> None:
        self.pin = pin
        try:
            import RPi.GPIO as GPIO
            self.gpio = GPIO
            self.gpio.setmode(GPIO.BCM)
            self.gpio.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            logger.info(f"RPi GPIO photoelectric sensor configured on BCM Pin {self.pin}")
        except Exception as e:
            logger.warning(f"RPi.GPIO unavailable ({e}). Fallback to simulation trigger.")
            self.gpio = None

    def wait_for_trigger(self) -> bool:
        if self.gpio is None:
            time.sleep(2.0)
            return True
        self.gpio.wait_for_edge(self.pin, self.gpio.RISING, timeout=5000)
        return True

    def cleanup(self) -> None:
        if self.gpio is not None:
            self.gpio.cleanup(self.pin)
