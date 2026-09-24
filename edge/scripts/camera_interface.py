from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image

from ml.src.utils.logging import get_logger

logger = get_logger("edge.camera")


class BaseCamera(ABC):

    @abstractmethod
    def capture_frame(self) -> Image.Image:
        pass

    @abstractmethod
    def release(self) -> None:
        pass


class OpenCVCamera(BaseCamera):

    def __init__(self, device_index: int = 0, resolution: tuple[int, int] = (640, 480)) -> None:
        self.device_index = device_index
        self.resolution = resolution
        self.cap = None

        try:
            import cv2
            self.cap = cv2.VideoCapture(device_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            logger.info(f"OpenCV Camera initialized at device index {device_index}")
        except Exception as e:
            logger.warning(f"Failed to open OpenCV camera index {device_index}: {e}")

    def capture_frame(self) -> Image.Image:
        if self.cap is None or not self.cap.isOpened():
            logger.warning("Camera hardware offline. Generating synthetic diagnostic test frame.")
            return MockCamera().capture_frame()

        import cv2
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to read frame from OpenCV camera.")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            logger.info("OpenCV camera released.")


class MockCamera(BaseCamera):

    def __init__(self, frame_size: tuple[int, int] = (640, 480)) -> None:
        self.frame_size = frame_size
        self._counter = 0

    def capture_frame(self) -> Image.Image:
        self._counter += 1
        is_defect = (self._counter % 4 == 0)
        color = (180, 50, 50) if is_defect else (50, 180, 50)
        img = Image.new("RGB", self.frame_size, color=color)
        logger.debug(f"Mock camera produced frame #{self._counter} (Simulated: {'DEFECT' if is_defect else 'OK'})")
        return img

    def release(self) -> None:
        pass


def create_camera(mode: str = "simulation", device_index: int = 0) -> BaseCamera:
    if mode == "opencv":
        return OpenCVCamera(device_index=device_index)
    return MockCamera()
