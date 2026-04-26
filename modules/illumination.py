"""
Illumination System Module
Controls LED-simulated light source and intensity
"""

import cv2
import numpy as np


class IlluminationSystem:
    """
    Simulates the endoscope LED illumination system.
    Applies a circular vignette/spotlight effect that mimics
    the real light cone of an endoscope tip LED.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.intensity = 1.0          # 0.0 – 2.0  (1.0 = normal)
        self.enabled = True
        self.color_temp = "white"     # white | warm | cool
        self._build_vignette_mask()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_intensity(self, value: float):
        """value in [0, 2]."""
        self.intensity = max(0.0, min(2.0, float(value)))

    def set_enabled(self, state: bool):
        self.enabled = state

    def set_color_temperature(self, mode: str):
        """mode: 'white' | 'warm' | 'cool'"""
        self.color_temp = mode

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply illumination effects to a frame."""
        if frame is None:
            return frame

        result = frame.astype(np.float32)

        if not self.enabled:
            # Lights off – almost black
            result = (result * 0.05).clip(0, 255).astype(np.uint8)
            return result

        # 1. Vignette mask
        result = result * self.vignette_mask

        # 2. Intensity scaling
        result = result * self.intensity

        # 3. Color temperature tint
        result = self._apply_color_temp(result)

        return result.clip(0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_vignette_mask(self):
        """Build a circular spotlight mask matching endoscope optics."""
        cx, cy = self.width // 2, self.height // 2
        Y, X = np.ogrid[:self.height, :self.width]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        radius = min(cx, cy) * 0.92
        mask = np.clip(1.0 - (dist / radius) ** 2.2, 0.0, 1.0)
        # Stack to 3 channels
        self.vignette_mask = np.stack([mask] * 3, axis=-1)

    def _apply_color_temp(self, frame: np.ndarray) -> np.ndarray:
        if self.color_temp == "warm":
            frame[:, :, 2] *= 1.12   # boost red
            frame[:, :, 0] *= 0.88   # reduce blue
        elif self.color_temp == "cool":
            frame[:, :, 0] *= 1.12   # boost blue
            frame[:, :, 2] *= 0.88   # reduce red
        return frame

    # ------------------------------------------------------------------
    # Properties (for UI display)
    # ------------------------------------------------------------------

    @property
    def intensity_percent(self) -> int:
        return int(self.intensity * 50)   # 0–100

    def resize(self, width: int, height: int):
        self.width = width
        self.height = height
        self._build_vignette_mask()