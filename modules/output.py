"""
Display & Output System Module
Handles image capture (screenshots), video recording,
and the HUD overlay drawn on frames.
"""

import cv2
import numpy as np
import os
import time


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")


class OutputSystem:
    """
    Manages image/video saving and adds the HUD overlay to frames.
    """

    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.recording = False
        self._writer = None
        self._rec_path = None
        self.captured_images = []   # list of file paths
        self.show_hud = True
        self.show_crosshair = True
        self.show_histogram = False

        # Frame counter for FPS calculation
        self._fps_t = time.time()
        self._fps_count = 0
        self._fps = 0.0

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture_image(self, frame: np.ndarray) -> str:
        """Save current frame as a JPEG, return file path."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(OUTPUT_DIR, f"capture_{ts}.jpg")
        cv2.imwrite(path, frame)
        self.captured_images.append(path)
        return path

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------

    def start_recording(self, width: int, height: int):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._rec_path = os.path.join(OUTPUT_DIR, f"video_{ts}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._writer = cv2.VideoWriter(self._rec_path, fourcc, 20,
                                       (width, height))
        self.recording = True

    def stop_recording(self) -> str:
        if self._writer:
            self._writer.release()
            self._writer = None
        self.recording = False
        return self._rec_path

    def write_frame(self, frame: np.ndarray):
        if self.recording and self._writer:
            self._writer.write(frame)

    # ------------------------------------------------------------------
    # HUD overlay (drawn on a copy of the frame)
    # ------------------------------------------------------------------

    def draw_hud(self, frame: np.ndarray, nav_status: dict,
                 illumination_intensity: int,
                 mode: str) -> np.ndarray:
        """Draw HUD elements onto the frame. Returns annotated copy."""
        if frame is None:
            return frame

        out = frame.copy()
        self._update_fps()

        if not self.show_hud:
            if self.show_crosshair:
                self._draw_crosshair(out)
            return out

        H, W = out.shape[:2]

        # Semi-transparent dark bar at bottom
        overlay = out.copy()
        cv2.rectangle(overlay, (0, H - 72), (W, H), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

        # Mode badge top-left
        badge_color = (60, 140, 60) if mode == "simulation" else (60, 100, 180)
        mode_label = "SIM" if mode == "simulation" else "CAM"
        cv2.rectangle(out, (8, 8), (68, 30), badge_color, -1, cv2.LINE_AA)
        cv2.putText(out, mode_label, (14, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1,
                    cv2.LINE_AA)

        # FPS badge
        cv2.putText(out, f"{self._fps:.0f} FPS", (W - 72, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1,
                    cv2.LINE_AA)

        # REC indicator
        if self.recording:
            cv2.circle(out, (W // 2, 18), 7, (0, 0, 220), -1, cv2.LINE_AA)
            cv2.putText(out, "REC", (W // 2 + 12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 220), 1,
                        cv2.LINE_AA)

        # Bottom bar: nav values
        x = 10
        y_base = H - 48
        items = [
            ("↑↓", nav_status.get("Up/Down", "0°")),
            ("←→", nav_status.get("Left/Right", "0°")),
            ("↺", nav_status.get("Rotation", "0°")),
            ("↧", nav_status.get("Depth", "0 cm")),
            ("💡", f"{illumination_intensity}%"),
        ]
        for label, val in items:
            cv2.putText(out, f"{label} {val}", (x, y_base),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1,
                        cv2.LINE_AA)
            x += 110

        # Crosshair
        if self.show_crosshair:
            self._draw_crosshair(out)

        # Histogram inset
        if self.show_histogram:
            self._draw_histogram(out, frame)

        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_crosshair(self, frame: np.ndarray):
        H, W = frame.shape[:2]
        cx, cy = W // 2, H // 2
        color = (200, 200, 200)
        cv2.line(frame, (cx - 18, cy), (cx + 18, cy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 18), (cx, cy + 18), color, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 22, color, 1, cv2.LINE_AA)

    def _draw_histogram(self, out: np.ndarray, frame: np.ndarray):
        H, W = out.shape[:2]
        hist_w, hist_h = 120, 60
        x0, y0 = W - hist_w - 10, 40
        cv2.rectangle(out, (x0 - 2, y0 - 2),
                      (x0 + hist_w + 2, y0 + hist_h + 2),
                      (50, 50, 50), -1)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i, col in enumerate(colors):
            hist = cv2.calcHist([frame], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
            for j in range(len(hist) - 1):
                x1 = x0 + int(j * hist_w / 32)
                x2 = x0 + int((j + 1) * hist_w / 32)
                y1 = y0 + hist_h - int(hist[j])
                y2 = y0 + hist_h - int(hist[j + 1])
                cv2.line(out, (x1, y1), (x2, y2), col, 1)

    def _update_fps(self):
        self._fps_count += 1
        now = time.time()
        if now - self._fps_t >= 1.0:
            self._fps = self._fps_count / (now - self._fps_t)
            self._fps_count = 0
            self._fps_t = now