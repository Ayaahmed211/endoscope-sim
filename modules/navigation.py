import math
import time


class NavigationSystem:
    """
    Tracks virtual endoscope position and orientation.
    Keyboard events (WASD / arrow keys) are translated into
    tip deflection (up/down/left/right) and rotation.
    """

    # Deflection limits in degrees
    MAX_UP_DOWN = 180      # ±180°
    MAX_LEFT_RIGHT = 160   # ±160°

    def __init__(self):
        self.deflection_up_down = 0.0      # degrees, + = up
        self.deflection_left_right = 0.0   # degrees, + = right
        self.rotation = 0.0                # shaft rotation degrees
        self.speed = 3.0                   # degrees per key-press

        # For smooth continuous movement
        self._keys_held = set()
        self._last_update = time.time()

        # Navigation log
        self.log = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def key_press(self, key_name: str):
        self._keys_held.add(key_name)

    def key_release(self, key_name: str):
        self._keys_held.discard(key_name)

    def update(self):
        """Call every frame to apply held-key movement."""
        now = time.time()
        dt = now - self._last_update #
        self._last_update = now
        step = self.speed * 60 * dt   # ~speed per frame at 60fps

        moved = False
        if "up" in self._keys_held or "w" in self._keys_held:
            self._move_up(step); moved = True
        if "down" in self._keys_held or "s" in self._keys_held:
            self._move_down(step); moved = True
        if "left" in self._keys_held or "a" in self._keys_held:
            self._move_left(step); moved = True
        if "right" in self._keys_held or "d" in self._keys_held:
            self._move_right(step); moved = True
        if "q" in self._keys_held:
            self._rotate(-step * 0.5); moved = True
        if "e" in self._keys_held:
            self._rotate(step * 0.5); moved = True

        if moved:
            self._log_position()

    # Direction helpers
    def _move_up(self, deg):
        self.deflection_up_down = min(self.MAX_UP_DOWN / 2,
                                      self.deflection_up_down + deg)

    def _move_down(self, deg):
        self.deflection_up_down = max(-self.MAX_UP_DOWN / 2,
                                      self.deflection_up_down - deg)

    def _move_left(self, deg):
        self.deflection_left_right = max(-self.MAX_LEFT_RIGHT / 2,
                                         self.deflection_left_right - deg)

    def _move_right(self, deg):
        self.deflection_left_right = min(self.MAX_LEFT_RIGHT / 2,
                                         self.deflection_left_right + deg)

    def _rotate(self, deg):
        self.rotation = (self.rotation + deg) % 360

    def reset(self):
        self.deflection_up_down = 0.0
        self.deflection_left_right = 0.0
        self.rotation = 0.0
        self.log.clear()

    # ------------------------------------------------------------------
    # Overlay helpers – affect how the frame is shifted/rotated
    # ------------------------------------------------------------------

    def get_frame_transform(self):
        """
        Returns (dx, dy, angle) to apply to the displayed frame,
        simulating camera tip pointing.
        """
        # Map deflection angles to pixel offsets (simple linear)
        max_px_h = 80
        max_px_v = 60
        dx = int(self.deflection_left_right / (self.MAX_LEFT_RIGHT / 2) * max_px_h)
        dy = int(-self.deflection_up_down / (self.MAX_UP_DOWN / 2) * max_px_v)
        angle = self.rotation
        return dx, dy, angle

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_position(self):
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "ud": round(self.deflection_up_down, 1),
            "lr": round(self.deflection_left_right, 1),
            "rot": round(self.rotation, 1),
        }
        self.log.append(entry)
        if len(self.log) > 200:
            self.log = self.log[-200:]

    # ------------------------------------------------------------------
    # Status dict for UI display
    # ------------------------------------------------------------------

    @property
    def status(self) -> dict:
        return {
            "Up/Down": f"{self.deflection_up_down:+.1f}°",
            "Left/Right": f"{self.deflection_left_right:+.1f}°",
            "Rotation": f"{self.rotation:.1f}°",
        }