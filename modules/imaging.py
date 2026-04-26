"""
Imaging System Module
Handles camera input OR synthetic endoscope tissue simulation.
Redesigned: proper circular endoscope viewport with visible light vignette,
richer mucosal tissue, better polyp rendering.
"""

import cv2
import numpy as np
import time
import math


class ImagingSystem:
    """
    Provides frames either from a real webcam or from a procedurally
    generated synthetic endoscope view (colon/tissue simulation).

    Key visual improvements:
    - Circular mask so the frame looks like a real endoscope portal
    - Soft gradient vignette that shows the illumination cone clearly
    - Multi-layer tissue texture for a more organic mucosal surface
    - More prominent, realistic polyp with shading
    - Subtle specular highlight pool at the scope tip
    """

    MODE_CAMERA     = "camera"
    MODE_SIMULATION = "simulation"

    def __init__(self, width: int = 520, height: int = 520):
        self.width  = width
        self.height = height
        self.mode   = self.MODE_SIMULATION
        self.cap    = None
        self._t0    = time.time()

        # Filters
        self.zoom_factor        = 1.0
        self.apply_sharpen      = False
        self.apply_denoise      = False
        self.apply_contrast     = False
        self.apply_edge_enhance = False
        self.flip_h             = False
        self.show_polyp         = True

        # Precompute reusable masks (expensive, done once)
        self._build_masks()

        self._try_open_camera()

    # ──────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ──────────────────────────────────────────────────────────────────

    def _try_open_camera(self):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap  = cap
            self.mode = self.MODE_CAMERA
        else:
            self.mode = self.MODE_SIMULATION

    def _build_masks(self):
        """Precompute the circular scope mask and a radial distance map."""
        H, W   = self.height, self.width
        cx, cy = W // 2, H // 2
        Y, X   = np.ogrid[:H, :W]
        dist   = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        # Radius slightly inset so a hard black ring is visible
        self._scope_radius = min(cx, cy) * 0.96
        self._dist_map     = dist          # raw pixel distances (float32)

        # Circular binary mask (True inside circle)
        self._circle_mask_bool = dist <= self._scope_radius

        # Smooth alpha for the port edge (anti-aliased 1→0 over ~4px ring)
        edge_feather = 4.0
        self._port_alpha = np.clip(
            (self._scope_radius - dist) / edge_feather, 0.0, 1.0
        ).astype(np.float32)

        # 3-channel stacks for fast multiply
        self._port_alpha3 = np.stack([self._port_alpha] * 3, axis=-1)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def get_frame(self) -> np.ndarray:
        if self.mode == self.MODE_CAMERA and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.flip_h:
                    frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (self.width, self.height))
                frame = self._apply_filters(frame)
                return self._apply_scope_mask(frame)

        return self._apply_filters(self._generate_simulation_frame())

    def set_zoom(self, factor: float):
        self.zoom_factor = max(1.0, min(4.0, float(factor)))

    def toggle_filter(self, name: str, state: bool):
        setattr(self, f"apply_{name}", state)

    def set_mode(self, mode: str):
        self.mode = mode

    def release(self):
        if self.cap:
            self.cap.release()

    # ──────────────────────────────────────────────────────────────────
    # Synthetic frame generator
    # ──────────────────────────────────────────────────────────────────

    def _generate_simulation_frame(self) -> np.ndarray:
        t = time.time() - self._t0
        H, W = self.height, self.width

        # ── 1. Base tissue colour (animate gently) ────────────────────
        base_r = 195 + 25 * math.sin(t * 0.28)
        base_g =  80 + 18 * math.sin(t * 0.37)
        base_b =  72 + 12 * math.sin(t * 0.45)

        frame = np.empty((H, W, 3), dtype=np.float32)
        frame[:, :, 0] = base_b
        frame[:, :, 1] = base_g
        frame[:, :, 2] = base_r

        # ── 2. Mucosal fold texture ───────────────────────────────────
        Y, X = np.mgrid[0:H, 0:W].astype(np.float32)

        # Large folds (scale with frame size)
        fold1 = (np.sin(X / (W * 0.10) + t * 0.18) *
                 np.cos(Y / (H * 0.14) + t * 0.13) * 42)
        # Medium corrugation
        fold2 = (np.sin(X / (W * 0.045) + Y / (H * 0.055) + t * 0.25) * 22)
        # Fine mucosal grain
        fold3 = (np.sin(X / (W * 0.018) - Y / (H * 0.022) + t * 0.42) * 10)
        # Slow wave (peristaltic illusion)
        fold4 = (np.sin(Y / (H * 0.25) + t * 0.6) * 18)

        texture = (fold1 + fold2 + fold3 + fold4).astype(np.float32)

        frame[:, :, 0] += texture * 0.25
        frame[:, :, 1] += texture * 0.38
        frame[:, :, 2] += texture * 0.55

        # ── 3. Depth shading – darker toward edges ────────────────────
        # Radial depth: centre is lit, periphery is shadowed
        depth = np.clip(1.0 - (self._dist_map / self._scope_radius) ** 1.6,
                        0.0, 1.0).astype(np.float32)
        depth3 = np.stack([depth] * 3, axis=-1)
        # Strong central illumination + softer periphery
        illum = 0.35 + 0.65 * depth3
        frame *= illum

        # ── 4. Blood vessels ──────────────────────────────────────────
        frame = self._draw_vessels(frame, t)

        # ── 5. Specular pool (reflection of LED on mucosa) ────────────
        frame = self._draw_specular(frame, t)

        # ── 6. Polyp ──────────────────────────────────────────────────
        if self.show_polyp:
            frame = self._draw_polyp(frame, t)

        # ── 7. Breathing motion ───────────────────────────────────────
        dx = int(6 * math.sin(t * 0.65))
        dy = int(4 * math.sin(t * 0.85))
        M  = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(frame, M, (W, H))

        frame = frame.clip(0, 255).astype(np.uint8)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        # ── 8. Apply circular scope mask ─────────────────────────────
        return self._apply_scope_mask(frame)

    # ──────────────────────────────────────────────────────────────────
    # Drawing helpers
    # ──────────────────────────────────────────────────────────────────

    def _draw_vessels(self, frame: np.ndarray, t: float) -> np.ndarray:
        H, W    = frame.shape[:2]
        overlay = frame.copy()
        # Dark burgundy in BGR
        vc = np.array([38, 28, 140], dtype=np.float32)

        paths = [
            [(int(W * 0.22 + 28 * math.sin(i / (H * 0.032) + t * 0.10)),
              int(i)) for i in range(0, H, 2)],
            [(int(W * 0.58 + 20 * math.cos(i / (H * 0.026) + t * 0.14)),
              int(i)) for i in range(0, H, 2)],
            [(int(i),
              int(H * 0.52 + 14 * math.sin(i / (W * 0.04) + t * 0.11)))
             for i in range(0, W, 2)],
            # Smaller branching vessel
            [(int(W * 0.38 + 12 * math.sin(i / (H * 0.06) + t * 0.2)),
              int(H * 0.2 + i * 0.4)) for i in range(0, int(H * 0.5), 2)],
        ]
        for path in paths:
            for j in range(len(path) - 1):
                x1, y1 = path[j]
                x2, y2 = path[j + 1]
                if (0 <= x1 < W and 0 <= y1 < H and
                        0 <= x2 < W and 0 <= y2 < H):
                    cv2.line(overlay, (x1, y1), (x2, y2),
                             vc.tolist(), 1, cv2.LINE_AA)

        return cv2.addWeighted(frame, 0.82, overlay, 0.18, 0)

    def _draw_specular(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Simulate glinting specular reflection from fluid on tissue."""
        H, W = frame.shape[:2]
        cx   = int(W * 0.48 + 12 * math.sin(t * 0.4))
        cy   = int(H * 0.44 + 8  * math.sin(t * 0.55))
        rx, ry = 22, 14

        spec = frame.copy()
        # Bright white-ish pool
        cv2.ellipse(spec, (cx, cy), (rx, ry), 20, 0, 360,
                    (240, 240, 255), -1, cv2.LINE_AA)
        # Inner bright highlight
        cv2.ellipse(spec, (cx - 4, cy - 4), (rx // 3, ry // 3), 10,
                    0, 360, (255, 255, 255), -1, cv2.LINE_AA)

        return cv2.addWeighted(frame, 0.88, spec, 0.12, 0)

    def _draw_polyp(self, frame: np.ndarray, t: float) -> np.ndarray:
        H, W = frame.shape[:2]
        cx   = int(W * 0.60)
        cy   = int(H * 0.42)

        pulse = 1.0 + 0.05 * math.sin(t * 1.8)
        rx    = int(34 * pulse)
        ry    = int(26 * pulse)

        overlay = frame.copy()

        # Shadow (dark halo around polyp base)
        cv2.ellipse(overlay, (cx + 3, cy + 4), (rx + 5, ry + 4),
                    0, 0, 360, (20, 10, 60), -1, cv2.LINE_AA)

        # Polyp body – reddish-purple mass
        cv2.ellipse(overlay, (cx, cy), (rx, ry),
                    0, 0, 360, (70, 50, 200), -1, cv2.LINE_AA)

        # Mid-tone shading (bottom half)
        cv2.ellipse(overlay, (cx + 2, cy + 4), (rx - 4, ry - 6),
                    0, 0, 360, (55, 38, 165), -1, cv2.LINE_AA)

        # Specular highlight
        cv2.ellipse(overlay, (cx - 8, cy - 8),
                    (rx // 4, ry // 4), 30, 0, 360,
                    (180, 140, 255), -1, cv2.LINE_AA)
        cv2.ellipse(overlay, (cx - 9, cy - 9),
                    (rx // 8, ry // 8), 30, 0, 360,
                    (240, 220, 255), -1, cv2.LINE_AA)

        # Outline
        cv2.ellipse(overlay, (cx, cy), (rx, ry),
                    0, 0, 360, (40, 20, 130), 2, cv2.LINE_AA)

        # Stalk (polyp attachment to wall)
        stalk_pts = np.array([
            [cx - 6, cy + ry - 2],
            [cx - 2, cy + ry + 14],
            [cx + 2, cy + ry + 14],
            [cx + 6, cy + ry - 2],
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [stalk_pts], (50, 35, 160))

        return cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)

    # ──────────────────────────────────────────────────────────────────
    # Circular scope mask  (makes light vignette visible!)
    # ──────────────────────────────────────────────────────────────────

    def _apply_scope_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Composite the tissue image inside the circular port.
        Everything outside the port is pure black – this is what makes
        the illumination vignette/spotlight clearly visible.
        """
        H, W = frame.shape[:2]

        # Rebuild masks only if size changed
        if H != self.height or W != self.width:
            self.height = H
            self.width  = W
            self._build_masks()

        f32   = frame.astype(np.float32)
        black = np.zeros_like(f32)

        # Blend tissue → black at the feathered edge
        result = f32 * self._port_alpha3 + black * (1.0 - self._port_alpha3)

        # Hard scope ring: thin dark-grey ring just inside the mask edge
        ring_inner = self._scope_radius - 6
        ring_mask  = ((self._dist_map >= ring_inner) &
                      (self._dist_map <= self._scope_radius + 2))
        result[ring_mask] = result[ring_mask] * 0.15   # very dark ring

        return result.clip(0, 255).astype(np.uint8)

    # ──────────────────────────────────────────────────────────────────
    # Filter pipeline
    # ──────────────────────────────────────────────────────────────────

    def _apply_filters(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return frame

        if self.zoom_factor > 1.0:
            frame = self._apply_zoom(frame)

        if self.apply_sharpen:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            frame  = cv2.filter2D(frame, -1, kernel)

        if self.apply_denoise:
            frame = cv2.fastNlMeansDenoisingColored(frame, None, 6, 6, 7, 21)

        if self.apply_contrast:
            lab       = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b   = cv2.split(lab)
            clahe     = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l         = clahe.apply(l)
            lab       = cv2.merge((l, a, b))
            frame     = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if self.apply_edge_enhance:
            edges     = cv2.Canny(frame, 40, 120)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            frame     = cv2.addWeighted(frame, 0.85, edges_bgr, 0.30, 0)

        return frame

    def _apply_zoom(self, frame: np.ndarray) -> np.ndarray:
        H, W   = frame.shape[:2]
        z      = self.zoom_factor
        new_w  = int(W / z)
        new_h  = int(H / z)
        x1     = (W - new_w) // 2
        y1     = (H - new_h) // 2
        return cv2.resize(frame[y1:y1 + new_h, x1:x1 + new_w],
                          (W, H), interpolation=cv2.INTER_LINEAR)