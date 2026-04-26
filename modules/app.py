"""
Intelligent Endoscopic Assistance System
Main GUI Application (Tkinter + OpenCV)
Redesigned: rich clinical dashboard with full circular viewport
"""

import tkinter as tk
from tkinter import ttk, font as tkfont
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import threading
import time
import os
import math

from modules.illumination import IlluminationSystem
from modules.imaging import ImagingSystem
from modules.navigation import NavigationSystem
from modules.output import OutputSystem

# ──────────────────────────────────────────────────────────────────────
# Colour palette – deep navy clinical with electric-blue accents
# ──────────────────────────────────────────────────────────────────────
CLR = {
    "bg":           "#0B0F1A",       # near-black navy
    "panel":        "#111827",       # dark panel
    "panel2":       "#161E2E",       # slightly lighter panel
    "border":       "#1F2D45",       # subtle border
    "border2":      "#2A3F5F",       # brighter border
    "accent":       "#00A8FF",       # electric blue
    "accent_dark":  "#0078C0",
    "accent_glow":  "#00A8FF33",
    "accent2":      "#00E5CC",       # teal accent
    "text":         "#E8EDF5",
    "text_sub":     "#7B8FA6",
    "text_dim":     "#445566",
    "success":      "#00C48C",
    "danger":       "#FF4757",
    "warn":         "#FFB300",
    "tag_sim":      "#00C48C",
    "tag_cam":      "#00A8FF",
    "scope_ring":   "#0A1628",       # outer ring around scope view
    "scope_bg":     "#000000",       # scope background
}

FONT_HEAD  = ("Segoe UI Semibold", 10)
FONT_BODY  = ("Segoe UI", 9)
FONT_MONO  = ("Consolas", 9)
FONT_BIG   = ("Segoe UI Semibold", 13)
FONT_MICRO = ("Segoe UI", 8)
FONT_LABEL = ("Segoe UI", 8)

FRAME_W = 520
FRAME_H = 520   # Square so circle fits perfectly


class EndoscopeApp:
    """Main application window."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("EndoSim — Intelligent Endoscopic Assistance System")
        self.root.configure(bg=CLR["bg"])
        self.root.resizable(True, True)
        self.root.minsize(1100, 720)

        # Sub-systems
        self.illumination = IlluminationSystem(FRAME_W, FRAME_H)
        self.imaging      = ImagingSystem(FRAME_W, FRAME_H)
        self.navigation   = NavigationSystem()
        self.output       = OutputSystem()

        # State
        self._running   = False
        self._paused    = False
        self._status_msg = tk.StringVar(value="System ready.")
        self._current_frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self._frame_count   = 0
        self._fps           = 0.0
        self._t_fps         = time.time()

        self._build_ui()
        self._bind_keys()

    # ══════════════════════════════════════════════════════════════════
    # UI Construction
    # ══════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self._build_header()
        self._build_body()
        self._build_statusbar()

    # ── Header ────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=CLR["panel"], height=56)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)

        # Accent line at very bottom of header
        accent_line = tk.Frame(hdr, bg=CLR["accent"], height=2)
        accent_line.place(relx=0, rely=1.0, relwidth=1.0, anchor="sw")

        # Logo
        logo_frame = tk.Frame(hdr, bg=CLR["panel"])
        logo_frame.pack(side="left", padx=18, pady=10)

        canvas_logo = tk.Canvas(logo_frame, width=32, height=32,
                                bg=CLR["panel"], highlightthickness=0)
        canvas_logo.pack(side="left")
        # Draw hex icon
        self._draw_hex_logo(canvas_logo)

        tk.Label(logo_frame, text="EndoSim", bg=CLR["panel"],
                 fg=CLR["accent"], font=("Segoe UI Semibold", 14)).pack(
            side="left", padx=(6, 0))

        tk.Label(hdr, text="Intelligent Endoscopic Assistance System",
                 bg=CLR["panel"], fg=CLR["text_sub"],
                 font=("Segoe UI", 10)).pack(side="left", pady=10)

        # Right side info
        right = tk.Frame(hdr, bg=CLR["panel"])
        right.pack(side="right", padx=18)

        # Live pulse indicator
        self._pulse_canvas = tk.Canvas(right, width=10, height=10,
                                       bg=CLR["panel"], highlightthickness=0)
        self._pulse_canvas.pack(side="left", padx=(0, 6))
        self._pulse_dot = self._pulse_canvas.create_oval(1, 1, 9, 9,
                                                          fill=CLR["success"], outline="")

        self._live_lbl = tk.Label(right, text="LIVE",
                                  bg=CLR["panel"], fg=CLR["success"],
                                  font=("Segoe UI Semibold", 9))
        self._live_lbl.pack(side="left", padx=(0, 16))

        tk.Label(right, text="SBE3220 — Medical Equipment (II)",
                 bg=CLR["panel"], fg=CLR["text_dim"],
                 font=("Segoe UI", 8)).pack(side="left")

    def _draw_hex_logo(self, canvas):
        cx, cy, r = 16, 16, 13
        points = []
        for i in range(6):
            angle = math.radians(60 * i - 30)
            points.extend([cx + r * math.cos(angle), cy + r * math.sin(angle)])
        canvas.create_polygon(points, fill=CLR["accent"], outline="", smooth=False)
        canvas.create_text(cx, cy, text="E", fill="white",
                           font=("Segoe UI Semibold", 10))

    # ── Body ──────────────────────────────────────────────────────────
    def _build_body(self):
        body = tk.Frame(self.root, bg=CLR["bg"])
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0)   # left sidebar
        body.columnconfigure(1, weight=1)   # center viewport
        body.columnconfigure(2, weight=0)   # right sidebar
        body.rowconfigure(0, weight=1)

        self._build_left_sidebar(body)
        self._build_center(body)
        self._build_right_sidebar(body)

    # ── Left sidebar ─────────────────────────────────────────────────
    def _build_left_sidebar(self, parent):
        sb = tk.Frame(parent, bg=CLR["bg"], width=230)
        sb.grid(row=0, column=0, sticky="ns", padx=(10, 6), pady=10)
        sb.grid_propagate(False)

        # ── Illumination ─────────────────────────────────────────────
        ill = self._card(sb, "💡  Illumination System")
        ill._outer.pack(fill="x", pady=(0, 8))

        # On/Off
        onoff_row = tk.Frame(ill, bg=CLR["panel"])
        onoff_row.pack(fill="x", pady=(0, 6))
        self._field_label(onoff_row, "Light Source").pack(side="left")
        self._light_var = tk.BooleanVar(value=True)
        self._toggle_btn(onoff_row, self._light_var,
                         self._on_light_toggle).pack(side="right")

        # Intensity
        self._field_label(ill, "Intensity").pack(anchor="w")
        int_row = tk.Frame(ill, bg=CLR["panel"])
        int_row.pack(fill="x", pady=(2, 6))
        self._intensity_var = tk.DoubleVar(value=1.0)
        self._intensity_lbl = tk.Label(int_row, text="100%",
                                       bg=CLR["panel"], fg=CLR["accent"],
                                       font=FONT_MONO, width=5)
        self._intensity_lbl.pack(side="right")
        tk.Scale(int_row, from_=0.0, to=2.0, resolution=0.05,
                 orient="horizontal", variable=self._intensity_var,
                 bg=CLR["panel"], fg=CLR["text"],
                 troughcolor=CLR["border2"],
                 activebackground=CLR["accent"],
                 highlightthickness=0, showvalue=False,
                 command=self._on_intensity_change).pack(side="left", fill="x",
                                                          expand=True)

        # Color temperature
        self._field_label(ill, "Color Temperature").pack(anchor="w")
        temp_row = tk.Frame(ill, bg=CLR["panel"])
        temp_row.pack(fill="x", pady=(2, 0))
        self._temp_var = tk.StringVar(value="white")
        for val, label, clr in [("cool", "Cool", "#88CCFF"),
                                 ("white", "White", "#FFFFFF"),
                                 ("warm", "Warm", "#FFAA44")]:
            rb = tk.Radiobutton(temp_row, text=label, variable=self._temp_var,
                                value=val, bg=CLR["panel"],
                                fg=clr, selectcolor=CLR["panel"],
                                activebackground=CLR["panel"],
                                font=FONT_MICRO,
                                indicatoron=0,
                                relief="flat", bd=0,
                                command=self._on_temp_change,
                                padx=6, pady=3, cursor="hand2")
            rb.pack(side="left", padx=2)

        # ── Imaging ──────────────────────────────────────────────────
        img = self._card(sb, "🎥  Imaging System")
        img._outer.pack(fill="x", pady=(0, 8))

        # Source
        self._field_label(img, "Source").pack(anchor="w")
        src_row = tk.Frame(img, bg=CLR["panel"])
        src_row.pack(fill="x", pady=(2, 6))
        self._src_var = tk.StringVar(value=self.imaging.mode)
        for val, label in [("simulation", "Simulation"), ("camera", "Camera")]:
            tk.Radiobutton(src_row, text=label, variable=self._src_var,
                           value=val, bg=CLR["panel"], fg=CLR["text"],
                           selectcolor=CLR["panel"],
                           activebackground=CLR["panel"],
                           font=FONT_BODY,
                           command=self._on_source_change).pack(side="left",
                                                                 padx=4)

        # Zoom
        self._field_label(img, "Zoom").pack(anchor="w")
        zm_row = tk.Frame(img, bg=CLR["panel"])
        zm_row.pack(fill="x", pady=(2, 6))
        self._zoom_var = tk.DoubleVar(value=1.0)
        self._zoom_lbl = tk.Label(zm_row, text="1.0×",
                                  bg=CLR["panel"], fg=CLR["accent"],
                                  font=FONT_MONO, width=5)
        self._zoom_lbl.pack(side="right")
        tk.Scale(zm_row, from_=1.0, to=4.0, resolution=0.1,
                 orient="horizontal", variable=self._zoom_var,
                 bg=CLR["panel"], fg=CLR["text"],
                 troughcolor=CLR["border2"],
                 highlightthickness=0, showvalue=False,
                 command=self._on_zoom_change).pack(side="left", fill="x",
                                                     expand=True)

        # Filters — compact 2×2 grid
        self._field_label(img, "Image Filters").pack(anchor="w", pady=(4, 2))
        flt_frame = tk.Frame(img, bg=CLR["panel"])
        flt_frame.pack(fill="x")
        self._filter_vars = {}
        filters = [("sharpen", "Sharpen"), ("denoise", "Denoise"),
                   ("contrast", "CLAHE"),  ("edge_enhance", "Edges")]
        for i, (key, label) in enumerate(filters):
            var = tk.BooleanVar(value=False)
            self._filter_vars[key] = var
            self._chk(flt_frame, label, var,
                      lambda k=key, v=var: self._on_filter_change(k, v)
                      ).grid(row=i // 2, column=i % 2, sticky="w", padx=2)

        # Polyp toggle
        sep = tk.Frame(img, bg=CLR["border"], height=1)
        sep.pack(fill="x", pady=6)
        self._polyp_var = tk.BooleanVar(value=True)
        self._chk(img, "Show Polyp (simulation)",
                  self._polyp_var,
                  self._on_polyp_toggle).pack(anchor="w")

        # ── Keyboard Reference ────────────────────────────────────────
        ref = self._card(sb, "⌨  Keyboard Reference")
        ref._outer.pack(fill="x")

        keys = [
            ("W / ↑", "Tip Up"),       ("S / ↓", "Tip Down"),
            ("A / ←", "Tip Left"),     ("D / →", "Tip Right"),
            ("Q",     "Rotate ↺"),     ("E",     "Rotate ↻"),
            ("C",     "Capture"),      ("R",     "Record"),
            ("Space", "Pause"),        ("Esc",   "Stop"),
        ]
        key_grid = tk.Frame(ref, bg=CLR["panel"])
        key_grid.pack(fill="x")
        for i, (k, v) in enumerate(keys):
            row_f = tk.Frame(key_grid, bg=CLR["panel"])
            row_f.grid(row=i, column=0, sticky="ew", pady=1)
            key_grid.columnconfigure(0, weight=1)
            tk.Label(row_f, text=k, bg=CLR["border2"],
                     fg=CLR["accent"], font=("Consolas", 8),
                     width=8, anchor="center",
                     pady=2).pack(side="left")
            tk.Label(row_f, text=v, bg=CLR["panel"],
                     fg=CLR["text_sub"], font=FONT_MICRO,
                     anchor="w").pack(side="left", padx=8)

    # ── Center viewport ───────────────────────────────────────────────
    def _build_center(self, parent):
        center = tk.Frame(parent, bg=CLR["bg"])
        center.grid(row=0, column=1, sticky="nsew", padx=6, pady=10)
        center.columnconfigure(0, weight=1)
        center.rowconfigure(0, weight=1)

        # ── Scope viewer card ─────────────────────────────────────────
        scope_card = tk.Frame(center, bg=CLR["panel"],
                              highlightthickness=1,
                              highlightbackground=CLR["border2"])
        scope_card.grid(row=0, column=0, sticky="nsew")
        scope_card.columnconfigure(0, weight=1)
        scope_card.rowconfigure(1, weight=1)

        # Top bar inside scope card
        top_bar = tk.Frame(scope_card, bg=CLR["panel2"], height=36)
        top_bar.grid(row=0, column=0, sticky="ew")
        top_bar.grid_propagate(False)

        tk.Label(top_bar, text="Live Endoscope View",
                 bg=CLR["panel2"], fg=CLR["text"],
                 font=FONT_HEAD).pack(side="left", padx=14, pady=8)

        self._mode_badge = tk.Label(top_bar, text="● SIMULATION",
                                    bg=CLR["tag_sim"], fg="white",
                                    font=("Segoe UI Semibold", 8),
                                    padx=8, pady=3)
        self._mode_badge.pack(side="left", padx=6, pady=6)

        self._rec_badge = tk.Label(top_bar, text="⏺ REC",
                                   bg=CLR["danger"], fg="white",
                                   font=("Segoe UI Semibold", 8),
                                   padx=8, pady=3)
        # hidden until recording

        self._fps_lbl = tk.Label(top_bar, text="0 fps",
                                 bg=CLR["panel2"], fg=CLR["text_dim"],
                                 font=FONT_MONO)
        self._fps_lbl.pack(side="right", padx=14)

        # Canvas for video (dark surround)
        canvas_wrap = tk.Frame(scope_card, bg=CLR["scope_bg"])
        canvas_wrap.grid(row=1, column=0, sticky="nsew")
        canvas_wrap.columnconfigure(0, weight=1)
        canvas_wrap.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_wrap,
                                width=FRAME_W, height=FRAME_H,
                                bg=CLR["scope_bg"],
                                highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=20, pady=16)

        # Control bar
        ctrl_bar = tk.Frame(scope_card, bg=CLR["panel2"], height=46)
        ctrl_bar.grid(row=2, column=0, sticky="ew")
        ctrl_bar.grid_propagate(False)

        btn_data = [
            ("▶  Start",    self.start,           CLR["accent"]),
            ("⏸  Pause",   self.toggle_pause,     CLR["border2"]),
            ("⏹  Stop",    self.stop,             CLR["danger"]),
            ("📷  Capture", self.capture_image,    CLR["success"]),
        ]
        for label, cmd, bg in btn_data:
            self._cta_btn(ctrl_bar, label, cmd, bg).pack(side="left",
                                                          padx=4, pady=8)

        tk.Frame(ctrl_bar, width=12, bg=CLR["panel2"]).pack(side="left")

        self._rec_btn = self._cta_btn(ctrl_bar, "⏺  Record",
                                       self.toggle_recording,
                                       CLR["border2"])
        self._rec_btn.pack(side="left", padx=4, pady=8)

        # ── Navigation panel ──────────────────────────────────────────
        self._build_navigation_panel(center)

    def _build_navigation_panel(self, parent):
        nav_card = tk.Frame(parent, bg=CLR["panel"],
                            highlightthickness=1,
                            highlightbackground=CLR["border2"])
        nav_card.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        nav_card.columnconfigure(1, weight=1)

        # Section label
        top = tk.Frame(nav_card, bg=CLR["panel2"], height=30)
        top.grid(row=0, column=0, columnspan=3, sticky="ew")
        top.grid_propagate(False)
        tk.Label(top, text="Navigation & Insertion Control",
                 bg=CLR["panel2"], fg=CLR["text"],
                 font=FONT_HEAD).pack(side="left", padx=14, pady=5)

        body = tk.Frame(nav_card, bg=CLR["panel"])
        body.grid(row=1, column=0, columnspan=3, sticky="ew",
                  padx=14, pady=8)

        # D-pad
        self._build_dpad(body)

        # Separator
        tk.Frame(body, width=1, bg=CLR["border"]).pack(side="left",
                                                         fill="y", padx=16)

        # Stats readout
        stats = tk.Frame(body, bg=CLR["panel"])
        stats.pack(side="left", fill="both", expand=True)

        self._nav_vars = {}
        labels = [("Up/Down", "°"), ("Left/Right", "°"),
                  ("Rotation", "°"), ("Depth", " cm")]
        for i, (lbl, unit) in enumerate(labels):
            r = tk.Frame(stats, bg=CLR["panel"])
            r.pack(fill="x", pady=2)
            tk.Label(r, text=lbl + ":", bg=CLR["panel"],
                     fg=CLR["text_dim"], font=FONT_LABEL,
                     width=11, anchor="e").pack(side="left")
            var = tk.StringVar(value=f"0.0{unit}")
            self._nav_vars[lbl] = var
            # Colored value
            tk.Label(r, textvariable=var, bg=CLR["panel"],
                     fg=CLR["accent2"], font=FONT_MONO,
                     anchor="w", width=10).pack(side="left")

        self._btn(stats, "↺ Reset All",
                  self._reset_navigation,
                  bg=CLR["border2"]).pack(anchor="w", pady=(6, 0))

        # Depth slider
        tk.Frame(body, width=1, bg=CLR["border"]).pack(side="left",
                                                         fill="y", padx=16)
        depth_frame = tk.Frame(body, bg=CLR["panel"])
        depth_frame.pack(side="left")
        tk.Label(depth_frame, text="Insertion\nDepth",
                 bg=CLR["panel"], fg=CLR["text_sub"],
                 font=FONT_MICRO, justify="center").pack()
        self._depth_var = tk.DoubleVar(value=0)
        tk.Scale(depth_frame, from_=150, to=0,
                 orient="vertical", length=80,
                 variable=self._depth_var,
                 bg=CLR["panel"], fg=CLR["accent"],
                 troughcolor=CLR["border2"],
                 activebackground=CLR["accent"],
                 highlightthickness=0,
                 command=self._on_depth_slide).pack()
        tk.Label(depth_frame, text="cm",
                 bg=CLR["panel"], fg=CLR["text_dim"],
                 font=FONT_MICRO).pack()

    def _build_dpad(self, parent):
        dpad_wrap = tk.Frame(parent, bg=CLR["panel"])
        dpad_wrap.pack(side="left")

        tk.Label(dpad_wrap, text="Tip Deflection",
                 bg=CLR["panel"], fg=CLR["text_dim"],
                 font=FONT_MICRO).grid(row=0, column=0,
                                       columnspan=3, pady=(0, 4))

        def dpad_btn(text):
            return tk.Button(dpad_wrap, text=text,
                             bg=CLR["panel2"], fg=CLR["text"],
                             activebackground=CLR["accent"],
                             activeforeground="white",
                             font=("Segoe UI", 10),
                             relief="flat", width=3, height=1,
                             cursor="hand2",
                             highlightthickness=1,
                             highlightbackground=CLR["border2"])

        up_btn   = dpad_btn("▲"); up_btn.grid(row=1, column=1, pady=2, padx=2)
        left_btn = dpad_btn("◄"); left_btn.grid(row=2, column=0, padx=2)
        ctr = tk.Frame(dpad_wrap, width=36, height=36, bg=CLR["border2"])
        ctr.grid(row=2, column=1, padx=2)
        right_btn= dpad_btn("►"); right_btn.grid(row=2, column=2, padx=2)
        down_btn = dpad_btn("▼"); down_btn.grid(row=3, column=1, pady=2, padx=2)

        for btn, key in [(up_btn,"up"),(left_btn,"left"),
                         (right_btn,"right"),(down_btn,"down")]:
            btn.bind("<ButtonPress-1>",   lambda e, k=key: self.navigation.key_press(k))
            btn.bind("<ButtonRelease-1>", lambda e, k=key: self.navigation.key_release(k))

        # Rotation row
        rot = tk.Frame(dpad_wrap, bg=CLR["panel"])
        rot.grid(row=4, column=0, columnspan=3, pady=(8, 0))
        tk.Label(rot, text="Rotate:", bg=CLR["panel"],
                 fg=CLR["text_sub"], font=FONT_MICRO).pack(side="left")
        for txt, key in [("↺ Q", "q"), ("↻ E", "e")]:
            b = tk.Button(rot, text=txt, bg=CLR["panel2"],
                          fg=CLR["accent2"],
                          activebackground=CLR["accent"],
                          activeforeground="white",
                          font=FONT_MICRO, relief="flat",
                          cursor="hand2", padx=6,
                          highlightthickness=1,
                          highlightbackground=CLR["border2"])
            b.pack(side="left", padx=3)
            b.bind("<ButtonPress-1>",   lambda e, k=key: self.navigation.key_press(k))
            b.bind("<ButtonRelease-1>", lambda e, k=key: self.navigation.key_release(k))

    # ── Right sidebar ─────────────────────────────────────────────────
    def _build_right_sidebar(self, parent):
        sb = tk.Frame(parent, bg=CLR["bg"], width=220)
        sb.grid(row=0, column=2, sticky="ns", padx=(6, 10), pady=10)
        sb.grid_propagate(False)

        # ── Vitals / metrics panel ────────────────────────────────────
        vitals = self._card(sb, "📊  Live Metrics")
        vitals._outer.pack(fill="x", pady=(0, 8))

        self._metric_vars = {}
        metrics = [
            ("Frame Rate", "fps"),
            ("Light Power", "%"),
            ("Zoom Level", "×"),
            ("Insertion",  "cm"),
        ]
        for i, (name, unit) in enumerate(metrics):
            row_f = tk.Frame(vitals, bg=CLR["panel"])
            row_f.pack(fill="x", pady=3)
            tk.Label(row_f, text=name, bg=CLR["panel"],
                     fg=CLR["text_sub"], font=FONT_MICRO,
                     anchor="w").pack(side="left")
            var = tk.StringVar(value=f"—")
            self._metric_vars[name] = (var, unit)
            val_lbl = tk.Label(row_f, textvariable=var,
                               bg=CLR["panel"], fg=CLR["accent"],
                               font=("Consolas", 10, "bold"),
                               anchor="e")
            val_lbl.pack(side="right")
            tk.Label(row_f, text=unit, bg=CLR["panel"],
                     fg=CLR["text_dim"], font=FONT_MICRO).pack(side="right")

            # Thin separator
            if i < len(metrics) - 1:
                tk.Frame(vitals, bg=CLR["border"], height=1).pack(
                    fill="x", pady=(2, 0))

        # ── Display Options ───────────────────────────────────────────
        disp = self._card(sb, "🖥  Display Options")
        disp._outer.pack(fill="x", pady=(0, 8))

        self._hud_var   = tk.BooleanVar(value=True)
        self._xhair_var = tk.BooleanVar(value=True)
        self._hist_var  = tk.BooleanVar(value=False)

        self._chk(disp, "Show HUD Overlay", self._hud_var,
                  lambda: setattr(self.output, "show_hud",
                                  self._hud_var.get())).pack(anchor="w", pady=2)
        self._chk(disp, "Show Crosshair", self._xhair_var,
                  lambda: setattr(self.output, "show_crosshair",
                                  self._xhair_var.get())).pack(anchor="w", pady=2)
        self._chk(disp, "Show Histogram", self._hist_var,
                  lambda: setattr(self.output, "show_histogram",
                                  self._hist_var.get())).pack(anchor="w", pady=2)

        # ── Captured files log ────────────────────────────────────────
        log_card = self._card(sb, "📁  Captured Files")
        log_card._outer.pack(fill="x", pady=(0, 8))

        self._log_text = tk.Text(log_card, height=5, font=FONT_MONO,
                                 bg=CLR["scope_bg"], fg=CLR["accent"],
                                 insertbackground=CLR["accent"],
                                 relief="flat", wrap="none",
                                 state="disabled",
                                 highlightthickness=1,
                                 highlightbackground=CLR["border"])
        self._log_text.pack(fill="x")

        # ── System Status ─────────────────────────────────────────────
        sys_card = self._card(sb, "⚙  System Status")
        sys_card._outer.pack(fill="x")

        status_items = [
            ("Camera", "Detected" if self.imaging.cap else "Not Found",
             CLR["success"] if self.imaging.cap else CLR["danger"]),
            ("Recorder", "Ready",    CLR["success"]),
            ("Navigation", "Online", CLR["success"]),
        ]
        for name, val, clr in status_items:
            r = tk.Frame(sys_card, bg=CLR["panel"])
            r.pack(fill="x", pady=2)
            # Status dot
            dot = tk.Canvas(r, width=8, height=8, bg=CLR["panel"],
                            highlightthickness=0)
            dot.pack(side="left", padx=(0, 6))
            dot.create_oval(1, 1, 7, 7, fill=clr, outline="")
            tk.Label(r, text=name, bg=CLR["panel"],
                     fg=CLR["text_sub"], font=FONT_MICRO,
                     anchor="w").pack(side="left")
            tk.Label(r, text=val, bg=CLR["panel"],
                     fg=clr, font=FONT_MICRO,
                     anchor="e").pack(side="right")

    # ── Status bar ────────────────────────────────────────────────────
    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=CLR["panel2"], height=26)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_propagate(False)

        # Accent line at top
        tk.Frame(bar, bg=CLR["border2"], height=1).place(
            relx=0, rely=0, relwidth=1.0)

        dot = tk.Canvas(bar, width=8, height=8, bg=CLR["panel2"],
                        highlightthickness=0)
        dot.pack(side="left", padx=(12, 4), pady=9)
        dot.create_oval(1, 1, 7, 7, fill=CLR["accent"], outline="")

        tk.Label(bar, textvariable=self._status_msg,
                 bg=CLR["panel2"], fg=CLR["text_sub"],
                 font=FONT_MICRO).pack(side="left", pady=4)

        tk.Label(bar, text="SBE3220 Medical Equipment (II) — Task 02",
                 bg=CLR["panel2"], fg=CLR["text_dim"],
                 font=FONT_MICRO).pack(side="right", padx=12, pady=4)

    # ══════════════════════════════════════════════════════════════════
    # Widget factory helpers
    # ══════════════════════════════════════════════════════════════════

    def _card(self, parent, title: str = "",
              layout: str = "grid") -> tk.Frame:
        """
        Create a styled card frame.
        layout="grid"  → caller places with .grid(...)
        layout="pack"  → caller places with .pack(...)
        Returns the inner content frame.
        """
        outer = tk.Frame(parent, bg=CLR["panel"],
                         highlightthickness=1,
                         highlightbackground=CLR["border2"])
        # Store outer so caller can place it
        outer._layout = layout

        if title:
            hdr = tk.Frame(outer, bg=CLR["panel2"])
            hdr.pack(fill="x")
            tk.Frame(hdr, bg=CLR["accent"], width=3).pack(side="left",
                                                           fill="y")
            tk.Label(hdr, text=title, bg=CLR["panel2"],
                     fg=CLR["text"], font=FONT_BODY,
                     padx=10, pady=6).pack(side="left")

        inner = tk.Frame(outer, bg=CLR["panel"], padx=10, pady=8)
        inner.pack(fill="x")
        # Attach outer to inner so callers can grid/pack the outer wrapper
        inner._outer = outer
        return inner

    def _btn(self, parent, text, command=None,
             bg=None, **kwargs) -> tk.Button:
        bg = bg or CLR["border2"]
        return tk.Button(parent, text=text, command=command,
                         bg=bg, fg=CLR["text"],
                         activebackground=CLR["accent"],
                         activeforeground="white",
                         font=FONT_MICRO, relief="flat",
                         cursor="hand2", padx=8, pady=3,
                         **kwargs)

    def _cta_btn(self, parent, text, command,
                 bg=CLR["border2"]) -> tk.Button:
        return tk.Button(parent, text=text, command=command,
                         bg=bg, fg="white",
                         activebackground=CLR["accent_dark"],
                         activeforeground="white",
                         font=("Segoe UI Semibold", 9),
                         relief="flat", cursor="hand2",
                         padx=12, pady=4)

    def _chk(self, parent, text, var, command=None) -> tk.Checkbutton:
        return tk.Checkbutton(parent, text=text, variable=var,
                              command=command,
                              bg=CLR["panel"], fg=CLR["text_sub"],
                              activebackground=CLR["panel"],
                              selectcolor=CLR["panel2"],
                              font=FONT_MICRO, cursor="hand2")

    def _field_label(self, parent, text: str) -> tk.Label:
        return tk.Label(parent, text=text, bg=CLR["panel"],
                        fg=CLR["text_dim"], font=FONT_LABEL)

    def _toggle_btn(self, parent, var: tk.BooleanVar,
                    command) -> tk.Checkbutton:
        return tk.Checkbutton(parent, text="ON",
                              variable=var, command=command,
                              bg=CLR["panel"], fg=CLR["accent"],
                              activebackground=CLR["panel"],
                              selectcolor=CLR["panel"],
                              font=("Segoe UI Semibold", 8),
                              cursor="hand2", relief="flat",
                              indicatoron=True)

    # ══════════════════════════════════════════════════════════════════
    # Key bindings
    # ══════════════════════════════════════════════════════════════════

    def _bind_keys(self):
        r = self.root
        for key, nav in [("Up","up"),("Down","down"),
                          ("Left","left"),("Right","right")]:
            r.bind(f"<KeyPress-{key}>",
                   lambda e, k=nav: self.navigation.key_press(k))
            r.bind(f"<KeyRelease-{key}>",
                   lambda e, k=nav: self.navigation.key_release(k))
        for key, nav in [("w","up"),("s","down"),
                          ("a","left"),("d","right"),
                          ("q","q"),("e","e")]:
            r.bind(f"<KeyPress-{key}>",
                   lambda e, k=nav: self.navigation.key_press(k))
            r.bind(f"<KeyRelease-{key}>",
                   lambda e, k=nav: self.navigation.key_release(k))

        r.bind("<space>",  lambda e: self.toggle_pause())
        r.bind("<Escape>", lambda e: self.stop())
        r.bind("<c>",      lambda e: self.capture_image())
        r.bind("<r>",      lambda e: self.toggle_recording())
        r.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════════════════
    # Control actions
    # ══════════════════════════════════════════════════════════════════

    def start(self):
        if self._running:
            return
        self._running = True
        self._paused  = False
        self._status("Endoscope system running.")
        self._loop()

    def stop(self):
        self._running = False
        self._paused  = False
        if self.output.recording:
            self.output.stop_recording()
            self._rec_badge.pack_forget()
        self._status("System stopped.")

    def toggle_pause(self):
        if not self._running:
            return
        self._paused = not self._paused
        self._status("Paused." if self._paused else "Resumed.")

    def capture_image(self):
        if not self._running:
            self._status("Start the system first.")
            return
        path = self.output.capture_image(self._current_frame)
        self._append_log(f"📷 {os.path.basename(path)}")
        self._status(f"Image saved → {os.path.basename(path)}")

    def toggle_recording(self):
        if not self._running:
            self._status("Start the system first.")
            return
        if self.output.recording:
            path = self.output.stop_recording()
            self._rec_badge.pack_forget()
            self._rec_btn.config(text="⏺  Record",
                                 bg=CLR["border2"])
            self._append_log(f"🎬 {os.path.basename(path)}")
            self._status(f"Video saved → {os.path.basename(path)}")
        else:
            self.output.start_recording(FRAME_W, FRAME_H)
            self._rec_badge.pack(side="right", padx=8, pady=6)
            self._rec_btn.config(text="⏹  Stop Rec",
                                 bg=CLR["danger"])
            self._status("Recording started.")

    def _reset_navigation(self):
        self.navigation.reset()
        self._depth_var.set(0)

    # ══════════════════════════════════════════════════════════════════
    # Main render loop
    # ══════════════════════════════════════════════════════════════════

    def _loop(self):
        if not self._running:
            return

        if not self._paused:
            self.navigation.update()

            frame = self.imaging.get_frame()
            self._current_frame = frame

            frame = self._apply_nav_transform(frame)
            frame = self.illumination.apply(frame)

            frame_hud = self.output.draw_hud(
                frame,
                self.navigation.status,
                self.illumination.intensity_percent,
                self.imaging.mode,
            )

            self.output.write_frame(frame)
            self._show_frame(frame_hud)
            self._update_ui()

        self.root.after(33, self._loop)

    def _apply_nav_transform(self, frame):
        if frame is None:
            return frame
        dx, dy, angle = self.navigation.get_frame_transform()
        H, W = frame.shape[:2]
        M_t = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(frame, M_t, (W, H),
                               borderMode=cv2.BORDER_REPLICATE)
        if abs(angle) > 0.5:
            M_r = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
            frame = cv2.warpAffine(frame, M_r, (W, H),
                                   borderMode=cv2.BORDER_REPLICATE)
        return frame

    def _show_frame(self, frame: np.ndarray):
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.config(width=frame.shape[1], height=frame.shape[0])
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)

        # FPS
        self._frame_count += 1
        now = time.time()
        if now - self._t_fps >= 1.0:
            self._fps = self._frame_count / (now - self._t_fps)
            self._frame_count = 0
            self._t_fps = now

    # ══════════════════════════════════════════════════════════════════
    # UI helpers
    # ══════════════════════════════════════════════════════════════════

    def _update_ui(self):
        status = self.navigation.status
        for key, var in self._nav_vars.items():
            if key in status:
                var.set(status[key])
        self._depth_var.set(self.navigation.insertion_depth)

        # Mode badge
        mode = self.imaging.mode
        if mode == "simulation":
            self._mode_badge.config(text="● SIMULATION",
                                    bg=CLR["tag_sim"])
        else:
            self._mode_badge.config(text="● CAMERA",
                                    bg=CLR["tag_cam"])

        # FPS label
        self._fps_lbl.config(text=f"{self._fps:.1f} fps")

        # Right sidebar metrics
        vals = {
            "Frame Rate": f"{self._fps:.1f}",
            "Light Power": f"{self.illumination.intensity_percent}",
            "Zoom Level":  f"{self.imaging.zoom_factor:.1f}",
            "Insertion":   f"{self.navigation.insertion_depth:.1f}",
        }
        for name, val in vals.items():
            if name in self._metric_vars:
                self._metric_vars[name][0].set(val)

        # Pulse dot animation (toggle colour)
        if self._frame_count % 2 == 0:
            self._pulse_canvas.itemconfig(
                self._pulse_dot,
                fill=CLR["success"] if self._running else CLR["text_dim"])

    def _append_log(self, msg: str):
        self._log_text.config(state="normal")
        self._log_text.insert("end", msg + "\n")
        self._log_text.see("end")
        self._log_text.config(state="disabled")

    def _status(self, msg: str):
        self._status_msg.set(msg)

    # ══════════════════════════════════════════════════════════════════
    # Control callbacks
    # ══════════════════════════════════════════════════════════════════

    def _on_intensity_change(self, val):
        v = float(val)
        self.illumination.set_intensity(v)
        self._intensity_lbl.config(text=f"{int(v * 50)}%")

    def _on_light_toggle(self):
        self.illumination.set_enabled(self._light_var.get())

    def _on_temp_change(self):
        self.illumination.set_color_temperature(self._temp_var.get())

    def _on_source_change(self):
        self.imaging.set_mode(self._src_var.get())

    def _on_zoom_change(self, val):
        v = float(val)
        self.imaging.set_zoom(v)
        self._zoom_lbl.config(text=f"{v:.1f}×")

    def _on_filter_change(self, key, var):
        self.imaging.toggle_filter(key, var.get())

    def _on_polyp_toggle(self):
        self.imaging.show_polyp = self._polyp_var.get()

    def _on_depth_slide(self, val):
        self.navigation.insertion_depth = float(val)

    def _on_close(self):
        self.stop()
        self.imaging.release()
        self.root.destroy()

    # ══════════════════════════════════════════════════════════════════
    # Run
    # ══════════════════════════════════════════════════════════════════

    def run(self):
        self._current_frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self.start()

        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w  = self.root.winfo_width()
        h  = self.root.winfo_height()
        x  = max(0, (sw - w) // 2)
        y  = max(0, (sh - h) // 2)
        self.root.geometry(f"+{x}+{y}")

        self.root.mainloop()