"""
Driver Fatigue Detection System - Frontend Main Application
============================================================
Technology Stack: Python + OpenCV + Tkinter
Architecture: Frontend-only shell, connects to backend via REST API
Author: Frontend Team
"""

import tkinter as tk
from tkinter import ttk, messagebox, font
import cv2
import threading
import time
import json
import base64
import numpy as np
from PIL import Image, ImageTk
import requests
from datetime import datetime
import winsound  # Windows sound; see utils/sound.py for cross-platform
import sys
import os

# ─── Import internal modules ───────────────────────────────────────────────────
from utils.api_client import APIClient
from utils.alert_manager import AlertManager
from utils.camera_handler import CameraHandler
from utils.ui_components import StatusBadge, AlertBanner, MetricCard


# ════════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (edit to match your backend team's API)
# ════════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "backend_url": "http://localhost:5000",   # ← backend URL
    "camera_index": 0,                         # 0 = default webcam
    "frame_rate": 15,                          # frames per second to process
    "alert_cooldown": 3,                       # seconds between repeated alerts
    "ear_threshold": 0.25,                     # Eye Aspect Ratio threshold
    "mar_threshold": 0.6,                      # Mouth Aspect Ratio threshold
    "consec_frames": 20,                       # consecutive frames for drowsy trigger
}


# ════════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION CLASS
# ════════════════════════════════════════════════════════════════════════════════
class FatigueDetectionApp:
    """
    Main application window.
    Manages three pages: Login  →  Identity Verification  →  Monitoring Dashboard
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Driver Fatigue Detection System")
        self.root.geometry("1280x800")
        self.root.minsize(1024, 680)
        self.root.configure(bg="#F8F9FA")

        # ── State variables ──────────────────────────────────────────────────
        self.current_page = None
        self.is_monitoring = False
        self.is_verified = False
        self.driver_name = tk.StringVar(value="")
        self.current_status = tk.StringVar(value="IDLE")
        self.ear_value = tk.DoubleVar(value=0.0)
        self.mar_value = tk.DoubleVar(value=0.0)
        self.alert_count = tk.IntVar(value=0)
        self.session_time = tk.StringVar(value="00:00:00")

        # ── Services ─────────────────────────────────────────────────────────
        self.api_client = APIClient(CONFIG["backend_url"])
        self.alert_manager = AlertManager()
        self.camera = CameraHandler(CONFIG["camera_index"])

        # ── Threading ─────────────────────────────────────────────────────────
        self._camera_thread = None
        self._session_thread = None
        self._session_start = None
        self._stop_event = threading.Event()

        # ── Build UI ─────────────────────────────────────────────────────────
        self._setup_fonts()
        self._setup_styles()
        self._build_layout()
        self._show_page("login")

        # ── Cleanup on close ─────────────────────────────────────────────────
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ────────────────────────────────────────────────────────────────────────────
    #  SETUP
    # ────────────────────────────────────────────────────────────────────────────
    def _setup_fonts(self):
        """Register custom fonts used throughout the UI."""
        self.font_title   = font.Font(family="Segoe UI", size=22, weight="bold")
        self.font_heading = font.Font(family="Segoe UI", size=14, weight="bold")
        self.font_body    = font.Font(family="Segoe UI", size=11)
        self.font_small   = font.Font(family="Segoe UI", size=9)
        self.font_metric  = font.Font(family="Segoe UI", size=28, weight="bold")
        self.font_label   = font.Font(family="Segoe UI", size=10)
        self.font_mono    = font.Font(family="Consolas", size=10)

    def _setup_styles(self):
        """Configure ttk widget styles for clean white theme."""
        style = ttk.Style()
        style.theme_use("clam")

        # ── Button styles ────────────────────────────────────────────────────
        style.configure("Primary.TButton",
            background="#1A1A2E", foreground="white",
            font=("Segoe UI", 11, "bold"), padding=(20, 10),
            borderwidth=0, relief="flat")
        style.map("Primary.TButton",
            background=[("active", "#16213E"), ("pressed", "#0F3460")])

        style.configure("Danger.TButton",
            background="#E63946", foreground="white",
            font=("Segoe UI", 11, "bold"), padding=(20, 10),
            borderwidth=0, relief="flat")
        style.map("Danger.TButton",
            background=[("active", "#C1121F"), ("pressed", "#9D0208")])

        style.configure("Success.TButton",
            background="#2DC653", foreground="white",
            font=("Segoe UI", 11, "bold"), padding=(20, 10),
            borderwidth=0, relief="flat")
        style.map("Success.TButton",
            background=[("active", "#25A244"), ("pressed", "#1A7A32")])

        style.configure("Ghost.TButton",
            background="#F8F9FA", foreground="#6C757D",
            font=("Segoe UI", 10), padding=(12, 8),
            borderwidth=1, relief="solid")
        style.map("Ghost.TButton",
            background=[("active", "#E9ECEF")])

        # ── Entry style ──────────────────────────────────────────────────────
        style.configure("Clean.TEntry",
            fieldbackground="white", borderwidth=1,
            relief="solid", padding=8)

        # ── Separator ────────────────────────────────────────────────────────
        style.configure("TSeparator", background="#E9ECEF")

    def _build_layout(self):
        """Build the top-level frame that holds all pages."""
        # Header bar
        self.header = tk.Frame(self.root, bg="#FFFFFF", height=64)
        self.header.pack(fill="x", side="top")
        self.header.pack_propagate(False)
        self._build_header()

        # Page container
        self.container = tk.Frame(self.root, bg="#F8F9FA")
        self.container.pack(fill="both", expand=True)

        # Status bar
        self.statusbar = tk.Frame(self.root, bg="#FFFFFF", height=28)
        self.statusbar.pack(fill="x", side="bottom")
        self.statusbar.pack_propagate(False)
        self._build_statusbar()

        # ── Pre-build all pages ──────────────────────────────────────────────
        self.pages = {}
        self.pages["login"]      = self._build_login_page()
        self.pages["verify"]     = self._build_verify_page()
        self.pages["monitor"]    = self._build_monitor_page()

    def _build_header(self):
        """Top navigation bar."""
        # Logo area
        logo_frame = tk.Frame(self.header, bg="#FFFFFF")
        logo_frame.pack(side="left", padx=24, pady=12)

        tk.Label(logo_frame, text="◉", font=("Segoe UI", 18),
                 bg="#FFFFFF", fg="#E63946").pack(side="left")
        tk.Label(logo_frame, text="  FatigueGuard",
                 font=("Segoe UI", 15, "bold"),
                 bg="#FFFFFF", fg="#1A1A2E").pack(side="left")

        # Nav status pills
        self.nav_frame = tk.Frame(self.header, bg="#FFFFFF")
        self.nav_frame.pack(side="right", padx=24, pady=16)

        self.nav_step1 = tk.Label(self.nav_frame, text="① Login",
            font=("Segoe UI", 9), bg="#1A1A2E", fg="white",
            padx=10, pady=4)
        self.nav_step1.pack(side="left", padx=3)

        self.nav_step2 = tk.Label(self.nav_frame, text="② Verify",
            font=("Segoe UI", 9), bg="#DEE2E6", fg="#6C757D",
            padx=10, pady=4)
        self.nav_step2.pack(side="left", padx=3)

        self.nav_step3 = tk.Label(self.nav_frame, text="③ Monitor",
            font=("Segoe UI", 9), bg="#DEE2E6", fg="#6C757D",
            padx=10, pady=4)
        self.nav_step3.pack(side="left", padx=3)

        # Divider
        tk.Frame(self.header, bg="#E9ECEF", width=1).pack(
            side="bottom", fill="x")

    def _build_statusbar(self):
        """Bottom status bar showing connection and time."""
        self.conn_label = tk.Label(self.statusbar,
            text="⬤  Backend: Disconnected",
            font=("Segoe UI", 8), bg="#FFFFFF", fg="#ADB5BD")
        self.conn_label.pack(side="left", padx=16, pady=6)

        self.time_label = tk.Label(self.statusbar, text="",
            font=("Segoe UI", 8), bg="#FFFFFF", fg="#ADB5BD")
        self.time_label.pack(side="right", padx=16, pady=6)
        self._tick_time()

    def _tick_time(self):
        self.time_label.config(
            text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.root.after(1000, self._tick_time)

    # ────────────────────────────────────────────────────────────────────────────
    #  PAGE: LOGIN
    # ────────────────────────────────────────────────────────────────────────────
    def _build_login_page(self) -> tk.Frame:
        """Driver name entry page."""
        page = tk.Frame(self.container, bg="#F8F9FA")

        # Center card
        card = tk.Frame(page, bg="#FFFFFF", padx=48, pady=40)
        card.place(relx=0.5, rely=0.5, anchor="center")

        # Icon
        tk.Label(card, text="🚗", font=("Segoe UI", 48),
                 bg="#FFFFFF").pack(pady=(0, 8))

        tk.Label(card, text="Driver Fatigue Detection System",
                 font=self.font_title, bg="#FFFFFF",
                 fg="#1A1A2E").pack()
        tk.Label(card, text="Real-time fatigue monitoring for road safety",
                 font=self.font_body, bg="#FFFFFF",
                 fg="#6C757D").pack(pady=(4, 32))

        # Input row
        input_frame = tk.Frame(card, bg="#FFFFFF")
        input_frame.pack(fill="x")

        tk.Label(input_frame, text="Driver Name",
                 font=self.font_label, bg="#FFFFFF",
                 fg="#495057").pack(anchor="w")

        entry_frame = tk.Frame(input_frame, bg="#E9ECEF", padx=1, pady=1)
        entry_frame.pack(fill="x", pady=(4, 0))

        self.name_entry = tk.Entry(entry_frame,
            textvariable=self.driver_name,
            font=self.font_body, bg="#FFFFFF",
            fg="#1A1A2E", relief="flat",
            insertbackground="#1A1A2E")
        self.name_entry.pack(fill="x", padx=8, pady=8)
        self.name_entry.bind("<Return>", lambda e: self._on_login())

        # Features grid
        feat_frame = tk.Frame(card, bg="#FFFFFF")
        feat_frame.pack(pady=28)

        features = [
            ("👁", "EAR Eye Tracking"),
            ("👄", "MAR Yawn Detection"),
            ("🧠", "Head Pose Analysis"),
            ("🔒", "Face Verification"),
        ]
        for i, (icon, label) in enumerate(features):
            col = tk.Frame(feat_frame, bg="#F8F9FA", padx=14, pady=12)
            col.grid(row=0, column=i, padx=6)
            tk.Label(col, text=icon, font=("Segoe UI", 18),
                     bg="#F8F9FA").pack()
            tk.Label(col, text=label, font=self.font_small,
                     bg="#F8F9FA", fg="#495057").pack(pady=(4, 0))

        # Start button
        ttk.Button(card, text="Start  →",
                   style="Primary.TButton",
                   command=self._on_login).pack(fill="x", pady=(8, 0))

        return page

    # ────────────────────────────────────────────────────────────────────────────
    #  PAGE: IDENTITY VERIFICATION
    # ────────────────────────────────────────────────────────────────────────────
    def _build_verify_page(self) -> tk.Frame:
        """Face verification page with live camera preview."""
        page = tk.Frame(self.container, bg="#F8F9FA")

        # Left panel: camera
        left = tk.Frame(page, bg="#FFFFFF", width=520)
        left.pack(side="left", fill="y", padx=(32, 16), pady=32)
        left.pack_propagate(False)

        tk.Label(left, text="Identity Verification",
                 font=self.font_heading, bg="#FFFFFF",
                 fg="#1A1A2E").pack(pady=(24, 4))
        tk.Label(left, text="Look directly at the camera to verify your identity",
                 font=self.font_small, bg="#FFFFFF",
                 fg="#6C757D").pack(pady=(0, 16))

        # Camera preview box
        cam_border = tk.Frame(left, bg="#E9ECEF", padx=2, pady=2)
        cam_border.pack(padx=24)

        self.verify_canvas = tk.Canvas(cam_border, width=460, height=345,
                                       bg="#1A1A2E", highlightthickness=0)
        self.verify_canvas.pack()

        # Overlay text on canvas
        self.verify_canvas.create_text(230, 172,
            text="Camera initializing…",
            fill="#6C757D", font=("Segoe UI", 12),
            tags="placeholder")

        # Scanning bar animation
        self.scan_bar = tk.Frame(left, bg="#E9ECEF", height=3)
        self.scan_bar.pack(fill="x", padx=24, pady=(0, 8))
        self.scan_anim = tk.Frame(self.scan_bar, bg="#1A1A2E",
                                  height=3, width=0)
        self.scan_anim.place(x=0, y=0)

        # Status label
        self.verify_status = tk.Label(left, text="Waiting to scan…",
            font=self.font_body, bg="#FFFFFF", fg="#6C757D")
        self.verify_status.pack(pady=8)

        # Right panel: instructions + controls
        right = tk.Frame(page, bg="#F8F9FA")
        right.pack(side="right", fill="both", expand=True,
                   padx=(0, 32), pady=32)

        # Instructions card
        instr = tk.Frame(right, bg="#FFFFFF", padx=24, pady=24)
        instr.pack(fill="x")

        tk.Label(instr, text="Instructions",
                 font=self.font_heading, bg="#FFFFFF",
                 fg="#1A1A2E").pack(anchor="w")

        steps = [
            ("1", "Position your face in the camera frame"),
            ("2", "Ensure good lighting on your face"),
            ("3", "Remove sunglasses or face coverings"),
            ("4", "Hold still while verification runs"),
        ]
        for num, text in steps:
            row = tk.Frame(instr, bg="#FFFFFF")
            row.pack(anchor="w", pady=6, fill="x")
            tk.Label(row, text=num,
                     font=("Segoe UI", 9, "bold"),
                     bg="#1A1A2E", fg="white",
                     width=2, pady=2).pack(side="left")
            tk.Label(row, text=f"  {text}",
                     font=self.font_body, bg="#FFFFFF",
                     fg="#495057").pack(side="left")

        # Obstruction warning card
        warn = tk.Frame(right, bg="#FFF3CD", padx=20, pady=16)
        warn.pack(fill="x", pady=12)
        tk.Label(warn,
            text="⚠  Anti-Theft Notice",
            font=("Segoe UI", 10, "bold"),
            bg="#FFF3CD", fg="#856404").pack(anchor="w")
        tk.Label(warn,
            text="Covering the camera or using an unauthorized\n"
                 "face will trigger the theft-prevention alarm.",
            font=self.font_small, bg="#FFF3CD",
            fg="#856404", justify="left").pack(anchor="w", pady=(4, 0))

        # Result card
        self.verify_result_card = tk.Frame(right, bg="#FFFFFF",
                                           padx=20, pady=16)
        self.verify_result_card.pack(fill="x")

        self.verify_result_icon = tk.Label(self.verify_result_card,
            text="○", font=("Segoe UI", 36),
            bg="#FFFFFF", fg="#DEE2E6")
        self.verify_result_icon.pack()

        self.verify_result_label = tk.Label(self.verify_result_card,
            text="Awaiting verification",
            font=self.font_body, bg="#FFFFFF", fg="#ADB5BD")
        self.verify_result_label.pack(pady=(4, 0))

        # Buttons
        btn_frame = tk.Frame(right, bg="#F8F9FA")
        btn_frame.pack(fill="x", pady=12)

        self.verify_btn = ttk.Button(btn_frame,
            text="▶  Begin Verification",
            style="Primary.TButton",
            command=self._on_start_verify)
        self.verify_btn.pack(fill="x", pady=4)

        self.verify_retry_btn = ttk.Button(btn_frame,
            text="↺  Retry",
            style="Ghost.TButton",
            command=self._on_retry_verify,
            state="disabled")
        self.verify_retry_btn.pack(fill="x", pady=4)

        return page

    # ────────────────────────────────────────────────────────────────────────────
    #  PAGE: MONITORING DASHBOARD
    # ────────────────────────────────────────────────────────────────────────────
    def _build_monitor_page(self) -> tk.Frame:
        """Main monitoring dashboard with live video + metrics."""
        page = tk.Frame(self.container, bg="#F8F9FA")

        # ── Top alert banner (hidden by default) ────────────────────────────
        self.alert_banner = tk.Frame(page, bg="#E63946", height=0)
        self.alert_banner.pack(fill="x")
        self.alert_banner.pack_propagate(False)
        self.alert_label = tk.Label(self.alert_banner,
            text="", font=("Segoe UI", 13, "bold"),
            bg="#E63946", fg="white")
        self.alert_label.pack(pady=10)

        # ── Main content ─────────────────────────────────────────────────────
        content = tk.Frame(page, bg="#F8F9FA")
        content.pack(fill="both", expand=True, padx=24, pady=16)

        # Left: video feed
        left = tk.Frame(content, bg="#FFFFFF")
        left.pack(side="left", fill="y", padx=(0, 16))

        # Driver info bar
        info_bar = tk.Frame(left, bg="#1A1A2E", padx=16, pady=10)
        info_bar.pack(fill="x")
        tk.Label(info_bar, text="◉  LIVE",
                 font=("Segoe UI", 9, "bold"),
                 bg="#1A1A2E", fg="#E63946").pack(side="left")
        self.driver_label = tk.Label(info_bar,
            text="Driver: —",
            font=("Segoe UI", 10, "bold"),
            bg="#1A1A2E", fg="white")
        self.driver_label.pack(side="left", padx=16)
        self.session_label = tk.Label(info_bar,
            textvariable=self.session_time,
            font=("Consolas", 10),
            bg="#1A1A2E", fg="#ADB5BD")
        self.session_label.pack(side="right")

        # Video canvas
        self.monitor_canvas = tk.Canvas(left, width=640, height=480,
                                        bg="#0D0D0D", highlightthickness=0)
        self.monitor_canvas.pack()

        # Control buttons
        ctrl = tk.Frame(left, bg="#FFFFFF", pady=12)
        ctrl.pack(fill="x")
        ctrl.columnconfigure((0, 1, 2), weight=1)

        self.start_btn = ttk.Button(ctrl,
            text="▶  Start Monitor",
            style="Success.TButton",
            command=self._on_toggle_monitor)
        self.start_btn.grid(row=0, column=0, padx=6, sticky="ew")

        ttk.Button(ctrl, text="📷  Snapshot",
                   style="Ghost.TButton",
                   command=self._on_snapshot).grid(
            row=0, column=1, padx=6, sticky="ew")

        ttk.Button(ctrl, text="✕  End Session",
                   style="Danger.TButton",
                   command=self._on_end_session).grid(
            row=0, column=2, padx=6, sticky="ew")

        # Right panel: metrics + log
        right = tk.Frame(content, bg="#F8F9FA")
        right.pack(side="right", fill="both", expand=True)

        # Status card
        status_card = tk.Frame(right, bg="#FFFFFF", padx=20, pady=16)
        status_card.pack(fill="x", pady=(0, 12))

        tk.Label(status_card, text="Driver Status",
                 font=self.font_heading, bg="#FFFFFF",
                 fg="#1A1A2E").pack(anchor="w")

        self.status_display = tk.Label(status_card, text="NORMAL",
            font=("Segoe UI", 32, "bold"),
            bg="#FFFFFF", fg="#2DC653")
        self.status_display.pack(pady=(8, 4))

        self.status_sub = tk.Label(status_card,
            text="No anomalies detected",
            font=self.font_body, bg="#FFFFFF", fg="#6C757D")
        self.status_sub.pack()

        # Metrics row
        metrics_frame = tk.Frame(right, bg="#F8F9FA")
        metrics_frame.pack(fill="x", pady=(0, 12))
        metrics_frame.columnconfigure((0, 1, 2), weight=1)

        # EAR metric
        self.ear_card = self._make_metric_card(
            metrics_frame, "EAR", "Eye Aspect Ratio",
            "—", "#E9ECEF", 0)
        self.ear_card.grid(row=0, column=0, padx=(0, 6), sticky="nsew")

        # MAR metric
        self.mar_card = self._make_metric_card(
            metrics_frame, "MAR", "Mouth Aspect Ratio",
            "—", "#E9ECEF", 1)
        self.mar_card.grid(row=0, column=1, padx=3, sticky="nsew")

        # Alert count metric
        self.alert_card = self._make_metric_card(
            metrics_frame, "ALERTS", "Session Total",
            "0", "#E9ECEF", 2)
        self.alert_card.grid(row=0, column=2, padx=(6, 0), sticky="nsew")

        # Threshold guide card
        guide = tk.Frame(right, bg="#FFFFFF", padx=20, pady=16)
        guide.pack(fill="x", pady=(0, 12))

        tk.Label(guide, text="Thresholds",
                 font=self.font_heading, bg="#FFFFFF",
                 fg="#1A1A2E").pack(anchor="w", pady=(0, 8))

        thresholds = [
            ("EAR < 0.25", "Eye closure / drowsiness",  "#E63946"),
            ("MAR > 0.60", "Yawning detected",           "#FD7E14"),
            ("Head tilt",  "Head pose anomaly",           "#6F42C1"),
        ]
        for val, desc, color in thresholds:
            row = tk.Frame(guide, bg="#FFFFFF")
            row.pack(fill="x", pady=3)
            tk.Frame(row, bg=color, width=4).pack(side="left", fill="y")
            tk.Label(row, text=f"  {val}",
                     font=self.font_mono, bg="#FFFFFF",
                     fg="#1A1A2E", width=14,
                     anchor="w").pack(side="left")
            tk.Label(row, text=desc,
                     font=self.font_small, bg="#FFFFFF",
                     fg="#6C757D").pack(side="left")

        # Event log
        log_frame = tk.Frame(right, bg="#FFFFFF", padx=16, pady=12)
        log_frame.pack(fill="both", expand=True)

        log_header = tk.Frame(log_frame, bg="#FFFFFF")
        log_header.pack(fill="x", pady=(0, 8))
        tk.Label(log_header, text="Event Log",
                 font=self.font_heading, bg="#FFFFFF",
                 fg="#1A1A2E").pack(side="left")
        ttk.Button(log_header, text="Clear",
                   style="Ghost.TButton",
                   command=self._clear_log).pack(side="right")

        log_scroll = tk.Frame(log_frame, bg="#FFFFFF")
        log_scroll.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_scroll,
            font=self.font_mono, bg="#F8F9FA",
            fg="#495057", relief="flat",
            state="disabled", wrap="word",
            height=8, padx=8, pady=8)

        scrollbar = ttk.Scrollbar(log_scroll,
            command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Color tags for log
        self.log_text.tag_configure("INFO",    foreground="#6C757D")
        self.log_text.tag_configure("WARNING", foreground="#FD7E14")
        self.log_text.tag_configure("ALERT",   foreground="#E63946",
                                               font=self.font_mono)
        self.log_text.tag_configure("OK",      foreground="#2DC653")

        return page

    def _make_metric_card(self, parent, title, subtitle, value, color, idx):
        """Create a single metric display card."""
        card = tk.Frame(parent, bg="#FFFFFF", padx=16, pady=12)
        tk.Label(card, text=title,
                 font=("Segoe UI", 9, "bold"),
                 bg="#FFFFFF", fg="#ADB5BD").pack(anchor="w")
        val_lbl = tk.Label(card, text=value,
                           font=("Segoe UI", 26, "bold"),
                           bg="#FFFFFF", fg="#1A1A2E")
        val_lbl.pack(anchor="w", pady=(4, 2))
        tk.Label(card, text=subtitle,
                 font=self.font_small,
                 bg="#FFFFFF", fg="#6C757D").pack(anchor="w")
        # Store reference for updates
        setattr(self, f"_metric_val_{idx}", val_lbl)
        return card

    # ────────────────────────────────────────────────────────────────────────────
    #  PAGE NAVIGATION
    # ────────────────────────────────────────────────────────────────────────────
    def _show_page(self, name: str):
        """Switch visible page and update header nav."""
        if self.current_page:
            self.pages[self.current_page].pack_forget()
        self.pages[name].pack(fill="both", expand=True)
        self.current_page = name

        nav_map = {
            "login":   (0, "#1A1A2E", "white"),
            "verify":  (1, "#1A1A2E", "white"),
            "monitor": (2, "#1A1A2E", "white"),
        }
        steps = [self.nav_step1, self.nav_step2, self.nav_step3]
        for i, step in enumerate(steps):
            step.configure(bg="#DEE2E6", fg="#6C757D")
        idx = list(nav_map.keys()).index(name)
        steps[idx].configure(bg="#1A1A2E", fg="white")

    # ────────────────────────────────────────────────────────────────────────────
    #  EVENT HANDLERS
    # ────────────────────────────────────────────────────────────────────────────
    def _on_login(self):
        name = self.driver_name.get().strip()
        if not name:
            self.name_entry.focus()
            self._flash_entry()
            return
        self._log_event(f"Session started for: {name}", "INFO")
        self._show_page("verify")
        self.camera.start()
        self._start_verify_feed()

    def _flash_entry(self):
        """Briefly highlight entry if empty."""
        orig = self.name_entry.cget("bg")
        self.name_entry.configure(bg="#FFE0E0")
        self.root.after(400, lambda: self.name_entry.configure(bg=orig))

    def _on_start_verify(self):
        """Send frame to backend for face verification."""
        frame = self.camera.get_frame()
        if frame is None:
            messagebox.showwarning("Camera", "Camera not ready. Please wait.")
            return
        self.verify_btn.configure(state="disabled")
        self.verify_status.configure(text="Verifying…", fg="#FD7E14")
        self._animate_scan(0)
        threading.Thread(target=self._do_verify,
                         args=(frame,), daemon=True).start()

    def _do_verify(self, frame: np.ndarray):
        """Background thread: send frame to /api/verify."""
        result = self.api_client.verify_identity(frame,
                    self.driver_name.get())
        self.root.after(0, self._on_verify_result, result)

    def _on_verify_result(self, result: dict):
        """Handle verification response from backend."""
        self.verify_btn.configure(state="normal")
        self.verify_retry_btn.configure(state="normal")

        if result.get("status") == "authorized":
            self.is_verified = True
            self.verify_result_icon.configure(text="✓", fg="#2DC653")
            self.verify_result_label.configure(
                text=f"Authorized: {result.get('name', self.driver_name.get())}",
                fg="#2DC653")
            self.verify_status.configure(text="Identity verified ✓", fg="#2DC653")
            self._update_conn_status(True)
            self.root.after(1500, self._goto_monitor)

        elif result.get("status") == "obstructed":
            self.verify_result_icon.configure(text="⚠", fg="#FD7E14")
            self.verify_result_label.configure(
                text="Camera obstructed – anti-theft triggered!", fg="#E63946")
            self.verify_status.configure(text="Obstruction detected!", fg="#E63946")
            self.alert_manager.trigger("OBSTRUCTION")

        elif result.get("status") == "unauthorized":
            self.verify_result_icon.configure(text="✗", fg="#E63946")
            self.verify_result_label.configure(
                text="Unauthorized face detected!", fg="#E63946")
            self.verify_status.configure(text="Access denied", fg="#E63946")
            self.alert_manager.trigger("UNAUTHORIZED")

        else:  # backend offline – demo mode
            self.verify_result_icon.configure(text="✓", fg="#2DC653")
            self.verify_result_label.configure(
                text="Demo mode (backend offline)", fg="#6C757D")
            self.verify_status.configure(
                text="Running in demo mode", fg="#6C757D")
            self.root.after(1500, self._goto_monitor)

    def _on_retry_verify(self):
        self.verify_result_icon.configure(text="○", fg="#DEE2E6")
        self.verify_result_label.configure(
            text="Awaiting verification", fg="#ADB5BD")
        self.verify_status.configure(text="Waiting to scan…", fg="#6C757D")
        self.verify_retry_btn.configure(state="disabled")

    def _goto_monitor(self):
        """Transition to monitoring dashboard."""
        self.camera.stop()
        self._show_page("monitor")
        self.driver_label.configure(
            text=f"Driver: {self.driver_name.get()}")
        self._log_event("Monitoring session initialized", "OK")
        self._log_event(
            f"Thresholds — EAR: {CONFIG['ear_threshold']}  "
            f"MAR: {CONFIG['mar_threshold']}", "INFO")

    def _on_toggle_monitor(self):
        if self.is_monitoring:
            self._stop_monitoring()
        else:
            self._start_monitoring()

    def _start_monitoring(self):
        self.is_monitoring = True
        self._stop_event.clear()
        self.start_btn.configure(text="⏸  Pause Monitor",
                                 style="Danger.TButton")
        self.camera.start()
        self._session_start = time.time()
        self._update_session_timer()
        self._camera_thread = threading.Thread(
            target=self._camera_loop, daemon=True)
        self._camera_thread.start()
        self._log_event("Monitoring started", "OK")

    def _stop_monitoring(self):
        self.is_monitoring = False
        self._stop_event.set()
        self.camera.stop()
        self.start_btn.configure(text="▶  Start Monitor",
                                 style="Success.TButton")
        self._log_event("Monitoring paused", "INFO")

    def _on_end_session(self):
        if messagebox.askyesno("End Session",
            "End this monitoring session and return to login?"):
            self._stop_monitoring()
            self.driver_name.set("")
            self.alert_count.set(0)
            self.session_time.set("00:00:00")
            self._update_status_display("IDLE", "#6C757D", "Session ended")
            self._update_metrics("—", "—", "0")
            self._clear_log()
            self._show_page("login")

    def _on_snapshot(self):
        """Save current frame as PNG."""
        frame = self.camera.get_frame()
        if frame is None:
            messagebox.showinfo("Snapshot", "No frame available.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"snapshot_{ts}.png"
        cv2.imwrite(fname, frame)
        self._log_event(f"Snapshot saved: {fname}", "INFO")
        messagebox.showinfo("Snapshot", f"Saved as {fname}")

    # ────────────────────────────────────────────────────────────────────────────
    #  CAMERA LOOP (background thread)
    # ────────────────────────────────────────────────────────────────────────────
    def _camera_loop(self):
        """
        Main loop:
        1. Grab frame from camera
        2. Send to backend /api/analyze
        3. Update UI with returned metrics
        """
        interval = 1.0 / CONFIG["frame_rate"]
        while not self._stop_event.is_set():
            t0 = time.time()
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            # Display frame on canvas (safely on main thread)
            self.root.after(0, self._display_frame, frame, self.monitor_canvas)

            # Send to backend for analysis
            result = self.api_client.analyze_frame(frame)
            self.root.after(0, self._process_analysis_result, result, frame)

            # Throttle
            elapsed = time.time() - t0
            sleep_t = max(0, interval - elapsed)
            time.sleep(sleep_t)

    def _process_analysis_result(self, result: dict, frame: np.ndarray):
        """Update metrics and trigger alerts based on backend response."""
        if not result:
            return

        ear = result.get("ear", 0.0)
        mar = result.get("mar", 0.0)
        status = result.get("status", "NORMAL")
        head_pose = result.get("head_pose", "normal")

        # Update metric cards
        ear_str = f"{ear:.3f}"
        mar_str = f"{mar:.3f}"
        self._metric_val_0.configure(
            text=ear_str,
            fg="#E63946" if ear < CONFIG["ear_threshold"] else "#1A1A2E")
        self._metric_val_1.configure(
            text=mar_str,
            fg="#FD7E14" if mar > CONFIG["mar_threshold"] else "#1A1A2E")
        self._metric_val_2.configure(
            text=str(self.alert_count.get()))

        # Determine overall state
        if status == "DROWSY" or ear < CONFIG["ear_threshold"]:
            self._trigger_alert("DROWSY", "⚠  DROWSINESS DETECTED — Eyes closing!")
        elif status == "YAWNING" or mar > CONFIG["mar_threshold"]:
            self._trigger_alert("YAWNING", "⚠  YAWNING DETECTED — Fatigue warning!")
        elif head_pose == "anomaly":
            self._trigger_alert("HEAD", "⚠  HEAD POSE ANOMALY — Attention loss!")
        else:
            self._clear_alert()
            self._update_status_display("NORMAL", "#2DC653", "No anomalies detected")

    def _trigger_alert(self, alert_type: str, message: str):
        """Show alert banner, update status, play sound."""
        count = self.alert_count.get() + 1
        self.alert_count.set(count)
        self._metric_val_2.configure(text=str(count))
        self._show_alert_banner(message)
        status_map = {
            "DROWSY":  ("DROWSY",  "#E63946", "Eye closure detected"),
            "YAWNING": ("YAWNING", "#FD7E14", "Yawning detected"),
            "HEAD":    ("FATIGUE", "#6F42C1", "Head pose anomaly"),
        }
        s, c, sub = status_map.get(alert_type, ("ALERT", "#E63946", ""))
        self._update_status_display(s, c, sub)
        self.alert_manager.trigger(alert_type)
        self._log_event(message, "ALERT")

    def _clear_alert(self):
        self._hide_alert_banner()

    def _show_alert_banner(self, message: str):
        self.alert_label.configure(text=message)
        self.alert_banner.configure(height=46)

    def _hide_alert_banner(self):
        self.alert_banner.configure(height=0)

    def _update_status_display(self, status: str, color: str, sub: str):
        self.status_display.configure(text=status, fg=color)
        self.status_sub.configure(text=sub)

    def _update_metrics(self, ear, mar, alerts):
        self._metric_val_0.configure(text=ear)
        self._metric_val_1.configure(text=mar)
        self._metric_val_2.configure(text=alerts)

    # ────────────────────────────────────────────────────────────────────────────
    #  VERIFY PAGE CAMERA FEED
    # ────────────────────────────────────────────────────────────────────────────
    def _start_verify_feed(self):
        """Continuously update the verify page camera preview."""
        def loop():
            while self.current_page == "verify":
                frame = self.camera.get_frame()
                if frame is not None:
                    self.root.after(0, self._update_verify_ui, frame)
                time.sleep(1 / 15)
        threading.Thread(target=loop, daemon=True).start()

    def _update_verify_ui(self, frame):
        if self.current_page == "verify":
            self._display_frame(frame, self.verify_canvas)
            self.verify_canvas.delete("placeholder")

    # ────────────────────────────────────────────────────────────────────────────
    #  UTILS
    # ────────────────────────────────────────────────────────────────────────────
    def _display_frame(self, frame: np.ndarray, canvas: tk.Canvas):
        """Convert OpenCV BGR frame to Tkinter PhotoImage and display."""
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 10 or h < 10:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil = pil.resize((w, h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas._photo_ref = photo  # prevent GC

    def _animate_scan(self, x: int):
        """Animate the scan progress bar on verify page."""
        if self.current_page != "verify":
            return
        w = self.scan_bar.winfo_width()
        if w < 10:
            w = 460
        x = (x + 6) % (w + 60)
        self.scan_anim.place(x=max(0, x - 60), y=0,
                             width=min(60, x))
        if not self.is_verified:
            self.root.after(30, self._animate_scan, x)

    def _update_session_timer(self):
        """Periodically update the session clock display on the main thread."""
        if not self.is_monitoring:
            return
        elapsed = int(time.time() - self._session_start)
        h, r = divmod(elapsed, 3600)
        m, s = divmod(r, 60)
        self.session_time.set(f"{h:02d}:{m:02d}:{s:02d}")
        self.root.after(1000, self._update_session_timer)

    def _log_event(self, message: str, level: str = "INFO"):
        """Append timestamped event to the log panel."""
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}]  {message}\n"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line, level)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _update_conn_status(self, connected: bool):
        if connected:
            self.conn_label.configure(
                text="⬤  Backend: Connected",
                fg="#2DC653")
        else:
            self.conn_label.configure(
                text="⬤  Backend: Disconnected (Demo Mode)",
                fg="#FD7E14")

    def _on_close(self):
        self._stop_event.set()
        self.camera.stop()
        self.root.destroy()


# ════════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = FatigueDetectionApp(root)
    root.mainloop()
