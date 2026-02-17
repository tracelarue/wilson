#!/usr/bin/env python3
"""
MLX GUI Node - Live visualization of MLX90393 magnetometer data.

TO EDIT FORCE CALCULATION FORMULAS:
    Find the methods `calculate_grip_force()` and `calculate_downforce()`
    in the MLXLiveGUI class (around line 30-60). Edit the formulas there
    to change how grip force and downforce are calculated from compensated
    X, Y, Z values:
    (mlx_force - tare_force) - (mlx_ambient - tare_ambient)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import MagneticField

import tkinter as tk
from tkinter import ttk
import time
import queue
import threading
from collections import deque
import csv
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class MLXLiveGUI:
    """Tkinter GUI for live visualization of compensated MLX90393 magnetometer data."""

    # ============================================================================
    # FORCE CALCULATION FORMULAS - EDIT THESE TO CHANGE CALCULATIONS
    # ============================================================================

    @staticmethod
    def calculate_grip_force(x, y, z):
        """
        Calculate grip force from magnetic field values.

        Args:
            x, y, z: Magnetic field values in µT (after tare and ambient compensation)

        Returns:
            Grip force in Newtons

        EDIT THIS FORMULA as needed for your calibration.
        """
        # Calibrated polynomial formula for grip force
        grip_force = (
            (-0.233304) * x +
            (0.128798) * y +
            (1.75378) * z +
            (0.000175423) * x**2 +
            (8.41809e-08) * x * y +
            (0.00013915) * x * z +
            (0.000156242) * y**2 +
            (2.84164e-05) * y * z +
            (0.000266599) * z**2
        ) / 40
        return grip_force

    @staticmethod
    def calculate_downforce(x, y, z):
        """
        Calculate downward force from magnetic field values.

        Args:
            x, y, z: Magnetic field values in µT (after tare and ambient compensation)

        Returns:
            Downward force in Newtons

        EDIT THIS FORMULA as needed for your calibration.
        """
        # Simple linear formula for downforce
        downforce = x / 40
        return downforce

    # ============================================================================

    def __init__(self, root, ros_node, data_queue, history_seconds=5.0):
        self.plot_frame = None
        self.root = root
        self.ros_node = ros_node
        self.data_queue = data_queue
        self.root.title("MLX Magnetometer – Live View")

        # Basic window sizing
        self.root.geometry("1100x650")
        self.history_seconds = history_seconds

        # Data buffers (compensated delta used for plotting and force formulas)
        self.times = deque()
        self.X_raw = deque()
        self.Y_raw = deque()
        self.Z_raw = deque()

        # Raw data buffers (force + ambient)
        self.Xf_raw = deque()
        self.Yf_raw = deque()
        self.Zf_raw = deque()
        self.Xa_raw = deque()
        self.Ya_raw = deque()
        self.Za_raw = deque()

        # Tared data buffers (force + ambient)
        self.Xf_tared = deque()
        self.Yf_tared = deque()
        self.Zf_tared = deque()
        self.Xa_tared = deque()
        self.Ya_tared = deque()
        self.Za_tared = deque()

        # Tare offsets (captured simultaneously for force + ambient)
        self.tare_fx = 0.0
        self.tare_fy = 0.0
        self.tare_fz = 0.0
        self.tare_ax = 0.0
        self.tare_ay = 0.0
        self.tare_az = 0.0

        self.min_y_raw = -10
        self.max_y_raw = 10

        # Simple Moving Average (SMA) buffers
        self.sma_window_samples = 5
        self.X_sma = deque()
        self.Y_sma = deque()
        self.Z_sma = deque()

        # Force calculation buffers
        self.grip_force = deque()
        self.downforce = deque()
        self.grip_force_sma = deque()
        self.downforce_sma = deque()

        self.min_y_force = -1
        self.max_y_force = 1

        # Capture state
        self.capture_active = False
        self.capture_buffer = []
        self.top_plot_mode = "compensated"
        self.top_plot_mode_var = tk.StringVar(value="Plot: Comp")

        # Styling
        self._init_style()

        # Layout UI
        self._build_layout()

        # Schedule periodic UI updates
        self._schedule_update()

    def _init_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # Base colors
        bg = "#f4f5f7"
        accent = "#0b7285"
        accent_light = "#15aabf"

        # Set root background
        self.root.configure(bg=bg)

        # Generic frame background
        style.configure("Main.TFrame", background=bg)
        style.configure("Card.TFrame", background="white", relief="groove", borderwidth=1)

        # Labels
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"),
                        background=bg, foreground=accent)
        style.configure("Section.TLabel", font=("Segoe UI", 12, "bold"),
                        background=bg, foreground=accent)
        style.configure("Status.TLabel", font=("Segoe UI", 10),
                        background=bg, foreground="#333333")
        style.configure("Value.TLabel", font=("Consolas", 16, "bold"),
                        background=bg, foreground="#111111")

        # Inline labels
        style.configure("Inline.TLabel", font=("Segoe UI", 10),
                        background=bg, foreground="#444444")

        # Buttons
        style.configure(
            "Accent.TButton",
            font=("Segoe UI", 10, "bold"),
            foreground="white",
            background=accent,
            padding=(8, 3),
        )
        style.map(
            "Accent.TButton",
            background=[("active", accent_light), ("disabled", "#9fb6bd")],
        )

        style.configure(
            "Secondary.TButton",
            font=("Segoe UI", 10),
            foreground="#222222",
            background="white",
            padding=(8, 3),
            borderwidth=1,
            relief="solid",
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#e9ecef"), ("disabled", "#f1f3f5")],
            foreground=[("disabled", "#999999")],
        )

    def _build_layout(self):
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)
        self.root.columnconfigure(0, weight=1)

        # ---- Top: Title and status ----
        top = ttk.Frame(self.root, padding=(10, 10, 10, 5), style="Main.TFrame")
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=0)

        title_lbl = ttk.Label(top, text="MLX90393 Magnetometer Live Viewer", style="Title.TLabel")
        title_lbl.grid(row=0, column=0, sticky="w")

        subtitle = ttk.Label(
            top,
            text="ROS2 /mlx + /mlx_ambient • compensated delta",
            style="Status.TLabel"
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.status_var = tk.StringVar(value="Status: Streaming")
        self.status_lbl = ttk.Label(top, textvariable=self.status_var, style="Status.TLabel")
        self.status_lbl.grid(row=1, column=1, sticky="e", padx=(10, 0))

        # ---- Middle: Plot ----
        mid = ttk.Frame(self.root, padding=(10, 5, 10, 5), style="Main.TFrame")
        mid.grid(row=1, column=0, sticky="nsew")
        mid.rowconfigure(0, weight=1)
        mid.columnconfigure(0, weight=1)

        self.plot_parent = mid
        self._build_plot_panel(mid)

        # ---- Bottom: controls ----
        bottom = ttk.Frame(self.root, padding=(10, 5, 10, 10), style="Main.TFrame")
        bottom.grid(row=2, column=0, sticky="ew")

        bottom.columnconfigure(0, weight=1)
        for c in range(1, 7):
            bottom.columnconfigure(c, weight=0)

        # Sample rate display
        self.sample_rate_var = tk.StringVar(value="–")
        status_lbl = ttk.Label(bottom, textvariable=self.sample_rate_var, style="Status.TLabel")
        status_lbl.grid(row=0, column=0, sticky="w")

        # Tare button
        self.tare_btn = ttk.Button(bottom, text="Tare", command=self.tare_stream,
                                   style="Secondary.TButton")
        self.tare_btn.grid(row=0, column=1, sticky="e", padx=(5, 5))

        self.plot_mode_btn = ttk.Button(
            bottom, textvariable=self.top_plot_mode_var, command=self._cycle_top_plot_mode,
            style="Secondary.TButton"
        )
        self.plot_mode_btn.grid(row=0, column=2, sticky="e", padx=(5, 5))

        # Capture controls (row 1)
        capture_lbl = ttk.Label(bottom, text="Capture:", style="Inline.TLabel")
        capture_lbl.grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0))

        self.start_cap_btn = ttk.Button(
            bottom, text="Start", command=self.start_capture,
            style="Accent.TButton"
        )
        self.start_cap_btn.grid(row=1, column=1, sticky="e", padx=(5, 5), pady=(5, 0))

        self.stop_cap_btn = ttk.Button(
            bottom, text="Stop", command=self.stop_capture, state="disabled",
            style="Secondary.TButton"
        )
        self.stop_cap_btn.grid(row=1, column=2, sticky="e", padx=(5, 5), pady=(5, 0))

        self.cancel_cap_btn = ttk.Button(
            bottom, text="Cancel", command=self.cancel_capture, state="disabled",
            style="Secondary.TButton"
        )
        self.cancel_cap_btn.grid(row=1, column=3, sticky="e", padx=(5, 5), pady=(5, 0))

        # Quit button
        quit_btn = ttk.Button(bottom, text="Quit", command=self.on_close,
                              style="Secondary.TButton")
        quit_btn.grid(row=1, column=4, sticky="e", padx=(10, 0), pady=(5, 0))

        # Clean shutdown on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_plot_panel(self, parent):
        if hasattr(self, "plot_frame") and self.plot_frame is not None:
            self.plot_frame.destroy()

        self.plot_frame = ttk.Frame(parent)
        self.plot_frame.grid(row=0, column=0, sticky="nsew")
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.fig.patch.set_facecolor("#f4f5f7")

        # Top plot: Compensated magnetic delta
        self.ax_raw = self.fig.add_subplot(211)
        self.ax_raw.set_facecolor("#f7f7f7")
        self.ax_raw.grid(axis="y", color="#cccccc", linewidth=0.5, alpha=0.35)
        self.ax_raw.spines["top"].set_visible(False)
        self.ax_raw.spines["right"].set_visible(False)
        self.ax_raw.spines["left"].set_color("#888888")
        self.ax_raw.spines["bottom"].set_color("#888888")
        self.ax_raw.tick_params(colors="#444444")
        self.ax_raw.set_title("Compensated Magnetic Delta (last {:.1f} s)".format(self.history_seconds))
        self.ax_raw.set_ylabel("Field (µT)")
        self.ax_raw.tick_params(labelbottom=False)

        self.text_overlay_raw = self.ax_raw.text(
            0.02, 0.98, "",
            transform=self.ax_raw.transAxes,
            va="top", ha="left",
            fontsize=11, fontfamily="Consolas", color="#222222",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.6, edgecolor="#cccccc"),
        )

        # Raw plot lines
        x_color = "#d62728"
        y_color = "#2ca02c"
        z_color = "#1f77b4"

        self.shadowX_raw, = self.ax_raw.plot([], [], linewidth=4, alpha=0.10, color=x_color)
        self.shadowY_raw, = self.ax_raw.plot([], [], linewidth=4, alpha=0.10, color=y_color)
        self.shadowZ_raw, = self.ax_raw.plot([], [], linewidth=4, alpha=0.10, color=z_color)
        self.line_x_raw, = self.ax_raw.plot([], [], label="X", linewidth=2.5, color=x_color)
        self.line_y_raw, = self.ax_raw.plot([], [], label="Y", linewidth=2.5, color=y_color)
        self.line_z_raw, = self.ax_raw.plot([], [], label="Z", linewidth=2.5, color=z_color)
        self.ax_raw.legend(loc="upper right", frameon=False)

        # Bottom plot: Force measurements
        self.ax_force = self.fig.add_subplot(212, sharex=self.ax_raw)
        self.ax_force.set_facecolor("#f7f7f7")
        self.ax_force.grid(axis="y", color="#cccccc", linewidth=0.5, alpha=0.35)
        self.ax_force.spines["top"].set_visible(False)
        self.ax_force.spines["right"].set_visible(False)
        self.ax_force.spines["left"].set_color("#888888")
        self.ax_force.spines["bottom"].set_color("#888888")
        self.ax_force.tick_params(colors="#444444")
        self.ax_force.set_title("Force Measurements")
        self.ax_force.set_xlabel("Time (s)")
        self.ax_force.set_ylabel("Force (N)")

        self.text_overlay_force = self.ax_force.text(
            0.02, 0.98, "",
            transform=self.ax_force.transAxes,
            va="top", ha="left",
            fontsize=11, fontfamily="Consolas", color="#222222",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.6, edgecolor="#cccccc"),
        )

        # Force plot lines
        grip_color = "#800080"
        down_color = "#ff7f0e"

        self.shadow_grip, = self.ax_force.plot([], [], linewidth=4, alpha=0.10, color=grip_color)
        self.shadow_down, = self.ax_force.plot([], [], linewidth=4, alpha=0.10, color=down_color)
        self.line_grip, = self.ax_force.plot([], [], label="Grip Force", linewidth=2.5, color=grip_color)
        self.line_down, = self.ax_force.plot([], [], label="Downforce", linewidth=2.5, color=down_color)
        self.ax_force.legend(loc="upper right", frameon=False)

        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _update_stream_status(self):
        status = "Status: Streaming"
        if getattr(self.ros_node, "ambient_stale", True):
            status += " (Ambient Stale)"
        if self.capture_active:
            status += " (Capturing)"
        self.status_var.set(status)

    def _cycle_top_plot_mode(self):
        if self.top_plot_mode == "compensated":
            self.top_plot_mode = "force_raw"
            self.top_plot_mode_var.set("Plot: Force")
        elif self.top_plot_mode == "force_raw":
            self.top_plot_mode = "ambient_raw"
            self.top_plot_mode_var.set("Plot: Ambient")
        else:
            self.top_plot_mode = "compensated"
            self.top_plot_mode_var.set("Plot: Comp")

    def tare_stream(self):
        """Set simultaneous tare offsets for force and ambient sensors."""
        if self.Xf_raw and self.Yf_raw and self.Zf_raw and self.Xa_raw and self.Ya_raw and self.Za_raw:
            self.tare_fx = self.Xf_raw[-1]
            self.tare_fy = self.Yf_raw[-1]
            self.tare_fz = self.Zf_raw[-1]
            self.tare_ax = self.Xa_raw[-1]
            self.tare_ay = self.Ya_raw[-1]
            self.tare_az = self.Za_raw[-1]
            self.ros_node.get_logger().info(
                "Dual tare set: "
                f"force=({self.tare_fx:.2f}, {self.tare_fy:.2f}, {self.tare_fz:.2f}) "
                f"ambient=({self.tare_ax:.2f}, {self.tare_ay:.2f}, {self.tare_az:.2f})"
            )
            self.status_var.set("Status: Tared (Dual)")
            self.root.after(1000, self._update_stream_status)

    def start_capture(self):
        """Begin buffering data for CSV export."""
        self.capture_active = True
        self.capture_buffer.clear()
        self._update_stream_status()
        self.start_cap_btn.config(state="disabled")
        self.stop_cap_btn.config(state="normal")
        self.cancel_cap_btn.config(state="normal")

    def stop_capture(self):
        """Stop capture and save to CSV."""
        if not self.capture_active:
            return
        self.capture_active = False

        data = list(self.capture_buffer)
        self.capture_buffer.clear()

        self._update_stream_status()
        self.start_cap_btn.config(state="normal")
        self.stop_cap_btn.config(state="disabled")
        self.cancel_cap_btn.config(state="disabled")

        if not data:
            self.ros_node.get_logger().info("Capture finished, but no data to save.")
            return

        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mlx_capture_{ts_str}.csv"

        try:
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "force_x_raw_uT", "force_y_raw_uT", "force_z_raw_uT",
                    "ambient_x_raw_uT", "ambient_y_raw_uT", "ambient_z_raw_uT",
                    "force_x_tared_uT", "force_y_tared_uT", "force_z_tared_uT",
                    "ambient_x_tared_uT", "ambient_y_tared_uT", "ambient_z_tared_uT",
                    "comp_x_uT", "comp_y_uT", "comp_z_uT",
                    "grip_force_N", "downforce_N",
                ])
                for row in data:
                    (
                        t, fx_raw, fy_raw, fz_raw, ax_raw, ay_raw, az_raw,
                        fx_t, fy_t, fz_t, ax_t, ay_t, az_t,
                        x_comp, y_comp, z_comp, grip, down
                    ) = row
                    writer.writerow([
                        f"{t:.6f}",
                        f"{fx_raw:.2f}", f"{fy_raw:.2f}", f"{fz_raw:.2f}",
                        f"{ax_raw:.2f}", f"{ay_raw:.2f}", f"{az_raw:.2f}",
                        f"{fx_t:.2f}", f"{fy_t:.2f}", f"{fz_t:.2f}",
                        f"{ax_t:.2f}", f"{ay_t:.2f}", f"{az_t:.2f}",
                        f"{x_comp:.2f}", f"{y_comp:.2f}", f"{z_comp:.2f}",
                        f"{grip:.4f}", f"{down:.4f}",
                    ])

            self.ros_node.get_logger().info(f"Capture saved to {filename} ({len(data)} samples)")
        except Exception as e:
            self.ros_node.get_logger().error(f"Error saving capture: {e}")

    def cancel_capture(self):
        """Cancel capture without saving."""
        self.capture_active = False
        self.capture_buffer.clear()
        self._update_stream_status()
        self.start_cap_btn.config(state="normal")
        self.stop_cap_btn.config(state="disabled")
        self.cancel_cap_btn.config(state="disabled")
        self.ros_node.get_logger().info("Capture cancelled.")

    def on_close(self):
        """Clean shutdown."""
        self.root.destroy()

    def _schedule_update(self):
        """Periodic GUI update loop."""
        self._update_from_queue()
        self._update_plot()
        self._update_sample_rate()
        self.root.after(50, self._schedule_update)

    def _update_from_queue(self):
        """Drain data queue and update buffers."""
        updated = False

        while not self.data_queue.empty():
            t, fx_raw, fy_raw, fz_raw, ax_raw, ay_raw, az_raw = self.data_queue.get()

            # Apply independent tare offsets.
            fx_t = fx_raw - self.tare_fx
            fy_t = fy_raw - self.tare_fy
            fz_t = fz_raw - self.tare_fz
            ax_t = ax_raw - self.tare_ax
            ay_t = ay_raw - self.tare_ay
            az_t = az_raw - self.tare_az

            # Compensated delta: (force_tared - ambient_tared)
            x = fx_t - ax_t
            y = fy_t - ay_t
            z = fz_t - az_t

            self.times.append(t)
            self.X_raw.append(x)
            self.Y_raw.append(y)
            self.Z_raw.append(z)

            self.Xf_raw.append(fx_raw)
            self.Yf_raw.append(fy_raw)
            self.Zf_raw.append(fz_raw)
            self.Xa_raw.append(ax_raw)
            self.Ya_raw.append(ay_raw)
            self.Za_raw.append(az_raw)

            self.Xf_tared.append(fx_t)
            self.Yf_tared.append(fy_t)
            self.Zf_tared.append(fz_t)
            self.Xa_tared.append(ax_t)
            self.Ya_tared.append(ay_t)
            self.Za_tared.append(az_t)

            updated = True

            # Simple Moving Average
            window = self.sma_window_samples

            xs = list(self.X_raw)[-window:] if len(self.X_raw) > window else list(self.X_raw)
            ys = list(self.Y_raw)[-window:] if len(self.Y_raw) > window else list(self.Y_raw)
            zs = list(self.Z_raw)[-window:] if len(self.Z_raw) > window else list(self.Z_raw)

            sma_x = sum(xs) / len(xs) if xs else x
            sma_y = sum(ys) / len(ys) if ys else y
            sma_z = sum(zs) / len(zs) if zs else z

            self.X_sma.append(sma_x)
            self.Y_sma.append(sma_y)
            self.Z_sma.append(sma_z)

            # Calculate forces using compensated values
            grip = self.calculate_grip_force(x, y, z)
            down = self.calculate_downforce(x, y, z)

            self.grip_force.append(grip)
            self.downforce.append(down)

            # SMA for forces
            grips = list(self.grip_force)[-window:] if len(self.grip_force) > window else list(self.grip_force)
            downs = list(self.downforce)[-window:] if len(self.downforce) > window else list(self.downforce)

            sma_grip = sum(grips) / len(grips) if grips else grip
            sma_down = sum(downs) / len(downs) if downs else down

            self.grip_force_sma.append(sma_grip)
            self.downforce_sma.append(sma_down)

            # Capture if active
            if self.capture_active:
                self.capture_buffer.append((
                    t,
                    fx_raw, fy_raw, fz_raw,
                    ax_raw, ay_raw, az_raw,
                    fx_t, fy_t, fz_t,
                    ax_t, ay_t, az_t,
                    x, y, z,
                    grip, down,
                ))

        # Trim to history window
        if updated:
            cutoff = time.time() - self.history_seconds
            while self.times and self.times[0] < cutoff:
                self.times.popleft()
                self.X_raw.popleft()
                self.Y_raw.popleft()
                self.Z_raw.popleft()
                self.X_sma.popleft()
                self.Y_sma.popleft()
                self.Z_sma.popleft()
                self.Xf_raw.popleft()
                self.Yf_raw.popleft()
                self.Zf_raw.popleft()
                self.Xa_raw.popleft()
                self.Ya_raw.popleft()
                self.Za_raw.popleft()
                self.Xf_tared.popleft()
                self.Yf_tared.popleft()
                self.Zf_tared.popleft()
                self.Xa_tared.popleft()
                self.Ya_tared.popleft()
                self.Za_tared.popleft()
                self.grip_force.popleft()
                self.downforce.popleft()
                self.grip_force_sma.popleft()
                self.downforce_sma.popleft()

        self._update_stream_status()

    def _update_plot(self):
        """Update matplotlib canvas with current data."""
        if len(self.times) < 2:
            return

        t0 = self.times[0]
        t_rel = [t - t0 for t in self.times]

        if self.top_plot_mode == "force_raw":
            X_disp = self.Xf_raw
            Y_disp = self.Yf_raw
            Z_disp = self.Zf_raw
            self.ax_raw.set_title("Force Sensor Raw Magnetic Field (last {:.1f} s)".format(self.history_seconds))
            overlay_label = "Force"
        elif self.top_plot_mode == "ambient_raw":
            X_disp = self.Xa_raw
            Y_disp = self.Ya_raw
            Z_disp = self.Za_raw
            self.ax_raw.set_title("Ambient Sensor Raw Magnetic Field (last {:.1f} s)".format(self.history_seconds))
            overlay_label = "Ambient"
        else:
            # Use SMA for compensated display if available
            if self.X_sma and len(self.X_sma) == len(self.times):
                X_disp = self.X_sma
                Y_disp = self.Y_sma
                Z_disp = self.Z_sma
            else:
                X_disp = self.X_raw
                Y_disp = self.Y_raw
                Z_disp = self.Z_raw
            self.ax_raw.set_title("Compensated Magnetic Delta (last {:.1f} s)".format(self.history_seconds))
            overlay_label = "Comp"

        # Update compensated magnetic field lines
        self.line_x_raw.set_data(t_rel, X_disp)
        self.line_y_raw.set_data(t_rel, Y_disp)
        self.line_z_raw.set_data(t_rel, Z_disp)

        self.shadowX_raw.set_data(t_rel, X_disp)
        self.shadowY_raw.set_data(t_rel, Y_disp)
        self.shadowZ_raw.set_data(t_rel, Z_disp)

        # Update text overlay for compensated data
        if X_disp and Y_disp and Z_disp:
            self.text_overlay_raw.set_text(
                f"{overlay_label} X: {X_disp[-1]:6.2f} µT\n"
                f"{overlay_label} Y: {Y_disp[-1]:6.2f} µT\n"
                f"{overlay_label} Z: {Z_disp[-1]:6.2f} µT"
            )

        # Update raw axis limits
        self.ax_raw.set_xlim(0, self.history_seconds)

        all_raw = (*X_disp, *Y_disp, *Z_disp)
        if all_raw:
            ymin = min(all_raw)
            ymax = max(all_raw)
            if ymin == ymax:
                ymin -= 1
                ymax += 1
            margin = (ymax - ymin) * 0.15

            ymin_r = min(ymin - margin, self.min_y_raw)
            ymax_r = max(ymax + margin, self.max_y_raw)

            self.ax_raw.set_ylim(ymin_r, ymax_r)

        # Update force plot
        if self.grip_force_sma and len(self.grip_force_sma) == len(self.times):
            grip_disp = self.grip_force_sma
            down_disp = self.downforce_sma
        else:
            grip_disp = self.grip_force
            down_disp = self.downforce

        if grip_disp and down_disp:
            self.line_grip.set_data(t_rel, grip_disp)
            self.line_down.set_data(t_rel, down_disp)

            self.shadow_grip.set_data(t_rel, grip_disp)
            self.shadow_down.set_data(t_rel, down_disp)

            # Update text overlay for forces
            self.text_overlay_force.set_text(
                f"Grip Force: {grip_disp[-1]:6.2f} N\n"
                f"Downforce: {down_disp[-1]:6.2f} N"
            )

            # Update force axis limits
            self.ax_force.set_xlim(0, self.history_seconds)

            all_force = (*grip_disp, *down_disp)
            if all_force:
                ymin = min(all_force)
                ymax = max(all_force)
                if ymin == ymax:
                    ymin -= 0.1
                    ymax += 0.1
                margin = (ymax - ymin) * 0.15

                ymin_f = min(ymin - margin, self.min_y_force)
                ymax_f = max(ymax + margin, self.max_y_force)

                self.ax_force.set_ylim(ymin_f, ymax_f)

        self.canvas.draw_idle()

    def _update_sample_rate(self):
        """Estimate sample rate from timestamps."""
        if len(self.times) >= 5:
            dt = self.times[-1] - self.times[0]
            if dt > 0:
                rate = (len(self.times) - 1) / dt
                self.sample_rate_var.set(f"Approx. sample rate: {rate:.1f} Hz")
        else:
            self.sample_rate_var.set("Approx. sample rate: –")


class MLXGuiNode(Node):
    """ROS2 node that subscribes to /mlx and /mlx_ambient and feeds combined data to GUI."""

    def __init__(self, data_queue):
        super().__init__('mlx_gui_node')
        self.data_queue = data_queue

        self.latest_ambient = (0.0, 0.0, 0.0)
        self.latest_ambient_time = None
        self.ambient_stale_timeout_s = 0.5
        self.ambient_stale = True

        self.force_subscription = self.create_subscription(
            MagneticField,
            '/mlx',
            self.mlx_callback,
            10
        )
        self.ambient_subscription = self.create_subscription(
            MagneticField,
            '/mlx_ambient',
            self.mlx_ambient_callback,
            10
        )

        self.get_logger().info("MLX GUI Node started, subscribing to /mlx and /mlx_ambient")

    def mlx_ambient_callback(self, msg):
        """Callback for /mlx_ambient topic - cache latest ambient baseline sample."""
        self.latest_ambient = (
            msg.magnetic_field.x,
            msg.magnetic_field.y,
            msg.magnetic_field.z,
        )
        self.latest_ambient_time = time.time()
        self.ambient_stale = False

    def mlx_callback(self, msg):
        """Callback for /mlx topic - combine with latest ambient and push to queue."""
        t = time.time()
        fx = msg.magnetic_field.x
        fy = msg.magnetic_field.y
        fz = msg.magnetic_field.z

        ax, ay, az = self.latest_ambient
        if self.latest_ambient_time is None:
            self.ambient_stale = True
        else:
            self.ambient_stale = (t - self.latest_ambient_time) > self.ambient_stale_timeout_s

        # Queue payload: (t, fx, fy, fz, ax, ay, az)
        self.data_queue.put((t, fx, fy, fz, ax, ay, az))
        self.get_logger().debug(
            f"Received force+ambient: t={t:.2f}, f=({fx:.6f},{fy:.6f},{fz:.6f}), "
            f"a=({ax:.6f},{ay:.6f},{az:.6f}), stale={self.ambient_stale}"
        )


def main(args=None):
    """Main entry point - initialize ROS2 and launch GUI."""
    rclpy.init(args=args)

    data_queue = queue.Queue()
    node = MLXGuiNode(data_queue)

    # Run ROS2 spin in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Run GUI on main thread
    root = tk.Tk()
    gui = MLXLiveGUI(root, node, data_queue, history_seconds=5.0)
    root.mainloop()

    # Cleanup
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
