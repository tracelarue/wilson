#!/usr/bin/env python3

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
    """Tkinter GUI for live visualization of MLX90393 magnetometer data from ROS2 /mlx topic."""

    def __init__(self, root, ros_node, data_queue, history_seconds=5.0):
        self.plot_frame = None
        self.root = root
        self.ros_node = ros_node
        self.data_queue = data_queue
        self.root.title("MLX Magnetometer – Live View")

        # Basic window sizing
        self.root.geometry("1100x650")
        self.history_seconds = history_seconds

        # Data buffers
        self.times = deque()
        self.X_raw = deque()
        self.Y_raw = deque()
        self.Z_raw = deque()

        # Tare offsets (applied to raw data)
        self.tare_x = 0.0
        self.tare_y = 0.0
        self.tare_z = 0.0

        self.min_y_raw = -10
        self.max_y_raw = 10

        # Simple Moving Average (SMA) buffers
        self.sma_window_samples = 5
        self.X_sma = deque()
        self.Y_sma = deque()
        self.Z_sma = deque()

        # Capture state
        self.capture_active = False
        self.capture_buffer = []

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
            text="ROS2 /mlx subscriber • Axioforce demo",
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
        for c in range(1, 6):
            bottom.columnconfigure(c, weight=0)

        # Sample rate display
        self.sample_rate_var = tk.StringVar(value="–")
        status_lbl = ttk.Label(bottom, textvariable=self.sample_rate_var, style="Status.TLabel")
        status_lbl.grid(row=0, column=0, sticky="w")

        # Tare button
        self.tare_btn = ttk.Button(bottom, text="Tare", command=self.tare_stream,
                                   style="Secondary.TButton")
        self.tare_btn.grid(row=0, column=1, sticky="e", padx=(5, 5))

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

        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.fig.patch.set_facecolor("#f4f5f7")

        # Single plot for raw magnetic field
        self.ax_raw = self.fig.add_subplot(111)
        self.ax_raw.set_facecolor("#f7f7f7")
        self.ax_raw.grid(axis="y", color="#cccccc", linewidth=0.5, alpha=0.35)
        self.ax_raw.spines["top"].set_visible(False)
        self.ax_raw.spines["right"].set_visible(False)
        self.ax_raw.spines["left"].set_color("#888888")
        self.ax_raw.spines["bottom"].set_color("#888888")
        self.ax_raw.tick_params(colors="#444444")
        self.ax_raw.set_title("Magnetic Field (last {:.1f} s)".format(self.history_seconds))
        self.ax_raw.set_xlabel("Time (s)")
        self.ax_raw.set_ylabel("Field (µT)")

        self.text_overlay_raw = self.ax_raw.text(
            0.02, 0.98, "",
            transform=self.ax_raw.transAxes,
            va="top", ha="left",
            fontsize=11, fontfamily="Consolas", color="#222222",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.6, edgecolor="#cccccc"),
        )

        # Plot lines
        self.shadowX_raw, = self.ax_raw.plot([], [], linewidth=4, alpha=0.10)
        self.shadowY_raw, = self.ax_raw.plot([], [], linewidth=4, alpha=0.10)
        self.shadowZ_raw, = self.ax_raw.plot([], [], linewidth=4, alpha=0.10)
        self.line_x_raw, = self.ax_raw.plot([], [], label="X", linewidth=2.5)
        self.line_y_raw, = self.ax_raw.plot([], [], label="Y", linewidth=2.5)
        self.line_z_raw, = self.ax_raw.plot([], [], label="Z", linewidth=2.5)
        self.ax_raw.legend(loc="upper right", frameon=False)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def tare_stream(self):
        """Set current magnetic field values as zero offsets."""
        if self.X_raw and self.Y_raw and self.Z_raw:
            self.tare_x = self.X_raw[-1]
            self.tare_y = self.Y_raw[-1]
            self.tare_z = self.Z_raw[-1]
            self.ros_node.get_logger().info(f"Tared: X={self.tare_x:.2f}, Y={self.tare_y:.2f}, Z={self.tare_z:.2f}")
            self.status_var.set("Status: Tared")
            self.root.after(1000, lambda: self.status_var.set("Status: Streaming"))

    def start_capture(self):
        """Begin buffering data for CSV export."""
        self.capture_active = True
        self.capture_buffer.clear()
        self.status_var.set("Status: Streaming (Capturing)")
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

        self.status_var.set("Status: Streaming")
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
                writer.writerow(["timestamp", "X_uT", "Y_uT", "Z_uT"])
                for t, x, y, z in data:
                    writer.writerow([
                        f"{t:.6f}",
                        f"{x:.2f}",
                        f"{y:.2f}",
                        f"{z:.2f}",
                    ])

            self.ros_node.get_logger().info(f"Capture saved to {filename} ({len(data)} samples)")
        except Exception as e:
            self.ros_node.get_logger().error(f"Error saving capture: {e}")

    def cancel_capture(self):
        """Cancel capture without saving."""
        self.capture_active = False
        self.capture_buffer.clear()
        self.status_var.set("Status: Streaming")
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
        self.root.after(20, self._schedule_update)

    def _update_from_queue(self):
        """Drain data queue and update buffers."""
        updated = False

        while not self.data_queue.empty():
            t, x_raw, y_raw, z_raw = self.data_queue.get()

            # Apply tare offsets
            x = x_raw - self.tare_x
            y = y_raw - self.tare_y
            z = z_raw - self.tare_z

            self.times.append(t)
            self.X_raw.append(x)
            self.Y_raw.append(y)
            self.Z_raw.append(z)
            updated = True

            # Capture if active
            if self.capture_active:
                self.capture_buffer.append((t, x, y, z))

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

    def _update_plot(self):
        """Update matplotlib canvas with current data."""
        if len(self.times) < 2:
            return

        t0 = self.times[0]
        t_rel = [t - t0 for t in self.times]

        # Use SMA for display if available
        if self.X_sma and len(self.X_sma) == len(self.times):
            X_disp = self.X_sma
            Y_disp = self.Y_sma
            Z_disp = self.Z_sma
        else:
            X_disp = self.X_raw
            Y_disp = self.Y_raw
            Z_disp = self.Z_raw

        # Update lines
        self.line_x_raw.set_data(t_rel, X_disp)
        self.line_y_raw.set_data(t_rel, Y_disp)
        self.line_z_raw.set_data(t_rel, Z_disp)

        self.shadowX_raw.set_data(t_rel, X_disp)
        self.shadowY_raw.set_data(t_rel, Y_disp)
        self.shadowZ_raw.set_data(t_rel, Z_disp)

        # Update text overlay
        if X_disp and Y_disp and Z_disp:
            self.text_overlay_raw.set_text(
                f"X: {X_disp[-1]:6.2f} µT\n"
                f"Y: {Y_disp[-1]:6.2f} µT\n"
                f"Z: {Z_disp[-1]:6.2f} µT"
            )

        # Update axis limits
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
    """ROS2 node that subscribes to /mlx topic and feeds data to GUI."""

    def __init__(self, data_queue):
        super().__init__('mlx_gui_node')
        self.data_queue = data_queue

        self.subscription = self.create_subscription(
            MagneticField,
            '/mlx',
            self.mlx_callback,
            10
        )

        self.get_logger().info("MLX GUI Node started, subscribing to /mlx")

    def mlx_callback(self, msg):
        """Callback for /mlx topic - extract data and push to queue."""
        # Use current time instead of message timestamp (which may be 0)
        t = time.time()
        x = msg.magnetic_field.x
        y = msg.magnetic_field.y
        z = msg.magnetic_field.z

        self.data_queue.put((t, x, y, z))
        self.get_logger().debug(f"Received: t={t:.2f}, x={x:.6f}, y={y:.6f}, z={z:.6f}")


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
