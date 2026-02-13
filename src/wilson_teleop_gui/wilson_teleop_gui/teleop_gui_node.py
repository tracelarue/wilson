#!/usr/bin/env python3

import threading
import tkinter as tk
from tkinter import ttk

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class TeleopGuiNode(Node):
    """Publishes Twist commands based on GUI button presses."""

    def __init__(self) -> None:
        super().__init__("wilson_teleop_gui")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("linear_speed", 0.25)
        self.declare_parameter("angular_speed", 0.9)
        self.declare_parameter("publish_rate_hz", 20.0)

        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        self.linear_speed = self.get_parameter("linear_speed").get_parameter_value().double_value
        self.angular_speed = self.get_parameter("angular_speed").get_parameter_value().double_value
        self.publish_rate = self.get_parameter("publish_rate_hz").get_parameter_value().double_value

        self._publisher = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self._cmd_lock = threading.Lock()
        self._target_linear = 0.0
        self._target_angular = 0.0

        timer_period = 1.0 / max(self.publish_rate, 1.0)
        self._timer = self.create_timer(timer_period, self._publish_twist)

        self.get_logger().info(
            f"Teleop GUI publishing Twist to {self.cmd_vel_topic} "
            f"(linear_speed={self.linear_speed}, angular_speed={self.angular_speed}, "
            f"rate={self.publish_rate}Hz)"
        )

    def command(self, linear: float, angular: float, publish_now: bool = False) -> None:
        with self._cmd_lock:
            self._target_linear = linear
            self._target_angular = angular
        if publish_now:
            self._publish_twist()

    def stop(self) -> None:
        self.command(0.0, 0.0, publish_now=True)

    def _publish_twist(self) -> None:
        with self._cmd_lock:
            linear = float(self._target_linear)
            angular = float(self._target_angular)

        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self._publisher.publish(msg)


class TeleopGuiApp:
    """Simple Tkinter front-end for differential-drive teleoperation."""

    def __init__(self, node: TeleopGuiNode) -> None:
        self._node = node
        self._active_button = None
        self._root = tk.Tk()
        self._root.title("Wilson Teleop")
        self._root.geometry("420x360")
        self._root.minsize(360, 320)
        self._root.configure(bg="#ecf2f5")

        self._init_style()
        self._build_layout()
        self._bind_keyboard()

        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.bind("<FocusOut>", self._on_focus_out)

    def run(self) -> None:
        self._root.mainloop()

    def _init_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Main.TFrame", background="#ecf2f5")
        style.configure(
            "Header.TLabel",
            background="#ecf2f5",
            foreground="#0d3b50",
            font=("Helvetica", 16, "bold"),
        )
        style.configure(
            "Hint.TLabel",
            background="#ecf2f5",
            foreground="#335266",
            font=("Helvetica", 10),
        )
        style.configure(
            "Move.TButton",
            font=("Helvetica", 11, "bold"),
            padding=(10, 10),
        )
        style.configure("Stop.TButton", font=("Helvetica", 11, "bold"), padding=(10, 10))

    def _build_layout(self) -> None:
        frame = ttk.Frame(self._root, style="Main.TFrame", padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

        title = ttk.Label(frame, text="Differential Drive Teleop", style="Header.TLabel")
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 3))

        hint = ttk.Label(
            frame,
            text="Press and hold buttons. Release to stop. Keyboard: W/A/S/D + Q/E/Z/C.",
            style="Hint.TLabel",
        )
        hint.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 10))

        self._add_move_button(frame, "FWD LEFT", 2, 0, 1.0, 1.0)
        self._add_move_button(frame, "FORWARD", 2, 1, 1.0, 0.0)
        self._add_move_button(frame, "FWD RIGHT", 2, 2, 1.0, -1.0)

        self._add_move_button(frame, "LEFT", 3, 0, 0.0, 1.0)
        stop_btn = ttk.Button(frame, text="STOP", style="Stop.TButton", command=self._on_release)
        stop_btn.grid(row=3, column=1, padx=5, pady=5, sticky="nsew")
        self._add_move_button(frame, "RIGHT", 3, 2, 0.0, -1.0)

        self._add_move_button(frame, "REV LEFT", 4, 0, -1.0, 1.0)
        self._add_move_button(frame, "REVERSE", 4, 1, -1.0, 0.0)
        self._add_move_button(frame, "REV RIGHT", 4, 2, -1.0, -1.0)

        self._linear_speed_var = tk.DoubleVar(value=self._node.linear_speed)
        self._angular_speed_var = tk.DoubleVar(value=self._node.angular_speed)
        self._linear_speed_text = tk.StringVar()
        self._angular_speed_text = tk.StringVar()
        self._update_speed_labels()

        linear_label = ttk.Label(frame, text="Linear Speed", style="Hint.TLabel")
        linear_label.grid(row=5, column=0, sticky="w", pady=(8, 2))
        linear_value = ttk.Label(frame, textvariable=self._linear_speed_text, style="Hint.TLabel")
        linear_value.grid(row=5, column=2, sticky="e", pady=(8, 2))
        linear_slider = ttk.Scale(
            frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self._linear_speed_var,
            command=lambda _v: self._update_speed_labels(),
        )
        linear_slider.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 6))

        angular_label = ttk.Label(frame, text="Angular Speed", style="Hint.TLabel")
        angular_label.grid(row=7, column=0, sticky="w", pady=(4, 2))
        angular_value = ttk.Label(frame, textvariable=self._angular_speed_text, style="Hint.TLabel")
        angular_value.grid(row=7, column=2, sticky="e", pady=(4, 2))
        angular_slider = ttk.Scale(
            frame,
            from_=0.0,
            to=2.5,
            orient=tk.HORIZONTAL,
            variable=self._angular_speed_var,
            command=lambda _v: self._update_speed_labels(),
        )
        angular_slider.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(0, 6))

        status_frame = ttk.Frame(frame, style="Main.TFrame")
        status_frame.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        self._status = tk.StringVar(value=self._status_text(0.0, 0.0))
        status_label = ttk.Label(status_frame, textvariable=self._status, style="Hint.TLabel")
        status_label.pack(anchor="w")

    def _bind_keyboard(self) -> None:
        key_map = {
            "w": (1.0, 0.0),
            "a": (0.0, 1.0),
            "s": (-1.0, 0.0),
            "d": (0.0, -1.0),
            "q": (1.0, 1.0),
            "e": (1.0, -1.0),
            "z": (-1.0, 1.0),
            "c": (-1.0, -1.0),
        }

        def on_key_press(evt: tk.Event) -> None:
            key = evt.keysym.lower()
            if key in key_map:
                linear_factor, angular_factor = key_map[key]
                self._on_press(linear_factor, angular_factor)
            elif key == "space":
                self._on_release()

        self._root.bind("<KeyPress>", on_key_press)
        self._root.bind("<KeyRelease>", lambda _evt: self._on_release())

    def _add_move_button(
        self,
        parent: ttk.Frame,
        text: str,
        row: int,
        col: int,
        linear_factor: float,
        angular_factor: float,
    ) -> None:
        btn = ttk.Button(parent, text=text, style="Move.TButton")
        btn.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        btn.bind(
            "<ButtonPress-1>",
            lambda evt: self._on_press(linear_factor, angular_factor, evt.widget),
        )
        btn.bind("<ButtonRelease-1>", lambda evt: self._on_release(evt.widget))
        btn.bind("<Leave>", lambda evt: self._on_release(evt.widget))

    def _on_press(self, linear_factor: float, angular_factor: float, source=None) -> None:
        self._active_button = source
        linear = linear_factor * self._linear_speed_var.get()
        angular = angular_factor * self._angular_speed_var.get()
        self._node.command(linear, angular, publish_now=True)
        self._status.set(self._status_text(linear, angular))

    def _on_release(self, source=None) -> None:
        if source is not None and self._active_button is not None and source is not self._active_button:
            return
        self._active_button = None
        self._node.stop()
        self._status.set(self._status_text(0.0, 0.0))

    @staticmethod
    def _status_text(linear: float, angular: float) -> str:
        return f"cmd_vel  linear.x={linear:.2f} m/s  angular.z={angular:.2f} rad/s"

    def _on_close(self) -> None:
        self._node.stop()
        self._root.quit()
        self._root.destroy()

    def _update_speed_labels(self) -> None:
        self._linear_speed_text.set(f"{self._linear_speed_var.get():.2f} m/s")
        self._angular_speed_text.set(f"{self._angular_speed_var.get():.2f} rad/s")

    def _on_focus_out(self, _evt) -> None:
        # Defer until focus settles; only stop if focus left this window.
        self._root.after_idle(self._stop_if_focus_left_window)

    def _stop_if_focus_left_window(self) -> None:
        focused_widget = self._root.focus_get()
        current = focused_widget
        while current is not None:
            if current is self._root:
                return
            current = current.master
        self._on_release()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeleopGuiNode()

    executor_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    executor_thread.start()

    try:
        app = TeleopGuiApp(node)
        app.run()
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()
        executor_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
