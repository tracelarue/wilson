#!/usr/bin/env python3

import math
import threading
import tkinter as tk
from tkinter import ttk

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class TeleopGuiNode(Node):
    """Publishes Twist commands based on GUI interactions."""

    def __init__(self) -> None:
        super().__init__("wilson_teleop_gui")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("linear_speed", 0.25)
        self.declare_parameter("angular_speed", 0.25)
        self.declare_parameter("linear_deadband", 0.1)
        self.declare_parameter("angular_deadband", 0.1)
        self.declare_parameter("publish_rate_hz", 20.0)

        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        self.linear_speed = self.get_parameter("linear_speed").get_parameter_value().double_value
        self.angular_speed = self.get_parameter("angular_speed").get_parameter_value().double_value
        self.linear_deadband = self.get_parameter("linear_deadband").get_parameter_value().double_value
        self.angular_deadband = self.get_parameter("angular_deadband").get_parameter_value().double_value
        self.publish_rate = self.get_parameter("publish_rate_hz").get_parameter_value().double_value

        self.linear_deadband = min(max(self.linear_deadband, 0.0), 0.95)
        self.angular_deadband = min(max(self.angular_deadband, 0.0), 0.95)

        self._publisher = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self._cmd_lock = threading.Lock()
        self._target_linear = 0.0
        self._target_angular = 0.0

        timer_period = 1.0 / max(self.publish_rate, 1.0)
        self._timer = self.create_timer(timer_period, self._publish_twist)

        self.get_logger().info(
            f"Teleop GUI publishing Twist to {self.cmd_vel_topic} "
            f"(linear_speed={self.linear_speed}, angular_speed={self.angular_speed}, "
            f"linear_deadband={self.linear_deadband}, angular_deadband={self.angular_deadband}, "
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
    """Tkinter front-end with virtual joystick for diff-drive teleoperation."""

    def __init__(self, node: TeleopGuiNode) -> None:
        self._node = node
        self._root = tk.Tk()
        self._root.title("Wilson Teleop")
        self._root.geometry("700x620")
        self._root.minsize(640, 560)
        self._root.configure(bg="#ecf2f5")

        self._joy_center = (160, 160)
        self._joy_radius = 120
        self._joy_knob_radius = 22
        self._joy_active = False
        self._active_button_id = None
        self._active_button_factors = None

        self._init_style()
        self._build_layout()

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

    def _build_layout(self) -> None:
        frame = ttk.Frame(self._root, style="Main.TFrame", padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)

        title = ttk.Label(frame, text="Differential Drive Teleop", style="Header.TLabel")
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        hint = ttk.Label(
            frame,
            text="Use button pad or joystick. Press/hold to drive. Release to stop.",
            style="Hint.TLabel",
        )
        hint.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))

        button_pad = ttk.Frame(frame, style="Main.TFrame")
        button_pad.grid(row=2, column=0, sticky="n", padx=(0, 10), pady=(0, 6))
        self._build_button_pad(button_pad)

        self._build_joystick(frame, row=2, col=1)

        self._linear_speed_var = tk.DoubleVar(value=self._node.linear_speed)
        self._angular_speed_var = tk.DoubleVar(value=self._node.angular_speed)
        self._linear_deadband_var = tk.DoubleVar(value=self._node.linear_deadband)
        self._angular_deadband_var = tk.DoubleVar(value=self._node.angular_deadband)
        self._linear_speed_text = tk.StringVar()
        self._angular_speed_text = tk.StringVar()
        self._linear_deadband_text = tk.StringVar()
        self._angular_deadband_text = tk.StringVar()
        self._update_tuning_labels()

        linear_label = ttk.Label(frame, text="Max Linear Speed", style="Hint.TLabel")
        linear_label.grid(row=3, column=0, sticky="w", pady=(8, 2))
        linear_value = ttk.Label(frame, textvariable=self._linear_speed_text, style="Hint.TLabel")
        linear_value.grid(row=3, column=1, sticky="e", pady=(8, 2))

        linear_slider = ttk.Scale(
            frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self._linear_speed_var,
            command=lambda _v: self._on_tuning_changed(),
        )
        linear_slider.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        angular_label = ttk.Label(frame, text="Max Angular Speed", style="Hint.TLabel")
        angular_label.grid(row=5, column=0, sticky="w", pady=(4, 2))
        angular_value = ttk.Label(frame, textvariable=self._angular_speed_text, style="Hint.TLabel")
        angular_value.grid(row=5, column=1, sticky="e", pady=(4, 2))

        angular_slider = ttk.Scale(
            frame,
            from_=0.0,
            to=2.5,
            orient=tk.HORIZONTAL,
            variable=self._angular_speed_var,
            command=lambda _v: self._on_tuning_changed(),
        )
        angular_slider.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        linear_deadband_label = ttk.Label(frame, text="Linear Deadband", style="Hint.TLabel")
        linear_deadband_label.grid(row=7, column=0, sticky="w", pady=(4, 2))
        linear_deadband_value = ttk.Label(
            frame, textvariable=self._linear_deadband_text, style="Hint.TLabel"
        )
        linear_deadband_value.grid(row=7, column=1, sticky="e", pady=(4, 2))

        linear_deadband_slider = ttk.Scale(
            frame,
            from_=0.0,
            to=0.35,
            orient=tk.HORIZONTAL,
            variable=self._linear_deadband_var,
            command=lambda _v: self._on_tuning_changed(),
        )
        linear_deadband_slider.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        angular_deadband_label = ttk.Label(frame, text="Angular Deadband", style="Hint.TLabel")
        angular_deadband_label.grid(row=9, column=0, sticky="w", pady=(4, 2))
        angular_deadband_value = ttk.Label(
            frame, textvariable=self._angular_deadband_text, style="Hint.TLabel"
        )
        angular_deadband_value.grid(row=9, column=1, sticky="e", pady=(4, 2))

        angular_deadband_slider = ttk.Scale(
            frame,
            from_=0.0,
            to=0.35,
            orient=tk.HORIZONTAL,
            variable=self._angular_deadband_var,
            command=lambda _v: self._on_tuning_changed(),
        )
        angular_deadband_slider.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        self._status = tk.StringVar(value=self._status_text(0.0, 0.0))
        status_label = ttk.Label(frame, textvariable=self._status, style="Hint.TLabel")
        status_label.grid(row=11, column=0, columnspan=2, sticky="w")

    def _build_button_pad(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=1)

        pad_title = ttk.Label(parent, text="Button Pad", style="Hint.TLabel")
        pad_title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))

        self._add_move_button(parent, "FWD LEFT", 1, 0, 1.0, 1.0, "fwd_left")
        self._add_move_button(parent, "FORWARD", 1, 1, 1.0, 0.0, "forward")
        self._add_move_button(parent, "FWD RIGHT", 1, 2, 1.0, -1.0, "fwd_right")

        self._add_move_button(parent, "LEFT", 2, 0, 0.0, 1.0, "left")
        stop_btn = ttk.Button(parent, text="STOP", command=self._stop_motion)
        stop_btn.grid(row=2, column=1, padx=4, pady=4, sticky="nsew")
        self._add_move_button(parent, "RIGHT", 2, 2, 0.0, -1.0, "right")

        self._add_move_button(parent, "REV LEFT", 3, 0, -1.0, 1.0, "rev_left")
        self._add_move_button(parent, "REVERSE", 3, 1, -1.0, 0.0, "reverse")
        self._add_move_button(parent, "REV RIGHT", 3, 2, -1.0, -1.0, "rev_right")

    def _add_move_button(
        self,
        parent: ttk.Frame,
        text: str,
        row: int,
        col: int,
        linear_factor: float,
        angular_factor: float,
        button_id: str,
    ) -> None:
        btn = ttk.Button(parent, text=text)
        btn.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        btn.bind(
            "<ButtonPress-1>",
            lambda _evt: self._on_button_press(linear_factor, angular_factor, button_id),
        )
        btn.bind("<ButtonRelease-1>", lambda _evt: self._on_button_release(button_id))
        btn.bind("<Leave>", lambda _evt: self._on_button_release(button_id))

    def _build_joystick(self, parent: ttk.Frame, row: int, col: int) -> None:
        self._canvas = tk.Canvas(
            parent,
            width=320,
            height=320,
            bg="#f7fbfd",
            highlightthickness=1,
            highlightbackground="#aac2cf",
        )
        self._canvas.grid(row=row, column=col, sticky="n", pady=(0, 6))

        cx, cy = self._joy_center
        r = self._joy_radius

        self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#4f7486", width=2)
        self._canvas.create_line(cx - r, cy, cx + r, cy, fill="#9ab3bf", width=1)
        self._canvas.create_line(cx, cy - r, cx, cy + r, fill="#9ab3bf", width=1)

        kr = self._joy_knob_radius
        self._knob_id = self._canvas.create_oval(
            cx - kr,
            cy - kr,
            cx + kr,
            cy + kr,
            fill="#0d6b94",
            outline="#084c69",
            width=2,
        )

        self._canvas.bind("<ButtonPress-1>", self._on_joystick_press)
        self._canvas.bind("<B1-Motion>", self._on_joystick_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_joystick_release)

    def _on_joystick_press(self, event: tk.Event) -> None:
        self._joy_active = True
        self._active_button_id = None
        self._active_button_factors = None
        self._apply_joystick_position(event.x, event.y)

    def _on_joystick_drag(self, event: tk.Event) -> None:
        if not self._joy_active:
            return
        self._apply_joystick_position(event.x, event.y)

    def _on_joystick_release(self, _event: tk.Event) -> None:
        self._joy_active = False
        self._reset_joystick(stop_robot=True)

    def _apply_joystick_position(self, x: float, y: float) -> None:
        cx, cy = self._joy_center
        dx = x - cx
        dy = y - cy

        distance = math.hypot(dx, dy)
        if distance > self._joy_radius and distance > 0.0:
            scale = self._joy_radius / distance
            dx *= scale
            dy *= scale

        self._move_knob(cx + dx, cy + dy)

        x_norm = dx / self._joy_radius
        y_norm = -dy / self._joy_radius

        y_norm = self._apply_axis_deadband(y_norm, self._linear_deadband_var.get())
        x_norm = self._apply_axis_deadband(x_norm, self._angular_deadband_var.get())

        linear = y_norm * self._linear_speed_var.get()
        angular = -x_norm * self._angular_speed_var.get()
        self._node.command(linear, angular, publish_now=True)
        self._status.set(self._status_text(linear, angular))

    def _on_button_press(self, linear_factor: float, angular_factor: float, button_id: str) -> None:
        self._joy_active = False
        self._active_button_id = button_id
        self._active_button_factors = (linear_factor, angular_factor)
        self._reset_joystick(stop_robot=False)
        self._apply_button_command(linear_factor, angular_factor)

    def _on_button_release(self, button_id: str) -> None:
        if self._active_button_id != button_id:
            return
        self._active_button_id = None
        self._active_button_factors = None
        self._stop_motion()

    def _apply_button_command(self, linear_factor: float, angular_factor: float) -> None:
        linear = linear_factor * self._linear_speed_var.get()
        angular = angular_factor * self._angular_speed_var.get()
        self._node.command(linear, angular, publish_now=True)
        self._status.set(self._status_text(linear, angular))

    def _move_knob(self, x: float, y: float) -> None:
        kr = self._joy_knob_radius
        self._canvas.coords(self._knob_id, x - kr, y - kr, x + kr, y + kr)

    def _reset_joystick(self, stop_robot: bool) -> None:
        cx, cy = self._joy_center
        self._move_knob(cx, cy)
        if stop_robot:
            self._node.stop()
            self._status.set(self._status_text(0.0, 0.0))

    def _stop_motion(self) -> None:
        self._joy_active = False
        self._active_button_id = None
        self._active_button_factors = None
        self._reset_joystick(stop_robot=True)

    def _on_tuning_changed(self) -> None:
        self._update_tuning_labels()
        if self._joy_active:
            knob = self._canvas.coords(self._knob_id)
            x = (knob[0] + knob[2]) / 2.0
            y = (knob[1] + knob[3]) / 2.0
            self._apply_joystick_position(x, y)
        elif self._active_button_factors is not None:
            linear_factor, angular_factor = self._active_button_factors
            self._apply_button_command(linear_factor, angular_factor)

    @staticmethod
    def _status_text(linear: float, angular: float) -> str:
        return f"cmd_vel  linear.x={linear:.2f} m/s  angular.z={angular:.2f} rad/s"

    def _on_close(self) -> None:
        self._node.stop()
        self._root.quit()
        self._root.destroy()

    def _update_tuning_labels(self) -> None:
        self._linear_speed_text.set(f"{self._linear_speed_var.get():.2f} m/s")
        self._angular_speed_text.set(f"{self._angular_speed_var.get():.2f} rad/s")
        self._linear_deadband_text.set(f"{self._linear_deadband_var.get():.2f}")
        self._angular_deadband_text.set(f"{self._angular_deadband_var.get():.2f}")

    @staticmethod
    def _apply_axis_deadband(value: float, deadband: float) -> float:
        magnitude = abs(value)
        if magnitude <= deadband:
            return 0.0
        if deadband >= 1.0:
            return 0.0
        scaled = (magnitude - deadband) / (1.0 - deadband)
        return math.copysign(scaled, value)

    def _on_focus_out(self, _evt) -> None:
        self._root.after_idle(self._stop_if_focus_left_window)

    def _stop_if_focus_left_window(self) -> None:
        focused_widget = self._root.focus_get()
        current = focused_widget
        while current is not None:
            if current is self._root:
                return
            current = current.master
        self._stop_motion()


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
