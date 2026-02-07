#!/usr/bin/env python3

import time

import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Int32

from ir_signal_action.action import IrSignal

try:
    import pigpio
except Exception:  # pragma: no cover - optional dependency on non-RPi systems
    pigpio = None

try:
    import gpiod
except Exception:  # pragma: no cover - optional dependency on non-RPi systems
    gpiod = None


class IrSignalActionServer(Node):
    """Action server to send a 38kHz IR pulse train for mini-fridge triggering."""

    DEFAULT_BURST_MS = 1000
    DEFAULT_WAIT_MS = 5000
    DEFAULT_GPIO = 18
    DEFAULT_FREQ_HZ = 38000
    DEFAULT_TRIGGER_PULSE_COUNT = 5
    MIN_PULSE_ON_MS = 30

    def __init__(self):
        super().__init__('ir_signal_action_server')

        self.declare_parameter('mode', 'robot')
        self.declare_parameter('gpio_backend', 'pigpio')
        self.declare_parameter('gpio_chip', '/dev/gpiochip0')
        self.declare_parameter('gpio_pin', self.DEFAULT_GPIO)
        self.declare_parameter('burst_duration_ms', self.DEFAULT_BURST_MS)
        self.declare_parameter('wait_ms', self.DEFAULT_WAIT_MS)
        self.declare_parameter('carrier_frequency_hz', self.DEFAULT_FREQ_HZ)
        self.declare_parameter('sim_topic', '/ir_signal')
        self.declare_parameter('trigger_pulse_count', self.DEFAULT_TRIGGER_PULSE_COUNT)

        self.mode = self.get_parameter('mode').get_parameter_value().string_value
        self.gpio_backend = self.get_parameter('gpio_backend').get_parameter_value().string_value.lower()
        self.gpio_chip = self.get_parameter('gpio_chip').get_parameter_value().string_value
        self.gpio_pin = self.get_parameter('gpio_pin').get_parameter_value().integer_value
        self.default_burst_ms = self.get_parameter('burst_duration_ms').get_parameter_value().integer_value
        self.default_wait_ms = self.get_parameter('wait_ms').get_parameter_value().integer_value
        self.frequency_hz = self.get_parameter('carrier_frequency_hz').get_parameter_value().integer_value
        self.sim_topic = self.get_parameter('sim_topic').get_parameter_value().string_value
        self.trigger_pulse_count = self.get_parameter('trigger_pulse_count').get_parameter_value().integer_value

        self.callback_group = ReentrantCallbackGroup()

        self.sim_pub = self.create_publisher(Int32, self.sim_topic, 10)
        self._publish_sim_signal(0)

        self.pi = None
        self.gpio_line = None
        self.gpio_chip_handle = None
        if self.mode != 'sim':
            if self.gpio_backend == 'pigpio':
                self._init_pigpio()
            elif self.gpio_backend == 'gpiod':
                self._init_gpiod()
            else:
                self.get_logger().error(
                    f"Unsupported gpio_backend '{self.gpio_backend}'. Use 'pigpio' or 'gpiod'."
                )

        self._action_server = ActionServer(
            self,
            IrSignal,
            'ir_signal_action',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        self.get_logger().info('IR signal action server started')

    def destroy_node(self):
        if self.gpio_line is not None:
            try:
                self.gpio_line.set_value(0)
                self.gpio_line.release()
            except Exception:
                pass
        if self.gpio_chip_handle is not None:
            try:
                self.gpio_chip_handle.close()
            except Exception:
                pass
        if self.pi is not None:
            try:
                self.pi.hardware_PWM(self.gpio_pin, 0, 0)
                self.pi.stop()
            except Exception:
                pass
        super().destroy_node()

    def goal_callback(self, goal_request):
        self.get_logger().info(
            f'Received IR signal goal (burst={goal_request.burstdurationms}ms, '
            f'wait={goal_request.waitms}ms)'
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancellation request for IR signal')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        result = IrSignal.Result()

        burst_ms = goal_handle.request.burstdurationms
        wait_ms = goal_handle.request.waitms

        if burst_ms <= 0:
            burst_ms = self.default_burst_ms
        if wait_ms <= 0:
            wait_ms = self.default_wait_ms
        pulse_count = max(1, int(self.trigger_pulse_count))

        if self.mode != 'sim' and not self._hardware_ready():
            result.success = False
            result.message = f"GPIO backend '{self.gpio_backend}' not available"
            goal_handle.abort()
            return result

        if self.gpio_backend == 'pigpio' and self.frequency_hz <= 0:
            result.success = False
            result.message = 'carrier_frequency_hz must be > 0'
            goal_handle.abort()
            return result

        on_ms, off_ms = self._compute_pulse_timing_ms(burst_ms, pulse_count)

        feedback = IrSignal.Feedback()
        feedback.currentstatus = (
            f'Starting IR pulse train ({pulse_count} pulses, {on_ms}ms on/{off_ms}ms off)'
        )
        feedback.progresspercentage = 0.0
        goal_handle.publish_feedback(feedback)
        self._send_pulse_train(goal_handle, pulse_count, on_ms, off_ms)

        if goal_handle.is_cancel_requested:
            result.success = False
            result.message = 'IR signal action canceled during pulse train'
            goal_handle.canceled()
            return result

        feedback.currentstatus = 'Waiting for fridge to open'
        feedback.progresspercentage = 50.0
        goal_handle.publish_feedback(feedback)

        time.sleep(wait_ms / 1000.0)

        if goal_handle.is_cancel_requested:
            result.success = False
            result.message = 'IR signal action canceled during wait'
            goal_handle.canceled()
            return result

        feedback.currentstatus = 'IR signal action complete'
        feedback.progresspercentage = 100.0
        goal_handle.publish_feedback(feedback)

        result.success = True
        result.message = self._result_message(pulse_count, on_ms, off_ms, burst_ms, wait_ms)
        goal_handle.succeed()
        return result

    def _init_pigpio(self):
        if pigpio is None:
            self.get_logger().error('pigpio is not available. Install python3-pigpio and start pigpiod.')
            return

        self.pi = pigpio.pi()
        if not self.pi.connected:
            self.get_logger().error('Failed to connect to pigpio daemon. Is pigpiod running?')
            self.pi = None
            return
        self.pi.set_mode(self.gpio_pin, pigpio.OUTPUT)
        self.pi.hardware_PWM(self.gpio_pin, 0, 0)

    def _init_gpiod(self):
        if gpiod is None:
            self.get_logger().error('gpiod is not available. Install python3-libgpiod.')
            return

        try:
            self.gpio_chip_handle = gpiod.Chip(self.gpio_chip)
            self.gpio_line = self.gpio_chip_handle.get_line(self.gpio_pin)
            self.gpio_line.request(
                consumer='ir_signal_action',
                type=gpiod.LINE_REQ_DIR_OUT,
                default_vals=[0],
            )
            self.gpio_line.set_value(0)
            self.get_logger().warn(
                'Using gpiod backend: output is unmodulated pulses, not 38kHz carrier PWM.'
            )
        except Exception as exc:
            self.get_logger().error(f'Failed to initialize gpiod on {self.gpio_chip}:{self.gpio_pin}: {exc}')
            self.gpio_line = None
            self.gpio_chip_handle = None

    def _hardware_ready(self) -> bool:
        if self.gpio_backend == 'pigpio':
            return self.pi is not None
        if self.gpio_backend == 'gpiod':
            return self.gpio_line is not None
        return False

    def _start_pwm(self):
        if self.gpio_backend == 'pigpio':
            duty_cycle = 500000  # 50% duty cycle in pigpio scale (0-1,000,000)
            self.pi.hardware_PWM(self.gpio_pin, self.frequency_hz, duty_cycle)
        elif self.gpio_backend == 'gpiod':
            self.gpio_line.set_value(1)

    def _stop_pwm(self):
        if self.gpio_backend == 'pigpio':
            self.pi.hardware_PWM(self.gpio_pin, 0, 0)
        elif self.gpio_backend == 'gpiod':
            self.gpio_line.set_value(0)

    def _compute_pulse_timing_ms(self, total_window_ms: int, pulse_count: int):
        """Split total window into count pulses + gaps for active-low receiver edge detection."""
        # For N pulses we need N "on" slots and N-1 "off" gaps.
        slots = max(1, (2 * pulse_count) - 1)
        base_ms = max(self.MIN_PULSE_ON_MS, int(total_window_ms / slots))
        on_ms = base_ms
        off_ms = base_ms
        return on_ms, off_ms

    def _send_pulse_train(self, goal_handle, pulse_count: int, on_ms: int, off_ms: int):
        for idx in range(pulse_count):
            if goal_handle.is_cancel_requested:
                break
            self._publish_sim_signal(1)
            if self.mode == 'sim':
                time.sleep(on_ms / 1000.0)
            else:
                self._start_pwm()
                time.sleep(on_ms / 1000.0)
                self._stop_pwm()
            self._publish_sim_signal(0)

            # No trailing gap needed after the last pulse.
            if idx < pulse_count - 1:
                time.sleep(off_ms / 1000.0)

    def _result_message(self, pulse_count: int, on_ms: int, off_ms: int, burst_ms: int, wait_ms: int):
        if self.gpio_backend == 'gpiod':
            signal = 'GPIO pulses (unmodulated)'
        else:
            signal = f'IR pulses @ {self.frequency_hz}Hz carrier'
        return (
            f'Sent {pulse_count} {signal} ({on_ms}ms on/{off_ms}ms off) '
            f'over ~{burst_ms}ms and waited {wait_ms}ms'
        )

    def _publish_sim_signal(self, value: int):
        msg = Int32()
        msg.data = int(value)
        self.sim_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = IrSignalActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
