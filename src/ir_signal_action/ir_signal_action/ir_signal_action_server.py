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


class IrSignalActionServer(Node):
    """Action server to send a 38kHz IR burst for a set duration."""

    DEFAULT_BURST_MS = 1000
    DEFAULT_WAIT_MS = 5000
    DEFAULT_GPIO = 18
    DEFAULT_FREQ_HZ = 38000

    def __init__(self):
        super().__init__('ir_signal_action_server')

        self.declare_parameter('mode', 'robot')
        self.declare_parameter('gpio_pin', self.DEFAULT_GPIO)
        self.declare_parameter('burst_duration_ms', self.DEFAULT_BURST_MS)
        self.declare_parameter('wait_ms', self.DEFAULT_WAIT_MS)
        self.declare_parameter('carrier_frequency_hz', self.DEFAULT_FREQ_HZ)
        self.declare_parameter('sim_topic', '/ir_signal')

        self.mode = self.get_parameter('mode').get_parameter_value().string_value
        self.gpio_pin = self.get_parameter('gpio_pin').get_parameter_value().integer_value
        self.default_burst_ms = self.get_parameter('burst_duration_ms').get_parameter_value().integer_value
        self.default_wait_ms = self.get_parameter('wait_ms').get_parameter_value().integer_value
        self.frequency_hz = self.get_parameter('carrier_frequency_hz').get_parameter_value().integer_value
        self.sim_topic = self.get_parameter('sim_topic').get_parameter_value().string_value

        self.callback_group = ReentrantCallbackGroup()

        self.sim_pub = self.create_publisher(Int32, self.sim_topic, 10)
        self._publish_sim_signal(0)

        self.pi = None
        if self.mode != 'sim':
            if pigpio is None:
                self.get_logger().error('pigpio is not available. Install and start pigpio daemon.')
            else:
                self.pi = pigpio.pi()
                if not self.pi.connected:
                    self.get_logger().error('Failed to connect to pigpio daemon. Is pigpiod running?')
                    self.pi = None
                else:
                    self.pi.set_mode(self.gpio_pin, pigpio.OUTPUT)
                    self.pi.hardware_PWM(self.gpio_pin, 0, 0)

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

        if self.mode != 'sim' and self.pi is None:
            result.success = False
            result.message = 'pigpio not available or daemon not running'
            goal_handle.abort()
            return result

        if self.frequency_hz <= 0:
            result.success = False
            result.message = 'carrier_frequency_hz must be > 0'
            goal_handle.abort()
            return result

        feedback = IrSignal.Feedback()
        feedback.currentstatus = 'Starting IR burst'
        feedback.progresspercentage = 0.0
        goal_handle.publish_feedback(feedback)

        self._publish_sim_signal(1)

        if self.mode == 'sim':
            time.sleep(burst_ms / 1000.0)
        else:
            self._start_pwm()
            time.sleep(burst_ms / 1000.0)
            self._stop_pwm()

        self._publish_sim_signal(0)

        if goal_handle.is_cancel_requested:
            result.success = False
            result.message = 'IR signal action canceled during burst'
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
        result.message = f'Sent {burst_ms}ms IR burst and waited {wait_ms}ms'
        goal_handle.succeed()
        return result

    def _start_pwm(self):
        duty_cycle = 500000  # 50% duty cycle in pigpio scale (0-1,000,000)
        self.pi.hardware_PWM(self.gpio_pin, self.frequency_hz, duty_cycle)

    def _stop_pwm(self):
        self.pi.hardware_PWM(self.gpio_pin, 0, 0)

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
