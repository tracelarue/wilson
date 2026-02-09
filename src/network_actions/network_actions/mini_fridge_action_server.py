#!/usr/bin/env python3

import time
import urllib.error
import urllib.parse
import urllib.request

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from network_actions.action import MiniFridge


class MiniFridgeActionServer(Node):
    """Action server that controls the mini fridge over network (ESP32)."""

    DEFAULT_WAIT_MS = 5000
    DEFAULT_TIMEOUT_MS = 12000
    DEFAULT_ESP32_HOST = '192.168.1.112'
    DEFAULT_ESP32_PORT = 80
    DEFAULT_REQUEST_PATH = '/mini_fridge'

    def __init__(self):
        super().__init__('mini_fridge_action_server')

        self.declare_parameter('esp32_host', self.DEFAULT_ESP32_HOST)
        self.declare_parameter('esp32_port', self.DEFAULT_ESP32_PORT)
        self.declare_parameter('request_path', self.DEFAULT_REQUEST_PATH)
        self.declare_parameter('wait_ms', self.DEFAULT_WAIT_MS)
        self.declare_parameter('request_timeout_ms', self.DEFAULT_TIMEOUT_MS)

        self.esp32_host = self.get_parameter('esp32_host').get_parameter_value().string_value.strip()
        self.esp32_port = self.get_parameter('esp32_port').get_parameter_value().integer_value
        self.request_path = self.get_parameter('request_path').get_parameter_value().string_value.strip()
        self.default_wait_ms = self.get_parameter('wait_ms').get_parameter_value().integer_value
        self.default_timeout_ms = self.get_parameter('request_timeout_ms').get_parameter_value().integer_value

        if not self.request_path.startswith('/'):
            self.request_path = '/' + self.request_path

        self.callback_group = ReentrantCallbackGroup()
        self.valid_commands = {'open', 'close', 'toggle'}

        self._action_server = ActionServer(
            self,
            MiniFridge,
            'mini_fridge',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        self.get_logger().info(
            f"Mini fridge action server started "
            f"(endpoint=http://{self.esp32_host}:{self.esp32_port}{self.request_path})"
        )

    def goal_callback(self, goal_request):
        self.get_logger().info(
            f"Received mini_fridge goal (command='{goal_request.command}', "
            f"wait={goal_request.waitms}ms, timeout={goal_request.timeoutms}ms)"
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancellation request for mini_fridge')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        result = MiniFridge.Result()

        command = goal_handle.request.command.strip().lower()
        if not command:
            command = 'toggle'

        if command not in self.valid_commands:
            result.success = False
            result.message = f"Invalid command '{command}'. Expected one of: open, close, toggle"
            goal_handle.abort()
            return result

        wait_ms = goal_handle.request.waitms
        timeout_ms = goal_handle.request.timeoutms

        if wait_ms <= 0:
            wait_ms = self.default_wait_ms
        if timeout_ms <= 0:
            timeout_ms = self.default_timeout_ms

        feedback = MiniFridge.Feedback()
        feedback.currentstatus = f"Sending '{command}' command"
        feedback.progresspercentage = 10.0
        goal_handle.publish_feedback(feedback)

        sent_ok, sent_message = self._send_command(command, timeout_ms)

        if not sent_ok:
            result.success = False
            result.message = sent_message
            goal_handle.abort()
            return result

        if goal_handle.is_cancel_requested:
            result.success = False
            result.message = 'mini_fridge action canceled after command dispatch'
            goal_handle.canceled()
            return result

        feedback.currentstatus = 'Command accepted, waiting for door motion'
        feedback.progresspercentage = 50.0
        goal_handle.publish_feedback(feedback)

        if not self._sleep_with_cancel(goal_handle, wait_ms):
            result.success = False
            result.message = 'mini_fridge action canceled during wait'
            goal_handle.canceled()
            return result

        feedback.currentstatus = 'Mini fridge action complete'
        feedback.progresspercentage = 100.0
        goal_handle.publish_feedback(feedback)

        result.success = True
        result.message = sent_message
        goal_handle.succeed()
        return result

    def _send_command(self, command: str, timeout_ms: int):
        timeout_sec = max(0.1, float(timeout_ms) / 1000.0)
        query = urllib.parse.urlencode({'command': command})
        url = f"http://{self.esp32_host}:{self.esp32_port}{self.request_path}?{query}"

        try:
            req = urllib.request.Request(url=url, method='GET')
            with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                status_code = response.getcode()
                body = response.read().decode('utf-8', errors='replace').strip()

            if status_code < 200 or status_code >= 300:
                return False, f"ESP32 returned HTTP {status_code}: {body}"

            if not body:
                body = f"Mini fridge command '{command}' sent"
            return True, body
        except urllib.error.HTTPError as exc:
            return False, f"ESP32 HTTP error: {exc.code} {exc.reason}"
        except urllib.error.URLError as exc:
            return False, f"ESP32 unreachable: {exc.reason}"
        except TimeoutError:
            return False, f"ESP32 request timed out after {timeout_sec:.1f}s"
        except Exception as exc:
            return False, f"Failed to send mini fridge command: {exc}"

    def _sleep_with_cancel(self, goal_handle, wait_ms: int):
        end_time = time.monotonic() + max(0.0, float(wait_ms) / 1000.0)
        while time.monotonic() < end_time:
            if goal_handle.is_cancel_requested:
                return False
            time.sleep(0.1)
        return True

def main(args=None):
    rclpy.init(args=args)

    node = MiniFridgeActionServer()
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
