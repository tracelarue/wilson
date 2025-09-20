import cv2
import numpy as np
import ArducamDepthCamera as ac
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# MAX_DISTANCE value modifiable  is 2000 or 4000
MAX_DISTANCE=2000

class UserRect:
    def __init__(self) -> None:
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0

    @property
    def rect(self):
        return (
            self.start_x,
            self.start_y,
            self.end_x - self.start_x,
            self.end_y - self.start_y,
        )

    @property
    def slice(self):
        return (slice(self.start_y, self.end_y), slice(self.start_x, self.end_x))

    @property
    def empty(self):
        return self.start_x == self.end_x and self.start_y == self.end_y


# Increase from default 30 to higher value
confidence_value = 30  # Experiment with values between 50-100
selectRect, followRect = UserRect(), UserRect()


def getPreviewRGB(preview: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    preview = np.nan_to_num(preview)
    preview[confidence < confidence_value] = (0, 0, 0)
    return preview


def on_mouse(event, x, y, flags, param):
    global selectRect, followRect

    if event == cv2.EVENT_LBUTTONDOWN:
        pass

    elif event == cv2.EVENT_LBUTTONUP:
        selectRect.start_x = x - 4
        selectRect.start_y = y - 4
        selectRect.end_x = x + 4
        selectRect.end_y = y + 4
    else:
        followRect.start_x = x - 4
        followRect.start_y = y - 4
        followRect.end_x = x + 4
        followRect.end_y = y + 4


def on_confidence_changed(value):
    global confidence_value
    confidence_value = value


def usage(argv0):
    print("Usage: python " + argv0 + " [options]")
    print("Available options are:")
    print(" -d        Choose the video to use")


class DepthFieldNode(Node):
    def __init__(self):
        super().__init__('depth_field_node')
        self.publisher_ = self.create_publisher(Image, 'depth_field', 10)
        self.bridge = CvBridge()
        self.cam = ac.ArducamCamera()
        self.cfg_path = None
        self.timer = None
        self.r = None
        self.info = None
        self.preview_enabled = True  # Set to False to disable OpenCV preview
        self.init_camera()

    def init_camera(self):
        if self.cfg_path is not None:
            ret = self.cam.openWithFile(self.cfg_path, 0)
        else:
            ret = self.cam.open(ac.Connection.CSI, 0)
        if ret != 0:
            self.get_logger().error(f"Failed to open camera. Error code: {ret}")
            return
        ret = self.cam.start(ac.FrameType.DEPTH)
        if ret != 0:
            self.get_logger().error(f"Failed to start camera. Error code: {ret}")
            self.cam.close()
            return
        self.cam.setControl(ac.Control.RANGE, MAX_DISTANCE)
        self.r = self.cam.getControl(ac.Control.RANGE)
        self.info = self.cam.getCameraInfo()
        self.get_logger().info(f"Camera resolution: {self.info.width}x{self.info.height}")
        if self.preview_enabled:
            cv2.namedWindow("preview", cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback("preview", on_mouse)
            if self.info.device_type == ac.DeviceType.VGA:
                cv2.createTrackbar(
                    "confidence", "preview", confidence_value, 255, on_confidence_changed
                )
        self.timer = self.create_timer(0.03, self.timer_callback)  # ~30Hz

    def timer_callback(self):
        frame = self.cam.requestFrame(2000)
        if frame is not None and isinstance(frame, ac.DepthData):
            depth_buf = frame.depth_data
            confidence_buf = frame.confidence_data
            # Correct depth values by subtracting 50mm, clamp to 0
            depth_buf = np.maximum(depth_buf - 45, 0)
            # Publish depth field as Image
            depth_img_msg = self.bridge.cv2_to_imgmsg(depth_buf.astype(np.float32), encoding='32FC1')
            self.publisher_.publish(depth_img_msg)
            if self.preview_enabled:
                result_image = (depth_buf * (255.0 / self.r)).astype(np.uint8)
                result_image = cv2.applyColorMap(result_image, cv2.COLORMAP_RAINBOW)
                result_image = getPreviewRGB(result_image, confidence_buf)
                cv2.normalize(confidence_buf, confidence_buf, 1, 0, cv2.NORM_MINMAX)
                cv2.imshow("preview_confidence", confidence_buf)
                cv2.rectangle(result_image, followRect.rect, (255,255,255), 1)
                if not selectRect.empty:
                    cv2.rectangle(result_image, selectRect.rect, (0,0,0), 2)
                    self.get_logger().info(f"select Rect distance: {np.mean(depth_buf[selectRect.slice])}")
                cv2.imshow("preview", result_image)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    rclpy.shutdown()
            self.cam.releaseFrame(frame)

    def destroy_node(self):
        self.cam.stop()
        self.cam.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DepthFieldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()