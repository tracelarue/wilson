import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class DepthPixelReader(Node):
    def __init__(self):
        super().__init__('depth_pixel_reader')
        self.subscription = self.create_subscription(
            Image,
            '/depth_camera/depth/image_raw',
            self.listener_callback,
            1)
        self.bridge = CvBridge()
        self.target_pixel = (320, 240)  # (u, v) coordinates

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        u, v = self.target_pixel
        depth = cv_image[v, u]
        self.get_logger().info(f"Distance = {depth} meters")

def main(args=None):
    rclpy.init(args=args)
    node = DepthPixelReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()