import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

class VideoRecorder(Node):
    """
    Subscribe to /camera/color/image_raw/compressed and record the stream to MP4.
    """
    def __init__(self):
        super().__init__('video_recorder_node')

        # Declare parameters with defaults
        self.declare_parameter('output_path', '~/ros2_recordings')
        self.declare_parameter('filename_prefix', 'recording')
        self.declare_parameter('fps', 30.0)

        # Resolve parameter values
        output_path_str = self.get_parameter('output_path').get_parameter_value().string_value
        self.output_path = os.path.expanduser(output_path_str) # Expand '~' to the home directory
        self.filename_prefix = self.get_parameter('filename_prefix').get_parameter_value().string_value
        self.fps = self.get_parameter('fps').get_parameter_value().double_value

        # Create the output directory if needed
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            self.get_logger().info(f"Created directory: {self.output_path}")

        # Initialize CvBridge to convert CompressedImage messages
        self.bridge = CvBridge()

        # Subscribe to /camera/color/image_raw/compressed
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.video_writer = None
        self.is_recording = False
        self.image_count = 0

        self.get_logger().info('Video Recorder Node has been started.')
        self.get_logger().info(f'Recordings will be saved to: {self.output_path}')
        self.get_logger().info(f'Waiting for images on topic /camera/color/image_raw/compressed...')

    def image_callback(self, msg):
        """
        Callback invoked whenever an image message arrives.
        """
        if not self.is_recording:
            # Start recording when the first frame arrives
            try:
                # Convert the compressed message into an OpenCV image
                cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                height, width, _ = cv_image.shape

                # Initialize the VideoWriter
                self.initialize_video_writer(width, height)
                self.is_recording = True
                self.get_logger().info(f"First image received. Starting recording to {self.output_filepath}")

            except Exception as e:
                self.get_logger().error(f"Failed to convert image: {e}")
                return
        
        # Append frames while recording
        if self.is_recording and self.video_writer is not None:
            try:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                self.video_writer.write(cv_image)
                self.image_count += 1
                if self.image_count % int(self.fps) == 0:
                     self.get_logger().info(f"Recorded {self.image_count} frames...")
            except Exception as e:
                self.get_logger().error(f"Failed to write frame: {e}")

    def initialize_video_writer(self, width, height):
        """
        Initialize the OpenCV VideoWriter.
        """
        # Include a timestamp in the filename to avoid collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.filename_prefix}_{timestamp}.mp4"
        self.output_filepath = os.path.join(self.output_path, filename)
        
        # Configure the codec (mp4v suits .mp4 containers)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        self.video_writer = cv2.VideoWriter(self.output_filepath, fourcc, self.fps, (width, height))

        if not self.video_writer.isOpened():
            self.get_logger().error("Could not open video writer.")
            self.video_writer = None


    def destroy_node(self):
        """
        Clean up resources when the node shuts down.
        """
        if self.video_writer is not None and self.video_writer.isOpened():
            self.video_writer.release()
            self.get_logger().info(f"Recording stopped. Video saved to {self.output_filepath}")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    video_recorder = VideoRecorder()
    try:
        rclpy.spin(video_recorder)
    except KeyboardInterrupt:
        # Handle Ctrl+C shutdown
        video_recorder.get_logger().info('Keyboard Interrupt (SIGINT) received. Shutting down...')
    finally:
        # Tear down the node and release resources
        video_recorder.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
