import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

class VideoRecorder(Node):
    """
    /camera1/image_raw/compressed 토픽을 구독하여 MP4 영상 파일로 저장하는 노드입니다.
    """
    def __init__(self):
        super().__init__('video_recorder_node')

        # 파라미터 선언 및 초기화
        self.declare_parameter('output_path', '~/ros2_recordings')
        self.declare_parameter('filename_prefix', 'recording')
        self.declare_parameter('fps', 30.0)

        # 파라미터 값 가져오기
        output_path_str = self.get_parameter('output_path').get_parameter_value().string_value
        self.output_path = os.path.expanduser(output_path_str) # '~'를 홈 디렉토리로 확장
        self.filename_prefix = self.get_parameter('filename_prefix').get_parameter_value().string_value
        self.fps = self.get_parameter('fps').get_parameter_value().double_value

        # 저장 경로가 없으면 생성
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            self.get_logger().info(f"Created directory: {self.output_path}")

        # CompressedImage 메시지를 OpenCV 이미지로 변환하기 위한 CvBridge 초기화
        self.bridge = CvBridge()

        # /camera1/image_raw/compressed 토픽 구독자 설정
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera1/image_raw/compressed',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.video_writer = None
        self.is_recording = False
        self.image_count = 0

        self.get_logger().info('Video Recorder Node has been started.')
        self.get_logger().info(f'Recordings will be saved to: {self.output_path}')
        self.get_logger().info(f'Waiting for images on topic /camera1/image_raw/compressed...')

    def image_callback(self, msg):
        """
        이미지 메시지를 수신할 때마다 호출되는 콜백 함수입니다.
        """
        if not self.is_recording:
            # 첫 번째 이미지를 받으면 녹화 시작
            try:
                # CompressedImage 메시지를 OpenCV 이미지(numpy.ndarray)로 변환
                cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                height, width, _ = cv_image.shape

                # VideoWriter 초기화
                self.initialize_video_writer(width, height)
                self.is_recording = True
                self.get_logger().info(f"First image received. Starting recording to {self.output_filepath}")

            except Exception as e:
                self.get_logger().error(f"Failed to convert image: {e}")
                return
        
        # 녹화 중일 때 프레임 쓰기
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
        OpenCV VideoWriter 객체를 초기화합니다.
        """
        # 파일명에 현재 시간을 포함하여 중복 방지
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.filename_prefix}_{timestamp}.mp4"
        self.output_filepath = os.path.join(self.output_path, filename)
        
        # 사용할 코덱 설정 (mp4v는 .mp4 파일 형식에 적합)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        self.video_writer = cv2.VideoWriter(self.output_filepath, fourcc, self.fps, (width, height))

        if not self.video_writer.isOpened():
            self.get_logger().error("Could not open video writer.")
            self.video_writer = None


    def destroy_node(self):
        """
        노드가 종료될 때 호출되어 리소스를 정리합니다.
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
        # Ctrl+C로 종료 시
        video_recorder.get_logger().info('Keyboard Interrupt (SIGINT) received. Shutting down...')
    finally:
        # 노드 종료 및 리소스 정리
        video_recorder.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()