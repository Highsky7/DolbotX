#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

class HsvPickerNode(Node):
    """
    ROS2 토픽을 구독하여 HSV 값을 추출하는 노드 클래스
    """
    def __init__(self):
        super().__init__('hsv_picker_node')
        self.get_logger().info('HSV Picker ROS2 Node has been started.')

        # ROS2 토픽 구독자 설정
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',  # 사용자 요청 토픽
            self.image_callback,
            10)
        
        self.cv_image = None
        self.window_name = 'HSV Picker - Click on the track'
        cv2.namedWindow(self.window_name)
        # 마우스 콜백에 self(노드 인스턴스)를 전달하여 콜백 함수 내에서 이미지에 접근 가능하게 함
        cv2.setMouseCallback(self.window_name, self.get_hsv_value, self)

        self.get_logger().info("ROS2 토픽에서 이미지를 기다리는 중...")
        self.get_logger().info("Track 위를 마우스로 클릭하여 HSV 값을 확인하세요. 종료하려면 'q'를 누르세요.")

    def get_hsv_value(self, event, x, y, flags, param):
        """
        마우스 클릭 이벤트 콜백 함수. 클릭된 픽셀의 BGR, HSV 값을 출력합니다.
        """
        # param으로 전달된 self(노드 인스턴스)를 사용
        node_instance = param
        if event == cv2.EVENT_LBUTTONDOWN:
            if node_instance.cv_image is not None:
                # BGR 이미지를 HSV로 변환
                hsv_pixel = cv2.cvtColor(np.uint8([[node_instance.cv_image[y, x]]]), cv2.COLOR_BGR2HSV)
                # 로그 출력
                self.get_logger().info(f"Clicked Pixel BGR: {node_instance.cv_image[y, x]}, HSV: {hsv_pixel[0][0]}")
            else:
                self.get_logger().warn("아직 이미지가 수신되지 않았습니다.")

    def image_callback(self, msg):
        """
        이미지 토픽을 수신하면 호출되는 콜백 함수.
        """
        try:
            # CompressedImage 메시지를 OpenCV 이미지로 디코딩
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f'Failed to decode image: {e}')
            return

    def run(self):
        """
        메인 루프를 실행하여 이미지 출력 및 사용자 입력을 처리합니다.
        """
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)  # ROS2 콜백 처리
            
            if self.cv_image is not None:
                cv2.imshow(self.window_name, self.cv_image)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("'q' 키 입력 -> 종료")
                break
        
        # 종료 전 리소스 정리
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    hsv_picker_node = HsvPickerNode()
    hsv_picker_node.run()

if __name__ == '__main__':
    main()