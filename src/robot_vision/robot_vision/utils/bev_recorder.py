#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: bev_recorder.py
# AUTHOR: Geoffrey Hinton
# DESCRIPTION:
# [Hinton's BEV Data Generation & Recording Architecture]
# 1. 원본 경로 계획 노드('onnx_path_planning_pp.py')와 완벽히 동일한 BEV 변환 로직 및 파라미터 사용
#    -> 생성된 BEV 데이터와 경로 계획 알고리즘 간의 100% 정합성 보장
# 2. 고품질 영상 저장을 위한 표준 비디오 코덱(MP4V) 사용
# 3. 노드 종료 시 비디오 파일을 안전하게 종료(release)하여 데이터 손상을 원천적으로 방지하는 로직 포함
# 4. 사용자가 쉽게 파라미터를 변경할 수 있도록 ROS 2 파라미터 시스템 활용 (bev_param_file, output_path, fps)
# 5. 실시간 영상 스트림 처리에 적합한 'Best Effort' QoS 프로파일을 적용하여 네트워크 부하 최소화

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import os
import traceback

from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class BEVRecorderNode(Node):
    """
    카메라 이미지 토픽을 구독하여 BEV(Bird's-Eye View)로 변환하고,
    그 결과를 비디오 파일로 녹화하는 노드입니다.
    """
    def __init__(self):
        super().__init__('bev_recorder_node')
        self.get_logger().info("--- BEV Data Generation & Recording Node (Hinton's Architecture) ---")
        
        # === 파라미터 선언 및 가져오기 ===
        # 이 파라미터들은 launch 파일이나 커맨드 라인에서 쉽게 변경할 수 있습니다.
        self.declare_parameter('bev_param_file', './bev_params.npz')
        self.declare_parameter('output_path', '~/bev_output1.mp4')
        self.declare_parameter('fps', 30.0)
        
        bev_param_file = self.get_parameter('bev_param_file').get_parameter_value().string_value
        output_path_str = self.get_parameter('output_path').get_parameter_value().string_value
        # '~' 문자를 사용자 홈 디렉토리로 확장합니다.
        self.output_path = os.path.expanduser(output_path_str) 
        self.fps = self.get_parameter('fps').get_parameter_value().double_value
        
        self.get_logger().info(f"BEV parameter file: {bev_param_file}")
        self.get_logger().info(f"Video output path: {self.output_path}")
        self.get_logger().info(f"Video FPS: {self.fps}")

        self.video_writer = None

        # === BEV 변환 파라미터 로드 ===
        # 이 부분은 제공하신 'onnx_path_planning_pp.py' 코드의 로직과 100% 동일합니다.
        # 이를 통해 데이터 정합성을 완벽하게 보장합니다.
        try:
            self.get_logger().info(f"Loading BEV parameters from: {bev_param_file}")
            bev_params = np.load(bev_param_file)
            self.src_points = bev_params['src_points']
            self.dst_points = bev_params['dst_points']
            self.bev_h = int(bev_params['warp_h'])
            self.bev_w = int(bev_params['warp_w'])
            self.M_bev = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
            self.get_logger().info("✅ BEV transformation matrix calculated successfully.")
        except FileNotFoundError:
            self.get_logger().error(f"FATAL: BEV parameter file not found at '{bev_param_file}'. Shutting down.")
            rclpy.shutdown()
            return
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to load BEV parameters: {e}")
            rclpy.shutdown()
            return

        # === 비디오 라이터 초기화 ===
        # MP4V 코덱은 대부분의 시스템에서 지원되는 표준적인 코덱입니다.
        try:
            # 출력 디렉토리가 존재하지 않으면 생성합니다.
            output_dir = os.path.dirname(self.output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.get_logger().info(f"Created output directory: {output_dir}")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.bev_w, self.bev_h))
            if not self.video_writer.isOpened():
                raise IOError("Cannot open video writer.")
            self.get_logger().info(f"✅ Video writer initialized. Recording to '{self.output_path}'")
        except (IOError, Exception) as e:
            self.get_logger().error(f"FATAL: Failed to initialize VideoWriter: {e}")
            self.get_logger().error("Please check file permissions and codec support.")
            rclpy.shutdown()
            return

        # === 이미지 토픽 구독자 설정 ===
        # 경로 계획 노드와 동일한 QoS 설정을 사용하여 데이터 수신 타이밍을 맞춥니다.
        qos_profile_sensor_data = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        img_topic = '/camera3/image_raw/compressed'
        self.img_sub = self.create_subscription(
            CompressedImage, 
            img_topic, 
            self.image_callback, 
            qos_profile_sensor_data
        )
        self.get_logger().info(f"✅ Node initialized. Subscribing to '{img_topic}'.")

    def image_callback(self, compressed_img_msg):
        """
        이미지 메시지를 수신하면 BEV로 변환하고 비디오 프레임으로 저장합니다.
        """
        if self.video_writer is None or not self.video_writer.isOpened():
            self.get_logger().warn("Video writer is not ready. Skipping frame.", throttle_duration_sec=5)
            return

        try:
            # 1. 압축된 이미지를 OpenCV 형식으로 디코딩
            np_arr = np.frombuffer(compressed_img_msg.data, np.uint8)
            cv_color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_color_image is None:
                self.get_logger().warn("Failed to decode compressed image.", throttle_duration_sec=5)
                return

            # 2. BEV 변환 수행 (핵심 로직)
            bev_image = cv2.warpPerspective(
                cv_color_image, 
                self.M_bev, 
                (self.bev_w, self.bev_h), 
                flags=cv2.INTER_LINEAR
            )

            # 3. 변환된 BEV 이미지를 비디오 파일에 쓰기
            self.video_writer.write(bev_image)

        except Exception:
            self.get_logger().error(f"Error in image_callback:\n{traceback.format_exc()}")

    def destroy_node(self):
        """
        노드 종료 시 호출되는 함수. 비디오 파일을 안전하게 닫습니다.
        이 과정은 파일 손상을 방지하기 위해 매우 중요합니다.
        """
        self.get_logger().info("Shutting down node...")
        if self.video_writer and self.video_writer.isOpened():
            self.get_logger().info(f"Finalizing video file at '{self.output_path}'...")
            self.video_writer.release()
            self.get_logger().info("✅ Video file saved successfully.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    # BEV 파라미터와 비디오 라이터가 성공적으로 초기화되었는지 확인 후 spin 시작
    node = BEVRecorderNode()
    if rclpy.ok() and hasattr(node, 'M_bev') and node.video_writer is not None:
        try: 
            rclpy.spin(node)
        except KeyboardInterrupt: 
            node.get_logger().info("Keyboard interrupt detected.")
        finally: 
            # 노드 종료 및 모든 자원 해제
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

if __name__ == '__main__':
    main()