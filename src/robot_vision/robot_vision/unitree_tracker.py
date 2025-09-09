#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: yolo_object_locator.py
# DESCRIPTION:
# Intel RealSense D435i 카메라와 unitree_go2.onnx YOLO 모델을 사용하여
# 객체를 탐지하고, 해당 객체의 카메라 기준 3D 좌표 (x, y)를 계산하여
# ROS 2 토픽 '/target_xy'로 발행하는 노드입니다.
# 제공된 onnx_multi_traffic_supply.py의 거리 계산 방식을 참고하여 작성되었습니다.

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import message_filters
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import Point # x, y 좌표 발행을 위한 메시지 타입
from cv_bridge import CvBridge

# pyrealsense2 라이브러리 임포트
import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

class YoloObjectLocatorNode(Node):
    def __init__(self):
        super().__init__('yolo_object_locator_node')
        self.get_logger().info("--- YOLO Object Locator Node for unitree_go2 ---")

        # 동시성 제어를 위한 Lock 객체
        self.processing_lock = threading.Lock()

        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")
        
        # ONNX 모델 로드
        try:
            # 사용할 ONNX 모델 경로 파라미터 선언
            self.declare_parameter('model_path', './unitree_go2.onnx')
            model_path = self.get_parameter('model_path').get_parameter_value().string_value
            self.model = YOLO(model_path, task='detect')
            self.get_logger().info(f"✅ YOLO ONNX model loaded successfully from: {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            self.destroy_node()
            return

        # 카메라 내부 파라미터(Intrinsics) 저장을 위한 변수
        self.intrinsics = None
        self.camera_info_sub = None

        # 처리 성능을 위한 이미지 크기 파라미터
        self.declare_parameter('proc_width', 640)
        self.declare_parameter('proc_height', 480)
        self.proc_width = self.get_parameter('proc_width').get_parameter_value().integer_value
        self.proc_height = self.get_parameter('proc_height').get_parameter_value().integer_value

        # 결과 발행을 위한 Publisher 생성
        # 토픽명: /target_xy, 메시지 타입: geometry_msgs/msg/Point
        self.target_pub = self.create_publisher(Point, '/target_xy', 10)
        
        # 시각화 이미지 발행을 위한 Publisher
        self.viz_pub = self.create_publisher(CompressedImage, '/yolo_locator/viz/compressed', 10)

        # ⭐️ [수정된 부분 1] 변수 이름을 self.yolo_executor 로 변경
        self.yolo_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='yolo_worker')
        self._is_shutting_down = False

        # RealSense 카메라 정보 토픽 구독
        # CameraInfo 메시지를 한 번만 받아서 내부 파라미터를 설정하고 구독을 해제합니다.
        info_topic = "/camera/color/camera_info"
        self.camera_info_sub = self.create_subscription(CameraInfo, info_topic, self.camera_info_callback, 10)
        self.get_logger().info(f"Waiting for CameraInfo on topic: {info_topic}")

    def camera_info_callback(self, info_msg):
        """
        카메라 정보 콜백 함수. 카메라 내부 파라미터를 한 번만 수신하여 저장합니다.
        """
        if self.intrinsics is not None:
            return
            
        try:
            self.get_logger().info("✅ CameraInfo received.")
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = info_msg.width
            self.intrinsics.height = info_msg.height
            self.intrinsics.ppx = info_msg.k[2]
            self.intrinsics.ppy = info_msg.k[5]
            self.intrinsics.fx = info_msg.k[0]
            self.intrinsics.fy = info_msg.k[4]
            
            if info_msg.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif info_msg.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            else:
                self.intrinsics.model = rs2.distortion.none

            self.intrinsics.coeffs = [i for i in info_msg.d]
            
            # 카메라 정보 수신 완료 후, 이미지 동기화 콜백 등록
            self.initialize_image_sync()

            # 카메라 정보 구독 해제
            if self.camera_info_sub:
                self.destroy_subscription(self.camera_info_sub)
                self.camera_info_sub = None
                self.get_logger().info("CameraInfo subscription destroyed. Starting image synchronization.")
        except Exception as e:
            self.get_logger().error(f"Error in camera_info_callback: {e}")

    def initialize_image_sync(self):
        """
        컬러 이미지와 깊이 이미지를 동기화하여 수신하기 위한 message_filters 설정
        """
        realsense_img_sub = message_filters.Subscriber(self, CompressedImage, '/camera/color/image_raw/compressed')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        
        # ApproximateTimeSynchronizer: 거의 비슷한 타임스탬프를 가진 메시지들을 묶어줍니다.
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [realsense_img_sub, depth_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.image_callback)
        self.get_logger().info("✅ YOLO Object Locator Node initialized successfully.")

    def image_callback(self, compressed_image_msg, depth_msg):
        """
        동기화된 이미지 메시지를 받았을 때 호출되는 메인 콜백 함수
        """
        if self.intrinsics is None or self._is_shutting_down:
            return

        # 이전 프레임 처리가 아직 진행 중이면 현재 프레임은 건너뜁니다.
        if self.processing_lock.acquire(blocking=False):
            try:
                # ⭐️ [수정된 부분 2] 변경된 이름으로 submit 호출
                self.yolo_executor.submit(self._process_images, compressed_image_msg, depth_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to submit image processing task: {e}")
                self.processing_lock.release()
        else:
            self.get_logger().warn("Dropping a frame, previous frame is still processing.", throttle_duration_sec=1)

    def _process_images(self, compressed_image_msg, depth_msg):
        """
        YOLO 추론과 좌표 계산을 수행하는 핵심 로직 (별도 스레드에서 실행됨)
        """
        try:
            # 1. 이미지 디코딩 및 변환
            np_arr = np.frombuffer(compressed_image_msg.data, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            
            if cv_color is None or cv_depth_raw is None:
                self.get_logger().warn("Failed to decode image. Skipping frame.")
                return

            # 시각화용 이미지 복사
            viz_image = cv_color.copy()

            # YOLO 입력용으로 이미지 리사이즈
            resized_color_yolo = cv2.resize(cv_color, (self.proc_width, self.proc_height), interpolation=cv2.INTER_AREA)

            # 2. YOLO 모델 추론 수행
            results = self.model(resized_color_yolo, verbose=False, device=self.device)

            # 3. 가장 신뢰도 높은 객체 찾기
            best_box = None
            max_conf = 0.0
            for box in results[0].boxes:
                if box.conf[0] > max_conf:
                    max_conf = box.conf[0]
                    best_box = box

            # 4. 객체가 검출된 경우 좌표 계산 및 발행
            if best_box is not None:
                # 리사이즈된 이미지 기준의 바운딩 박스 좌표
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                cx_res, cy_res = (x1 + x2) // 2, (y1 + y2) // 2

                # 원본 이미지 크기에 대한 비율 계산
                scale_w = self.intrinsics.width / self.proc_width
                scale_h = self.intrinsics.height / self.proc_height

                # 원본 이미지 기준의 픽셀 좌표
                orig_cx = int(cx_res * scale_w)
                orig_cy = int(cy_res * scale_h)

                # 해당 픽셀의 깊이 값(mm 단위) 가져오기
                if 0 <= orig_cy < self.intrinsics.height and 0 <= orig_cx < self.intrinsics.width:
                    depth_in_mm = cv_depth_raw[orig_cy, orig_cx]
                    
                    # 깊이 값이 유효한 경우 (0보다 큼)
                    if depth_in_mm > 0:
                        # 5. 2D 픽셀 좌표를 3D 공간 좌표로 변환 (Deprojection)
                        # pyrealsense2의 deproject_pixel_to_point 함수 사용
                        # 결과는 미터(m) 단위로 변환되어야 함
                        coords_in_camera = rs2.rs2_deproject_pixel_to_point(
                            self.intrinsics, [orig_cx, orig_cy], depth_in_mm
                        )
                        
                        # mm 단위를 m 단위로 변환
                        x_coord = coords_in_camera[0] / 1000.0
                        y_coord = coords_in_camera[1] / 1000.0
                        z_coord = coords_in_camera[2] / 1000.0

                        # 6. /target_xy 토픽으로 Point 메시지 발행
                        target_point_msg = Point()
                        target_point_msg.x = x_coord
                        target_point_msg.y = y_coord
                        target_point_msg.z = z_coord # z 좌표도 함께 발행
                        self.target_pub.publish(target_point_msg)

                        # 7. 시각화: 바운딩 박스 및 좌표 정보 그리기
                        orig_x1, orig_y1 = int(x1 * scale_w), int(y1 * scale_h)
                        orig_x2, orig_y2 = int(x2 * scale_w), int(y2 * scale_h)
                        
                        label = f"Target: x={x_coord:.2f}, y={y_coord:.2f}, z={z_coord:.2f} m"
                        cv2.rectangle(viz_image, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 0), 2)
                        cv2.putText(viz_image, label, (orig_x1, orig_y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 시각화 이미지 발행
            self.publish_compressed_viz(viz_image)

        except Exception as e:
            self.get_logger().error(f"Error in image processing worker: {e}\n{traceback.format_exc()}")
        finally:
            # 작업 완료 후 Lock 해제
            self.processing_lock.release()

    def publish_compressed_viz(self, cv_image):
        """
        시각화 이미지를 JPEG 형식으로 압축하여 발행하는 함수
        """
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        success, encoded_image = cv2.imencode('.jpg', cv_image)
        if success:
            msg.data = np.array(encoded_image).tobytes()
            self.viz_pub.publish(msg)

    def destroy_node(self):
        """
        노드 종료 시 스레드 풀을 안전하게 종료
        """
        self.get_logger().info("Shutting down the thread pool.")
        self._is_shutting_down = True
        # ⭐️ [수정된 부분 3] 변경된 이름으로 shutdown 호출
        self.yolo_executor.shutdown(wait=True)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloObjectLocatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()