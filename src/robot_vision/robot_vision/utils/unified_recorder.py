#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: unified_recorder.py
# AUTHOR: Geoffrey Hinton
# DESCRIPTION:
# [Hinton's Unified Multi-Camera Data Recording Architecture]
# 1. 단일 ROS 2 노드에서 여러 카메라 토픽을 동시에 구독하여 데이터 수집
#    -> 모든 영상 녹화의 시작과 종료를 동기화하여 데이터셋의 정합성 보장
# 2. 특정 토픽('camera3')에 대해서만 BEV(Bird's-Eye View) 변환 로직을 적용
#    -> BEV 데이터와 원본 영상 데이터를 한 번의 주행으로 수집 가능
# 3. 확장성을 극대화한 파라미터 설계: 카메라 토픽, 출력 경로 등을 리스트로 관리
#    -> 코드 수정 없이 launch 파일에서 파라미터 변경만으로 카메라 추가/삭제 가능
# 4. 안정적인 종료(Graceful Shutdown) 보장: 노드 종료 시 모든 비디오 파일을 안전하게 release
# 5. CvBridge와 Numpy 직접 디코딩 방식을 혼용하여 기존 로직과의 호환성 및 안정성 확보

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import os
import traceback
from datetime import datetime
from functools import partial

from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge

class UnifiedRecorderNode(Node):
    """
    여러 카메라 토픽을 구독하여 동기화된 비디오 파일로 녹화하는 통합 노드입니다.
    특정 카메라에 대해서는 BEV 변환을 적용합니다.
    """
    def __init__(self):
        super().__init__('unified_recorder_node')
        self.get_logger().info("--- Hinton's Unified Multi-Camera Recorder ---")

        # === 1. 파라미터 선언 및 가져오기 ===
        self.declare_parameter('camera_topics', 
            ['/camera1/image_raw/compressed', 
             '/camera2/image_raw/compressed', 
             '/camera/color/image_raw/compressed',
             '/camera3/image_raw/compressed'])
        self.declare_parameter('output_dir', '~/ros2_recordings/unified')
        self.declare_parameter('output_filenames', 
            ['camera1.mp4', 'camera2.mp4', 'realsense.mp4', 'bev_output.mp4'])
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('bev_param_file', './bev_params.npz')
        self.declare_parameter('bev_target_topic', '/camera3/image_raw/compressed')

        self.camera_topics = self.get_parameter('camera_topics').get_parameter_value().string_array_value
        output_dir_str = self.get_parameter('output_dir').get_parameter_value().string_value
        self.output_filenames = self.get_parameter('output_filenames').get_parameter_value().string_array_value
        self.fps = self.get_parameter('fps').get_parameter_value().double_value
        bev_param_file = self.get_parameter('bev_param_file').get_parameter_value().string_value
        self.bev_target_topic = self.get_parameter('bev_target_topic').get_parameter_value().string_value

        self.output_dir = os.path.expanduser(output_dir_str)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.get_logger().info(f"Created output directory: {self.output_dir}")

        if len(self.camera_topics) != len(self.output_filenames):
            self.get_logger().error("FATAL: The number of camera_topics must match the number of output_filenames. Shutting down.")
            rclpy.shutdown()
            return
            
        self.get_logger().info(f"Subscribing to topics: {self.camera_topics}")
        self.get_logger().info(f"Outputting to files: {self.output_filenames}")
        self.get_logger().info(f"Video FPS set to: {self.fps}")

        # === 2. BEV 변환 파라미터 로드 (camera3용) ===
        self.M_bev, self.bev_w, self.bev_h = None, None, None
        if self.bev_target_topic in self.camera_topics:
            try:
                self.get_logger().info(f"Loading BEV parameters from: {bev_param_file}")
                bev_params = np.load(bev_param_file)
                self.src_points = bev_params['src_points']
                self.dst_points = bev_params['dst_points']
                self.bev_h = int(bev_params['warp_h'])
                self.bev_w = int(bev_params['warp_w'])
                self.M_bev = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
                self.get_logger().info("✅ BEV transformation matrix calculated successfully.")
            except Exception as e:
                self.get_logger().error(f"FATAL: Failed to load BEV parameters: {e}. Shutting down.")
                rclpy.shutdown()
                return
        
        # === 3. 변수 초기화 ===
        self.bridge = CvBridge()
        self.video_writers = {}  # 토픽 이름을 key로 사용하여 VideoWriter 객체 저장
        self.is_recording_started = False
        self.frame_counters = {topic: 0 for topic in self.camera_topics}

        # === 4. 각 토픽에 대한 구독자 생성 ===
        qos_profile_sensor_data = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10 # 여러 스트림을 다루므로 약간의 버퍼를 둠
        )

        for topic in self.camera_topics:
            # partial을 사용하여 콜백 함수에 토픽 이름을 인자로 넘겨줌
            callback_with_topic = partial(self.image_callback, topic_name=topic)
            self.create_subscription(
                CompressedImage,
                topic,
                callback_with_topic,
                qos_profile_sensor_data
            )
        
        self.get_logger().info("✅ Unified recorder node initialized successfully. Waiting for images...")

    def image_callback(self, msg, topic_name):
        """
        모든 카메라 토픽이 공유하는 통합 콜백 함수.
        토픽 이름(topic_name)을 기반으로 적절한 처리를 수행합니다.
        """
        # 첫 프레임 수신 시, 모든 비디오 라이터 초기화 및 녹화 시작
        if not self.is_recording_started:
            self.initialize_all_video_writers(msg, topic_name)
            if not self.is_recording_started: # 초기화 실패 시
                return

        # 해당 토픽의 비디오 라이터가 준비되었는지 확인
        writer = self.video_writers.get(topic_name)
        if writer is None or not writer.isOpened():
            self.get_logger().warn(f"Video writer for '{topic_name}' is not ready. Skipping frame.", throttle_duration_sec=5)
            return

        try:
            # === 프레임 디코딩 및 BEV 변환 (필요시) ===
            processed_image = None
            if topic_name == self.bev_target_topic and self.M_bev is not None:
                # BEV 변환 로직 (기존 bev_recorder.py와 동일)
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if cv_color_image is not None:
                    processed_image = cv2.warpPerspective(
                        cv_color_image, self.M_bev, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR
                    )
            else:
                # 일반 카메라 디코딩 로직 (CvBridge 사용)
                processed_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

            # === 프레임 쓰기 ===
            if processed_image is not None:
                writer.write(processed_image)
                self.frame_counters[topic_name] += 1
            else:
                self.get_logger().warn(f"Failed to decode or process image from {topic_name}", throttle_duration_sec=5)

        except Exception:
            self.get_logger().error(f"Error processing frame from {topic_name}:\n{traceback.format_exc()}")

    def initialize_all_video_writers(self, initial_msg, initial_topic):
        """
        첫 이미지 메시지를 기반으로 모든 VideoWriter 객체를 초기화하고 녹화를 시작합니다.
        이 함수는 단 한 번만 실행됩니다.
        """
        self.get_logger().info("First image received. Initializing all video writers for synchronized recording...")
        
        # 타임스탬프를 모든 파일명에 공통으로 적용하여 세트임을 명시
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        try:
            for topic, filename_template in zip(self.camera_topics, self.output_filenames):
                # 파일명 생성 (예: camera1_20250910_153000.mp4)
                base, ext = os.path.splitext(filename_template)
                filename = f"{base}_{timestamp}{ext}"
                output_path = os.path.join(self.output_dir, filename)

                width, height = -1, -1

                # BEV 영상은 미리 정의된 크기 사용
                if topic == self.bev_target_topic and self.bev_w is not None:
                    width, height = self.bev_w, self.bev_h
                else:
                    # 다른 영상들은 첫 프레임의 크기를 동적으로 파악
                    temp_image = self.bridge.compressed_imgmsg_to_cv2(initial_msg, "bgr8")
                    h, w, _ = temp_image.shape
                    width, height = w, h
                
                if width > 0 and height > 0:
                    writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
                    if not writer.isOpened():
                        raise IOError(f"Cannot open video writer for {output_path}")
                    self.video_writers[topic] = writer
                    self.get_logger().info(f"✅ Initialized recorder for '{topic}' -> '{output_path}'")
                else:
                    raise ValueError(f"Invalid image dimensions for topic {topic}")

            self.is_recording_started = True
            self.get_logger().info("🚀 All recorders are active. Synchronized recording has started!")

        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize video writers: {e}")
            self.get_logger().error("Please check image topics, file permissions, and codec support.")
            # 실패 시, 부분적으로 생성된 writer들을 정리
            for writer in self.video_writers.values():
                if writer.isOpened():
                    writer.release()
            self.video_writers = {}
            rclpy.shutdown()

    def destroy_node(self):
        """
        노드 종료 시 모든 비디오 파일을 안전하게 닫습니다.
        """
        self.get_logger().info("Shutting down node and finalizing videos...")
        for topic, writer in self.video_writers.items():
            if writer and writer.isOpened():
                output_path = "N/A"
                # This is a simplification; in a real scenario, you'd store the full path.
                for filename in self.output_filenames:
                    if os.path.splitext(filename)[0] in topic:
                         output_path = f"{self.output_dir}/{filename}"
                         break
                
                writer.release()
                self.get_logger().info(f"✅ Video for '{topic}' saved successfully. Total frames: {self.frame_counters[topic]}")
        self.get_logger().info("All video files have been saved.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = UnifiedRecorderNode()
    if rclpy.ok(): # 노드 초기화 성공 여부 확인
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Keyboard interrupt detected.")
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

if __name__ == '__main__':
    main()