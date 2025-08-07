#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
유틸리티 스크립트: BEV(Birds-Eye View) 파라미터 설정 (ROS2 버전)
----------------------------------------------------------------
ROS2 토픽 (/camera/color/image_raw/compressed)을 구독하여
실시간 영상에서 BEV 파라미터를 설정합니다.

- 사용자가 BEV 설정에 필요한 4개의 점을 직접 선택
- 오른쪽 점(오른쪽 아래, 오른쪽 위)의 y좌표는 왼쪽 점의 y좌표에 자동 정렬
- 선택된 4개 원본 좌표(src_points)를 npz 및 txt 파일로 저장

설정 후 's' 키를 누르면 BEV 파라미터가 저장됩니다.
"""

import cv2
import numpy as np
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

class BevParamSetterNode(Node):
    def __init__(self):
        super().__init__('bev_param_setter_node')
        self.get_logger().info('BEV Parameter Setter ROS2 Node has been started.')

        # BEV 파라미터
        self.warp_w = 640
        self.warp_h = 640
        self.out_npz_file = 'bev_params.npz'
        self.out_txt_file = 'selected_bev_src_points.txt'

        # ROS2 토픽 구독자
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',  # 사용자 요청 토픽
            self.image_callback,
            10
        )

        # 상태 변수 (클래스 인스턴스 변수로 관리)
        self.cv_image = None
        self.src_points = []
        self.max_points = 4

        # OpenCV 윈도우 및 마우스 콜백 설정
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Original", self.mouse_callback, self)

        self.print_instructions()

    def print_instructions(self):
        self.get_logger().info("\n[INSTRUCTIONS]")
        self.get_logger().info("ROS2 토픽에서 이미지를 기다리는 중...")
        self.get_logger().info("왼쪽 마우스 클릭으로 원본 영상에서 4개의 점을 선택하세요.")
        self.get_logger().info("클릭 순서: 1.왼쪽 아래 -> 2.오른쪽 아래 -> 3.왼쪽 위 -> 4.오른쪽 위")
        self.get_logger().info("✨ 오른쪽 점들은 왼쪽 점들의 y좌표에 자동으로 맞춰집니다.")
        self.get_logger().info("'r' 키: 리셋 (선택한 모든 점 초기화)")
        self.get_logger().info("'s' 키: BEV 파라미터 저장 후 종료")
        self.get_logger().info("'q' 키: 종료 (저장 안 함)\n")

    def image_callback(self, msg):
        """이미지 토픽을 수신하여 self.cv_image를 업데이트합니다."""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f'이미지 디코딩 실패: {e}')

    def mouse_callback(self, event, x, y, flags, param):
        """마우스 클릭 이벤트를 처리합니다."""
        node_instance = param  # self
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(node_instance.src_points) < node_instance.max_points:
                point_order = ["왼쪽 아래", "오른쪽 아래", "왼쪽 위", "오른쪽 위"]
                current_point_index = len(node_instance.src_points)
                final_point = (x, y)

                # 2번째, 4번째 점의 y좌표를 자동 정렬
                if current_point_index == 1:   # 오른쪽 아래
                    if len(node_instance.src_points) > 0:
                        y_bottom = node_instance.src_points[0][1]
                        final_point = (x, y_bottom)
                elif current_point_index == 3: # 오른쪽 위
                    if len(node_instance.src_points) > 2:
                        y_top = node_instance.src_points[2][1]
                        final_point = (x, y_top)

                node_instance.src_points.append(final_point)
                self.get_logger().info(f"[{point_order[current_point_index]}] 점 추가: {final_point} ({len(node_instance.src_points)}/{node_instance.max_points})")

                if len(node_instance.src_points) == node_instance.max_points:
                    self.get_logger().info("4점 모두 선택 완료. 's'로 저장하거나 'r'로 리셋하세요.")
            else:
                self.get_logger().warn("이미 4개의 점이 모두 선택되었습니다. 'r'로 리셋하거나 's'로 저장하세요.")

    def save_params(self):
        """BEV 파라미터를 파일로 저장합니다."""
        if len(self.src_points) < self.max_points:
            self.get_logger().warn("4개의 점을 모두 선택해야 저장이 가능합니다.")
            return False

        self.get_logger().info("'s' 키 입력 -> BEV 파라미터 저장 후 종료")
        
        dst_points_default = np.float32([
            [0, self.warp_h],          # 왼 하단
            [self.warp_w, self.warp_h],# 오른 하단
            [0, 0],                    # 왼 상단
            [self.warp_w, 0]           # 오른 상단
        ])
        
        src_arr = np.float32(self.src_points)
        
        # NPZ 파일 저장
        np.savez(self.out_npz_file,
                 src_points=src_arr,
                 dst_points=dst_points_default,
                 warp_w=self.warp_w,
                 warp_h=self.warp_h)
        self.get_logger().info(f"'{self.out_npz_file}' 파일에 BEV 파라미터 저장 완료.")

        # TXT 파일 저장
        point_labels = ["Left-Bottom", "Right-Bottom", "Left-Top", "Right-Top"]
        try:
            with open(self.out_txt_file, 'w') as f:
                f.write("# Selected BEV Source Points (x, y) for original image\n")
                f.write("# Order: Left-Bottom, Right-Bottom, Left-Top, Right-Top\n")
                for i, point in enumerate(self.src_points):
                    f.write(f"{point[0]}, {point[1]} # {point_labels[i]}\n")
            self.get_logger().info(f"'{self.out_txt_file}' 파일에 선택된 좌표 저장 완료.")
        except Exception as e:
            self.get_logger().error(f"TXT 파일 저장 중 오류 발생: {e}")
            
        return True

    def run(self):
        """메인 루프를 실행하여 ROS2 이벤트 처리 및 화면 갱신, 사용자 입력을 담당합니다."""
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            if self.cv_image is None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.get_logger().info("'q' 키 입력 -> 종료")
                    break
                continue

            disp = self.cv_image.copy()

            # 선택된 점과 라인 그리기
            point_labels = ["1 (L-Bot)", "2 (R-Bot)", "3 (L-Top)", "4 (R-Top)"]
            for i, pt in enumerate(self.src_points):
                cv2.circle(disp, pt, 5, (0, 255, 0), -1)
                label = point_labels[i] if i < len(point_labels) else f"{i+1}"
                cv2.putText(disp, label, (pt[0] + 5, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if len(self.src_points) == 4:
                cv2.polylines(disp, [np.array(self.src_points, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

            cv2.imshow("Original", disp)

            # BEV 변환 및 표시
            bev_result = np.zeros((self.warp_h, self.warp_w, 3), dtype=np.uint8)
            if len(self.src_points) == 4:
                src_np = np.float32(self.src_points)
                dst_points_default = np.float32([
                    [0, self.warp_h], [self.warp_w, self.warp_h],
                    [0, 0], [self.warp_w, 0]
                ])
                M = cv2.getPerspectiveTransform(src_np, dst_points_default)
                bev_result = cv2.warpPerspective(self.cv_image, M, (self.warp_w, self.warp_h))
            cv2.imshow("BEV", bev_result)

            # 키 입력 처리
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                self.get_logger().info("'q' 키 입력 -> 종료 (저장 안 함)")
                break
            elif key == ord('r'):
                self.get_logger().info("'r' 키 입력 -> 4점 좌표 초기화")
                self.src_points = []
            elif key == ord('s'):
                if self.save_params():
                    break  # 저장 성공 시 루프 종료

        # 리소스 정리
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()
        self.get_logger().info("bev_utils_ros2.py 종료.")

def main(args=None):
    rclpy.init(args=args)
    bev_node = BevParamSetterNode()
    bev_node.run()

if __name__ == '__main__':
    main()