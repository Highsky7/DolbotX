#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
유틸리티 스크립트: BEV(Birds-Eye View) 파라미터 설정
---------------------------------------------------
(수정본) 1280×720 해상도 기준 웹캠, 저장된 영상, 이미지 모두 지원
    - 영상/카메라 소스 지정 (--source)
    - 이미지 파일일 경우 단일 이미지를 이용하여 BEV 파라미터 지정
    - 영상 파일/웹캠인 경우 프레임이 계속 갱신되며 설정 가능

수정 사항:
- 사용자가 BEV 설정에 필요한 4개의 점을 직접 선택
- 오른쪽 점(오른쪽 아래, 오른쪽 위)의 y좌표는 왼쪽 점의 y좌표에 자동 정렬
- 선택된 4개 원본 좌표(src_points)를 txt 파일로 저장 기능은 유지

설정 후 's' 키를 누르면 BEV 파라미터가 npz 파일 및 txt 파일로 저장됩니다.
"""

import cv2
import numpy as np
import argparse
import os

# 전역 변수: 원본 영상에서 선택한 4점 좌표
src_points = []
max_points = 4

def parse_args():
    parser = argparse.ArgumentParser(description="BEV 파라미터 설정 유틸리티")
    parser.add_argument('--source', type=str, default='2',
                        help='영상/카메라 소스. 숫자 (예: 0,1,...)는 웹캠, 파일 경로는 영상 또는 이미지')
    parser.add_argument('--warp-width', type=int, default=640,
                        help='BEV 결과 영상 너비 (기본 640)')
    parser.add_argument('--warp-height', type=int, default=640,
                        help='BEV 결과 영상 높이 (기본 640)')
    parser.add_argument('--out-npz', type=str, default='bev_params.npz',
                        help='저장할 NPZ 파라미터 파일 이름')
    parser.add_argument('--out-txt', type=str, default='selected_bev_src_points.txt',
                        help='저장할 TXT 좌표 파일 이름')
    return parser.parse_args()

def mouse_callback(event, x, y, flags, param):
    """
    마우스 클릭 이벤트를 처리합니다.
    오른쪽 점의 y좌표는 왼쪽 점의 y좌표와 동일하게 자동 정렬됩니다.
    """
    global src_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(src_points) < max_points:
            point_order = ["왼쪽 아래", "오른쪽 아래", "왼쪽 위", "오른쪽 위"]
            current_point_index = len(src_points)
            final_point = (x, y)

            # 2번째, 4번째 점의 y좌표를 자동 정렬
            if current_point_index == 1:   # 오른쪽 아래 점을 찍을 차례
                y_bottom = src_points[0][1] # 첫 번째 점(왼쪽 아래)의 y좌표를 가져옴
                final_point = (x, y_bottom)
            elif current_point_index == 3: # 오른쪽 위 점을 찍을 차례
                y_top = src_points[2][1]    # 세 번째 점(왼쪽 위)의 y좌표를 가져옴
                final_point = (x, y_top)

            src_points.append(final_point)
            print(f"[INFO] {point_order[current_point_index]} 점 추가: {final_point} ({len(src_points)}/{max_points})")

            if len(src_points) == max_points:
                print("[INFO] 4점 모두 선택 완료. 's'로 저장하거나 'r'로 리셋하세요.")
        else:
            print("[WARNING] 이미 4개의 점이 모두 선택되었습니다. 'r' 키로 초기화하거나 's' 키로 저장하세요.")


def main():
    global src_points  # 전역 변수 사용 선언
    args = parse_args()
    source = args.source
    warp_w, warp_h = args.warp_width, args.warp_height

    is_image = False
    cap = None
    static_img = None

    # current_frame_width를 main 함수 내 지역 변수로 사용
    current_frame_width = 0

    if source.isdigit():
        cap_idx = int(source)
        cap = cv2.VideoCapture(cap_idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[ERROR] 카메라({cap_idx})를 열 수 없습니다.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        current_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height_cam = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] 웹캠 ({current_frame_width}x{frame_height_cam})을 통한 실시간 영상 모드")
    else:
        ext = os.path.splitext(source)[1].lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        if ext in image_extensions:
            static_img = cv2.imread(source)
            if static_img is None:
                print(f"[ERROR] 이미지 파일({source})을 열 수 없습니다.")
                return
            is_image = True
            current_frame_width = static_img.shape[1]
            print(f"[INFO] 이미지 파일({source}, {static_img.shape[1]}x{static_img.shape[0]})을 통한 단일 이미지 모드")
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"[ERROR] 비디오 파일({source})을 열 수 없습니다.")
                return
            current_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[INFO] 비디오 파일({source}, {current_frame_width}x{frame_height_vid})을 통한 영상 모드")

    dst_points_default = np.float32([
        [0,       warp_h],    # 왼 하단
        [warp_w,  warp_h],    # 오른 하단
        [0,       0],         # 왼 상단
        [warp_w,  0]          # 오른 상단
    ])

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Original", mouse_callback)

    print("\n[INFO] 왼쪽 마우스 클릭으로 원본 영상에서 4개의 점을 선택하세요.")
    print("       클릭 순서: 1.왼쪽 아래 -> 2.오른쪽 아래 -> 3.왼쪽 위 -> 4.오른쪽 위")
    print("       ✨ 오른쪽 점들은 왼쪽 점들의 y좌표에 자동으로 맞춰집니다.")
    print("       'r' 키: 리셋 (선택한 모든 점 초기화)")
    print("       's' 키: BEV 파라미터 저장 후 종료")
    print("       'q' 키: 종료 (저장 안 함)\n")

    while True:
        if is_image:
            frame = static_img.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                if cap and cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    print("[INFO] 비디오의 끝에 도달했습니다. 마지막 프레임을 사용합니다.")
                    if frame is None:
                         print("[ERROR] 마지막 프레임도 가져올 수 없습니다. 종료합니다.")
                         break
                else:
                    print("[WARNING] 프레임 읽기 실패 또는 영상 종료 -> 종료")
                    break

        disp = frame.copy()
        point_labels = ["1 (L-Bot)", "2 (R-Bot)", "3 (L-Top)", "4 (R-Top)"]
        for i, pt in enumerate(src_points):
            cv2.circle(disp, pt, 5, (0, 255, 0), -1)
            label = point_labels[i] if i < len(point_labels) else f"{i+1}"
            cv2.putText(disp, label, (pt[0]+5, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(src_points) == 4:
             cv2.polylines(disp, [np.array(src_points, dtype=np.int32)], isClosed=True, color=(0,0,255), thickness=2)

        cv2.imshow("Original", disp)

        bev_result = np.zeros((warp_h, warp_w, 3), dtype=np.uint8)
        if len(src_points) == 4:
            # 점 선택 순서가 '왼쪽 아래, 오른쪽 아래, 왼쪽 위, 오른쪽 위' 이므로
            # dst_points_default 순서와 일치합니다.
            src_np = np.float32(src_points)
            M = cv2.getPerspectiveTransform(src_np, dst_points_default)
            bev_result = cv2.warpPerspective(frame, M, (warp_w, warp_h))
        cv2.imshow("BEV", bev_result)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("[INFO] 'q' 키 입력 -> 종료 (저장 안 함)")
            break
        elif key == ord('r'):
            print("[INFO] 'r' 키 입력 -> 4점 좌표 초기화")
            src_points = []
        elif key == ord('s'):
            if len(src_points) < 4:
                print("[WARNING] 4개의 점을 모두 선택해야 저장이 가능합니다.")
            else:
                print("[INFO] 's' 키 입력 -> BEV 파라미터 저장 후 종료")
                out_file_npz = args.out_npz
                out_file_txt = args.out_txt

                src_arr = np.float32(src_points)
                dst_arr = dst_points_default

                np.savez(out_file_npz,
                         src_points=src_arr,
                         dst_points=dst_arr,
                         warp_w=warp_w,
                         warp_h=warp_h)
                print(f"[INFO] '{out_file_npz}' 파일에 BEV 파라미터 저장 완료.")

                try:
                    with open(out_file_txt, 'w') as f:
                        f.write("# Selected BEV Source Points (x, y) for original image\n")
                        f.write("# Order: Left-Bottom, Right-Bottom, Left-Top, Right-Top\n")
                        for i, point in enumerate(src_points):
                            f.write(f"{point[0]}, {point[1]} # {point_labels[i]}\n")
                    print(f"[INFO] '{out_file_txt}' 파일에 선택된 좌표 저장 완료.")
                except Exception as e:
                    print(f"[ERROR] TXT 파일 저장 중 오류 발생: {e}")
                break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] bev_utils.py 종료.")

if __name__ == '__main__':
    main()