# hsv_picker.py
import cv2
import numpy as np

# 마우스 클릭 이벤트 콜백 함수
def get_hsv_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_pixel = cv2.cvtColor(np.uint8([[frame[y,x]]]), cv2.COLOR_BGR2HSV)
        print(f"Clicked Pixel BGR: {frame[y,x]}, HSV: {hsv_pixel[0][0]}")

# 1단계에서 캡처한 이미지 파일의 경로를 입력하세요.
image_path = '/home/highsky/Desktop/2025-07-31-015447.jpg'
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Image not found!")
    exit()
frame = cv2.resize(frame, (800, 600))  # 이미지 크기 조정

cv2.imshow('HSV Picker - Click on the track', frame)
cv2.setMouseCallback('HSV Picker - Click on the track', get_hsv_value)

print("Track 위를 마우스로 클릭하여 HSV 값을 확인하세요. 종료하려면 'q'를 누르세요.")
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()