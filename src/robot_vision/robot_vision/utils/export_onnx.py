#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: export_onnx.py
# AUTHOR: Geoffrey Hinton
# DESCRIPTION: PyTorch(.pt) 모델을 TensorRT 변환에 최적화된 ONNX(.onnx) 형식으로 변환합니다.

from ultralytics import YOLO
import torch

def export_single_model(model_path, model_name):
    """지정된 단일 모델을 ONNX 형식으로 변환하는 함수"""
    print("─" * 50)
    print(f"🚀 Starting export for: {model_name} ({model_path})")
    
    # 1. 변환할 .pt 모델을 로드합니다.
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model {model_path}. Reason: {e}")
        return

    # 2. ONNX로 모델을 변환(export)합니다.
    #    - half=True: FP16 정밀도로 변환하여 RTX 40 시리즈 GPU 성능 극대화
    #    - dynamic=True: 다양한 배치 크기 입력을 허용 (TensorRT 빌드 시 필수)
    #    - opset=12: 안정적인 ONNX 버전 세트 지정
    try:
        output_name = model.export(
            format='onnx',
            half=True,
            dynamic=True,
            opset=12
        )
        print(f"✅ SUCCESS: Model exported to ONNX format at: {output_name}")
    except Exception as e:
        print(f"❌ ERROR: Failed to export {model_name}. Reason: {e}")
    print("─" * 50)

def main():
    # 변환할 모델 목록 (파일 경로와 별칭)
    models_to_export = {
        'tracking': './tracking.pt',
        'vision_enemy': './vision_enemy2.pt',
        'traffic_light': './traffic_light.pt',
        'path_planning': './YOLOTL.pt'
    }
    
    # PyTorch 및 CUDA 환경 정보 출력
    print("PyTorch Version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device Name:", torch.cuda.get_device_name(0))

    # 목록에 있는 모든 모델을 순차적으로 변환
    for name, path in models_to_export.items():
        export_single_model(path, name)

if __name__ == '__main__':
    main()