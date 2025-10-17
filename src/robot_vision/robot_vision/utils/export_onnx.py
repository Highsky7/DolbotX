#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: export_onnx.py
# AUTHOR: DolbotX Team
# DESCRIPTION: Convert PyTorch (.pt) models into TensorRT-ready ONNX (.onnx) files.

from ultralytics import YOLO
import torch

def export_single_model(model_path, model_name):
    """Export a single model to ONNX format."""
    print("─" * 50)
    print(f"🚀 Starting export for: {model_name} ({model_path})")
    
    # 1. Load the source .pt model.
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model {model_path}. Reason: {e}")
        return

    # 2. Export the model to ONNX.
    #    - half=True: use FP16 precision to maximise RTX 40-series performance
    #    - dynamic=True: allow variable batch sizes (required for TensorRT builds)
    #    - opset=12: target a stable ONNX opset
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
    # Models to export (path -> alias)
    models_to_export = {
        'tracking': './tracking.pt',
        'vision_enemy': './vision_enemy2.pt',
        'traffic_light': './traffic_light.pt',
        'path_planning': './YOLOTL.pt'
    }
    
    # Report PyTorch and CUDA environment info
    print("PyTorch Version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device Name:", torch.cuda.get_device_name(0))

    # Export each configured model sequentially
    for name, path in models_to_export.items():
        export_single_model(path, name)

if __name__ == '__main__':
    main()
