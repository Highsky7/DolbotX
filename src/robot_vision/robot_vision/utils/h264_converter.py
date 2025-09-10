#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: video_converter_v2.py
# AUTHOR: Geoffrey Hinton
# DESCRIPTION:
# [Hinton's Workflow-Optimized Video Transcoder]
# 1. 입력 경로의 기본값을 '~/ros2_recordings/unified'로 설정하여 편의성 극대화
# 2. 출력 경로를 입력 경로 기반으로 자동 생성 (e.g., input -> input_h264)
# 3. 커맨드라인 인자를 통해 기본 경로를 오버라이드할 수 있는 유연성 유지
# 4. FFmpeg 및 하드웨어 가속(NVENC 등)을 활용한 초고속 변환 기능은 그대로 유지

import os
import subprocess
import argparse
from pathlib import Path

def convert_videos_to_h264(input_dir, output_dir, codec, crf, preset, delete_original):
    """
    지정된 디렉토리의 비디오 파일들을 FFmpeg를 사용하여 H.264로 변환합니다.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"--- 🚀 Starting H.264 Conversion ---")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Codec: {codec}, CRF: {crf}, Preset: {preset}")
    print("-" * 35)

    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.mov', '.avi'))]

    if not video_files:
        print(f"No video files found in '{input_dir}'. Exiting.")
        return

    for filename in video_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        print(f"▶️ Converting '{filename}'...")

        # --- 이 부분이 수정되었습니다 ---
        # 기본 명령어를 구성합니다.
        command = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', codec
        ]
        
        # libx264 (CPU) 코덱인 경우에만 CRF 옵션을 추가합니다.
        # 옵션과 값을 함께 extend로 추가하여 순서가 꼬이지 않도록 합니다.
        if codec == 'libx264':
            command.extend(['-crf', str(crf)])

        # 나머지 옵션과 출력 경로를 추가합니다.
        command.extend([
            '-preset', preset,
            '-c:a', 'copy', 
            output_path
        ])
        # --- 수정 끝 ---

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"✅ Successfully converted '{filename}'.")

            if delete_original:
                os.remove(input_path)
                print(f"🗑️ Deleted original file: '{filename}'")

        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR converting '{filename}':")
            print(e.stderr)
        except FileNotFoundError:
            print("❌ FATAL ERROR: 'ffmpeg' command not found.")
            print("Please ensure FFmpeg is installed and in your system's PATH.")
            return

def main():
    # --- 이 부분은 수정되지 않았습니다 ---
    parser = argparse.ArgumentParser(
        description="Convert videos to H.264. Defaults to converting '~/ros2_recordings/unified'.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    default_input_path = os.path.expanduser('~/ros2_recordings/unified')
    parser.add_argument(
        "input_dir", 
        nargs='?', 
        default=default_input_path,
        help=f"[Optional] Directory with source videos.\n(Defaults to: '{default_input_path}')"
    )

    parser.add_argument("--codec", default="libx264", help="H.264 encoder to use (e.g., 'libx264' for CPU, 'h264_nvenc' for NVIDIA GPU).")
    parser.add_argument("--crf", type=int, default=23, help="Constant Rate Factor (quality) for libx264. Lower is better quality (0-51). Default: 23.")
    parser.add_argument("--preset", default="fast", help="Encoding speed preset (e.g., 'ultrafast', 'fast', 'medium'). Default: fast.")
    parser.add_argument("--delete-original", action="store_true", help="Delete the original file after successful conversion.")
    
    args = parser.parse_args()
    
    input_path = args.input_dir
    output_path = f"{os.path.normpath(input_path)}_h264"
    
    if not os.path.isdir(input_path):
        print(f"❌ FATAL ERROR: The specified input directory does not exist: {input_path}")
        return

    convert_videos_to_h264(input_path, output_path, args.codec, args.crf, args.preset, args.delete_original)

if __name__ == "__main__":
    main()