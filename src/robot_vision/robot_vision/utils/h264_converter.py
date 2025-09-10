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
    (이 함수의 내용은 이전 버전과 동일합니다)
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

        command = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', codec, '-preset', preset,
            '-c:a', 'copy', output_path
        ]
        
        # libx264 (CPU) 코덱인 경우에만 CRF 옵션 추가
        if codec == 'libx264':
            command.insert(7, '-crf')
            command.insert(8, str(crf))

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
    # --- 이 부분이 수정되었습니다 ---
    parser = argparse.ArgumentParser(
        description="Convert videos to H.264. Defaults to converting '~/ros2_recordings/unified'.",
        formatter_class=argparse.RawTextHelpFormatter # 도움말 줄바꿈을 예쁘게 표시
    )
    
    # 입력 디렉토리를 위치 인자(positional argument)이면서 동시에 기본값을 갖도록 설정
    # nargs='?': 인자가 없을 수도 있음을 의미
    # os.path.expanduser: '~'를 사용자의 홈 디렉토리로 변환
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
    
    # 입력 디렉토리를 기반으로 출력 디렉토리 자동 설정
    input_path = args.input_dir
    output_path = f"{os.path.normpath(input_path)}_h264" # 입력 경로 끝에 슬래시가 있어도 처리
    
    if not os.path.isdir(input_path):
        print(f"❌ FATAL ERROR: The specified input directory does not exist: {input_path}")
        return

    convert_videos_to_h264(input_path, output_path, args.codec, args.crf, args.preset, args.delete_original)

if __name__ == "__main__":
    main()