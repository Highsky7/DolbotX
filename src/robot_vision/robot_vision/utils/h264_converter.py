#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: h264_converter.py
# AUTHOR: Geoffrey Hinton
# DESCRIPTION:
# [Hinton's FFMPEG-based H.264 Transcoding Architecture]
# 1. 산업 표준 비디오 처리 도구인 'FFmpeg'를 사용하여 최고의 신뢰성과 품질 보장
# 2. H.264(libx264) 코덱을 사용하여 압축 효율과 장치 호환성을 극대화
# 3. CRF(Constant Rate Factor) 옵션을 통해 화질은 유지하면서 파일 크기를 최적화 (crf=23)
# 4. 인코딩 속도와 압축률의 균형을 맞추는 'preset' 옵션 적용 (preset=medium)
# 5. 재사용성과 편의성을 위해 커맨드 라인 인자(--input, --output)를 받는 전문가용 인터페이스 채택
# 6. 변환 과정 중 발생할 수 있는 모든 예외를 처리하는 강력한 에러 핸들링 로직 포함

import subprocess
import argparse
import sys
import os

def check_ffmpeg_installed():
    """
    시스템에 FFmpeg가 설치되어 있는지 확인하는 함수입니다.
    전문가라면, 도구가 준비되었는지 항상 먼저 확인해야 합니다.
    """
    try:
        # 'ffmpeg -version' 명령을 실행하여 FFmpeg의 존재를 확인합니다.
        # stdout과 stderr를 DEVNULL로 보내 출력을 숨깁니다.
        subprocess.run(
            ['ffmpeg', '-version'], 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        print("✅ FFmpeg is installed and ready.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FATAL: FFmpeg is not installed or not found in your system's PATH.")
        print("Please install FFmpeg to proceed. For example, on Ubuntu/Debian:")
        print("sudo apt-get update && sudo apt-get install ffmpeg")
        return False

def convert_to_h264(input_path: str, output_path: str):
    """
    지정된 비디오 파일을 H.264 코덱을 사용하여 변환합니다.

    :param input_path: 원본 비디오 파일 경로
    :param output_path: H.264로 변환하여 저장할 파일 경로
    """
    # 입력 파일이 실제로 존재하는지 확인합니다.
    if not os.path.exists(input_path):
        print(f"❌ ERROR: Input file not found at '{input_path}'")
        return

    print(f"--- Starting H.264 Conversion (Hinton's Method) ---")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")

    # 이것이 바로 FFmpeg의 핵심 커맨드입니다. 각 옵션은 최고의 결과물을 위해 제가 신중히 선택했습니다.
    ffmpeg_command = [
        'ffmpeg',
        '-y',  # 출력 파일이 이미 존재할 경우 덮어쓰기를 허용합니다.
        '-i', input_path,  # 입력 파일을 지정합니다.
        '-c:v', 'libx264',  # 비디오 코덱을 H.264로 설정합니다. 'libx264'는 가장 널리 쓰이는 고품질 H.264 인코더입니다.
        '-preset', 'medium',  # 인코딩 속도와 압축률 간의 균형을 설정합니다. 'medium'이 표준적인 선택입니다.
        '-crf', '23',  # Constant Rate Factor. 화질을 제어하는 옵션입니다. 18-28 사이가 일반적이며, 23은 좋은 화질과 파일 크기의 균형점입니다.
        '-c:a', 'aac',  # 오디오 코덱을 AAC로 설정합니다. MP4 컨테이너의 표준 오디오 코덱입니다.
        '-b:a', '128k', # 오디오 비트레이트를 128kbps로 설정합니다. 원본에 오디오가 없더라도 호환성을 위해 포함하는 것이 좋습니다.
        output_path
    ]

    try:
        print("\n🚀 Executing FFmpeg command...")
        print(f"   Command: {' '.join(ffmpeg_command)}")
        
        # FFmpeg 명령을 실행합니다. check=True는 명령 실행 중 오류가 발생하면 예외를 발생시킵니다.
        result = subprocess.run(
            ffmpeg_command, 
            check=True, 
            capture_output=True, # stdout, stderr 결과를 캡처합니다.
            text=True # 결과를 텍스트로 디코딩합니다.
        )
        
        print("\n✅ Conversion successful!")
        print(f"H.264 encoded video saved to: {output_path}")
        # 상세한 FFmpeg 로그를 보고 싶다면 아래 주석을 해제하십시오.
        # print("\n--- FFmpeg Log ---")
        # print(result.stderr)
        # print("--------------------")

    except FileNotFoundError:
        print(f"❌ ERROR: 'ffmpeg' command not found. Make sure FFmpeg is installed and in your PATH.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ An error occurred during conversion.")
        print("FFmpeg returned a non-zero exit code, indicating failure.")
        print("\n--- FFmpeg Error Log ---")
        print(e.stderr) # FFmpeg가 출력한 에러 메시지를 보여주어 디버깅을 돕습니다.
        print("--------------------------")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

if __name__ == '__main__':
    # 스크립트를 커맨드 라인에서 쉽게 사용할 수 있도록 argparse를 사용합니다.
    parser = argparse.ArgumentParser(
        description="Hinton's High-Quality H.264 Video Converter using FFmpeg.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input', 
        type=str, 
        required=True, 
        help='Path to the input MP4 file.'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True, 
        help='Path for the output H.264 encoded MP4 file.'
    )
    
    args = parser.parse_args()

    # 스크립트 실행 전, FFmpeg가 설치되어 있는지 먼저 확인합니다.
    if check_ffmpeg_installed():
        convert_to_h264(args.input, args.output)
    else:
        sys.exit(1) # FFmpeg가 없으면 프로그램을 종료합니다.