#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: video_converter_v2.py
# AUTHOR: DolbotX Team
# DESCRIPTION:
# Workflow-oriented video transcoder.
# 1. Defaults the input directory to '~/ros2_recordings/unified' for convenience.
# 2. Automatically derives an output directory (e.g., input -> input_h264).
# 3. Allows overriding defaults via command-line arguments.
# 4. Preserves high-speed conversion using FFmpeg and hardware accelerators (NVENC, etc.).

import os
import subprocess
import argparse
from pathlib import Path

def convert_videos_to_h264(input_dir, output_dir, codec, crf, preset, delete_original):
    """
    Convert all videos in the given directory to H.264 using FFmpeg.
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

        # Build the FFmpeg command.
        command = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', codec
        ]
        
        # Add CRF only when the CPU encoder (libx264) is used.
        if codec == 'libx264':
            command.extend(['-crf', str(crf)])

        # Add the remaining options and the output path.
        command.extend([
            '-preset', preset,
            '-c:a', 'copy', 
            output_path
        ])
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
