#!/usr/bin/env python3
"""
Depth-Anything-V2 Inference Runner
Runs Depth-Anything-V2 on the RGB images from captures/<capture>/rgb/
Generates an MP4 video using the native Depth-Anything-V2 colormap (Spectral_r).
"""

import os
import sys
import glob
import argparse
import subprocess
import cv2
import numpy as np
import torch
import matplotlib

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "models", "Depth-Anything-V2"))

from depth_anything_v2.dpt import DepthAnythingV2
from runners.utils import resolve_capture, send_video_to_discord

def main():
    parser = argparse.ArgumentParser(description="Depth-Anything-V2 Inference Runner")
    parser.add_argument("--capture", "-c", type=str, default="example_dont_change", help="Capture name or folder")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--no-discord", action="store_true", help="Skip Discord upload")
    args = parser.parse_args()

    capture_dir, capture_short = resolve_capture(args.capture, repo_root)
    print(f"[Depth-Anything-V2] Target capture: {capture_short} ({capture_dir})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Depth-Anything-V2] Using device: {device}")

    # Model configuration
    model_config = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    checkpoint_path = os.path.join(repo_root, "models", "Depth-Anything-V2", "checkpoints", "depth_anything_v2_vits.pth")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[Depth-Anything-V2] Loading weights from {checkpoint_path}...")
    model = DepthAnythingV2(**model_config)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device).eval()

    # Input RGB images
    rgb_dir = os.path.join(capture_dir, "rgb")
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    if args.max_frames:
        rgb_files = rgb_files[:args.max_frames]
    num_frames = len(rgb_files)
    print(f"[Depth-Anything-V2] Found {num_frames} RGB frames in {rgb_dir}")

    if num_frames == 0:
        raise RuntimeError(f"No RGB frames found in {rgb_dir}!")

    # Output setup
    out_dir = os.path.join(repo_root, "outputs", "videos")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if capture_short == "example_dont_change" else f"_{capture_short}"
    out_video_path = os.path.join(out_dir, f"depth_anything_v2{suffix}.mp4")

    # Native colormap Spectral_r as per run.py
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")

    # Read first image to determine output dimensions
    sample_img = cv2.imread(rgb_files[0])
    h, w = sample_img.shape[:2]

    # Setup ffmpeg process for encoding
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", "30",
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        out_video_path
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    print(f"[Depth-Anything-V2] Processing {num_frames} frames...")
    with torch.no_grad():
        for i, fpath in enumerate(rgb_files):
            img = cv2.imread(fpath)
            depth = model.infer_image(img, input_size=518)

            # Native normalization to [0, 255] and Spectral_r mapping
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                depth_norm = (depth - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth)

            depth_uint8 = (depth_norm * 255.0).astype(np.uint8)
            colored = (cmap(depth_uint8)[:, :, :3] * 255.0)[:, :, ::-1].astype(np.uint8)
            proc.stdin.write(colored.tobytes())

            if (i + 1) % 50 == 0 or (i + 1) == num_frames:
                print(f"  Processed {i+1}/{num_frames} frames")

    proc.stdin.close()
    proc.wait()

    file_size_mb = os.path.getsize(out_video_path) / (1024 * 1024)
    print(f"[Depth-Anything-V2] Video successfully written: {out_video_path} ({file_size_mb:.2f} MB)")

    # Send to Discord using skill
    if not args.no_discord:
        send_video_to_discord(
            repo_root=repo_root,
            video_path=out_video_path,
            model_name="Depth-Anything-V2 (RGB Baseline)",
            dataset_name=f"{capture_short} (RGB)",
            title=f"🎬 Benchmark: Depth-Anything-V2 ({capture_short})",
            message=f"Monocular RGB baseline inference completed ({num_frames} frames @ 30 FPS). Native Spectral_r colormap."
        )

if __name__ == "__main__":
    main()
