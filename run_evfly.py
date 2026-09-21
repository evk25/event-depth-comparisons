#!/usr/bin/env python3
"""
EVFly Inference Runner
Runs EVFly (OrigUNet with recurrent ConvLSTM) on event frames.
Maintains temporal hidden states across the sequence.
Outputs native EVFly grayscale depth estimation video at 30 FPS.
"""

import os
import sys
import glob
import argparse
import subprocess
import cv2
import numpy as np
import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "models", "evfly", "learner"))

from learner_models import OrigUNet
from metavision_core.event_io import EventsIterator
from runners.utils import (
    resolve_capture,
    get_capture_frame_count,
    load_event_frame_from_png,
    events_to_2channel_grid,
    send_video_to_discord,
)

def main():
    parser = argparse.ArgumentParser(description="EVFly Inference Runner")
    parser.add_argument("--capture", "-c", type=str, default="example_dont_change", help="Capture name or folder")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--no-discord", action="store_true", help="Skip Discord upload")
    args = parser.parse_args()

    capture_dir, capture_short = resolve_capture(args.capture, repo_root)
    print(f"[EVFly] Target capture: {capture_short} ({capture_dir})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EVFly] Using device: {device}")

    # Model architecture
    net = OrigUNet(
        num_in_channels=1,
        num_out_channels=1,
        num_recurrent=[1],
        input_shape=[1, 1, 260, 346],
        velpred=0,
        device=device
    ).to(device)

    # Load weights
    checkpoint_path = os.path.join(repo_root, "models", "evfly", "pretrained_models", "real_forest_Dtheta.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[EVFly] Loading weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(ckpt, strict=True)
    net.eval()

    event_dir = os.path.join(capture_dir, "event_frames")
    npy_files = sorted(glob.glob(os.path.join(event_dir, "event_2ch_*.npy"))) if os.path.isdir(event_dir) else []
    png_files = sorted(glob.glob(os.path.join(event_dir, "event_*.png"))) if os.path.isdir(event_dir) else []
    
    raw_path = os.path.join(capture_dir, "events_recording.raw")
    target_num_frames = get_capture_frame_count(capture_dir)
    if args.max_frames:
        target_num_frames = min(target_num_frames, args.max_frames) if target_num_frames else args.max_frames

    out_dir = os.path.join(repo_root, "outputs", "videos")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if capture_short == "example_dont_change" else f"_{capture_short}"
    out_video_path = os.path.join(out_dir, f"evfly{suffix}.mp4")

    target_h, target_w = 260, 346

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{target_w}x{target_h}",
        "-pix_fmt", "bgr24",
        "-r", "30",
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        out_video_path
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    hidden_state = None
    processed_frames = 0

    print(f"[EVFly] Processing frames with recurrent ConvLSTM...")
    with torch.no_grad():
        if len(npy_files) > 0:
            if target_num_frames:
                npy_files = npy_files[:target_num_frames]
            num_frames = len(npy_files)
            print(f"[EVFly] Found {num_frames} precomputed event frame files.")
            for i, fpath in enumerate(npy_files):
                npy = np.load(fpath) # (2, 360, 640)
                evframe = npy[0] - npy[1]
                evframe_resized = cv2.resize(evframe, (target_w, target_h), interpolation=cv2.INTER_AREA)

                input_tensor = torch.from_numpy(evframe_resized).view(1, 1, target_h, target_w).to(device).float()
                q = torch.quantile(input_tensor.abs(), 0.97)
                if q > 0:
                    input_tensor = torch.clip(input_tensor / q, -1.0, 1.0)

                full_input = [input_tensor, None, [hidden_state, None]]
                vel, (y_interp, y_upconv, (hidden_state, _)) = net(full_input)

                pred_depth = y_interp.squeeze().cpu().numpy()
                pred_depth_clipped = np.clip(pred_depth, 0.0, 1.0)
                depth_uint8 = (pred_depth_clipped * 255.0).astype(np.uint8)
                depth_bgr = np.stack([depth_uint8] * 3, axis=-1)

                proc.stdin.write(depth_bgr.tobytes())
                processed_frames += 1

                if (i + 1) % 50 == 0 or (i + 1) == num_frames:
                    print(f"  Processed {i+1}/{num_frames} frames (depth min: {pred_depth.min():.3f}, max: {pred_depth.max():.3f})")
        elif len(png_files) > 0:
            if target_num_frames:
                png_files = png_files[:target_num_frames]
            num_frames = len(png_files)
            print(f"[EVFly] Found {num_frames} event PNG files.")
            for i, fpath in enumerate(png_files):
                npy = load_event_frame_from_png(fpath, target_shape=(360, 640))
                evframe = npy[0] - npy[1]
                evframe_resized = cv2.resize(evframe, (target_w, target_h), interpolation=cv2.INTER_AREA)

                input_tensor = torch.from_numpy(evframe_resized).view(1, 1, target_h, target_w).to(device).float()
                q = torch.quantile(input_tensor.abs(), 0.97)
                if q > 0:
                    input_tensor = torch.clip(input_tensor / q, -1.0, 1.0)

                full_input = [input_tensor, None, [hidden_state, None]]
                vel, (y_interp, y_upconv, (hidden_state, _)) = net(full_input)

                pred_depth = y_interp.squeeze().cpu().numpy()
                pred_depth_clipped = np.clip(pred_depth, 0.0, 1.0)
                depth_uint8 = (pred_depth_clipped * 255.0).astype(np.uint8)
                depth_bgr = np.stack([depth_uint8] * 3, axis=-1)

                proc.stdin.write(depth_bgr.tobytes())
                processed_frames += 1

                if (i + 1) % 50 == 0 or (i + 1) == num_frames:
                    print(f"  Processed {i+1}/{num_frames} frames (depth min: {pred_depth.min():.3f}, max: {pred_depth.max():.3f})")
        else:
            if not os.path.exists(raw_path):
                raise FileNotFoundError(f"Neither event_*.png nor events_recording.raw found in {capture_dir}")
            print(f"[EVFly] Streaming events directly from {os.path.basename(raw_path)} (target: {target_num_frames} frames)...")
            it = EventsIterator(raw_path, delta_t=33333)
            for ev in it:
                if target_num_frames and processed_frames >= target_num_frames:
                    break
                npy = events_to_2channel_grid(ev, target_shape=(360, 640), orig_shape=(720, 1280), accumulation_ms=10.0)
                evframe = npy[0] - npy[1]
                evframe_resized = cv2.resize(evframe, (target_w, target_h), interpolation=cv2.INTER_AREA)

                input_tensor = torch.from_numpy(evframe_resized).view(1, 1, target_h, target_w).to(device).float()
                q = torch.quantile(input_tensor.abs(), 0.97)
                if q > 0:
                    input_tensor = torch.clip(input_tensor / q, -1.0, 1.0)

                full_input = [input_tensor, None, [hidden_state, None]]
                vel, (y_interp, y_upconv, (hidden_state, _)) = net(full_input)

                pred_depth = y_interp.squeeze().cpu().numpy()
                pred_depth_clipped = np.clip(pred_depth, 0.0, 1.0)
                depth_uint8 = (pred_depth_clipped * 255.0).astype(np.uint8)
                depth_bgr = np.stack([depth_uint8] * 3, axis=-1)

                proc.stdin.write(depth_bgr.tobytes())
                processed_frames += 1

                if processed_frames % 50 == 0 or (target_num_frames and processed_frames == target_num_frames):
                    print(f"  Processed {processed_frames}/{target_num_frames or '?'} frames (depth min: {pred_depth.min():.3f}, max: {pred_depth.max():.3f})")

    proc.stdin.close()
    proc.wait()

    file_size_mb = os.path.getsize(out_video_path) / (1024 * 1024)
    print(f"[EVFly] Video successfully written: {out_video_path} ({file_size_mb:.2f} MB)")

    if not args.no_discord:
        send_video_to_discord(
            repo_root=repo_root,
            video_path=out_video_path,
            model_name="EVFly (OrigUNet ConvLSTM)",
            dataset_name=f"{capture_short} (Event)",
            title=f"🎬 Benchmark: EVFly ({capture_short})",
            message=f"Monocular Event-to-Depth inference completed ({processed_frames} frames @ 30 FPS). Native EVFly recurrent grayscale depth."
        )

if __name__ == "__main__":
    main()
