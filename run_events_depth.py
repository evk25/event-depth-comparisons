#!/usr/bin/env python3
"""
EventsDepth Inference Runner
Runs EventDepth on 2-channel event frames (from PNGs, NPYs, or streamed from RAW).
Generates an MP4 video using native events_depth colormap (turbo).
"""

import os
import sys
import glob
import argparse
import subprocess
import cv2
import numpy as np
import torch
import matplotlib.cm as cm

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "models", "events_depth"))

from infer import EventDepthPredictor
from metavision_core.event_io import EventsIterator
from runners.utils import (
    resolve_capture,
    get_capture_frame_count,
    load_event_frame_from_png,
    events_to_2channel_grid,
    send_video_to_discord,
)

def main():
    parser = argparse.ArgumentParser(description="EventsDepth Inference Runner")
    parser.add_argument("--capture", "-c", type=str, default="example_dont_change", help="Capture name or folder")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--no-discord", action="store_true", help="Skip Discord upload")
    args = parser.parse_args()

    capture_dir, capture_short = resolve_capture(args.capture, repo_root)
    print(f"[EventsDepth] Target capture: {capture_short} ({capture_dir})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EventsDepth] Using device: {device}")

    checkpoint_path = os.path.join(repo_root, "models", "events_depth", "checkpoint_epoch_034.pt")
    predictor = EventDepthPredictor(checkpoint_path=checkpoint_path, device=device)

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
    out_video_path = os.path.join(out_dir, f"events_depth{suffix}.mp4")

    # Native colormap turbo as per infer.py
    colormap = cm.get_cmap("turbo")
    h, w = 360, 640

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
        "-crf", "26",
        out_video_path
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    processed_frames = 0
    with torch.no_grad():
        if len(npy_files) > 0:
            if target_num_frames:
                npy_files = npy_files[:target_num_frames]
            num_frames = len(npy_files)
            print(f"[EventsDepth] Processing {num_frames} frames from precomputed NPY files...")
            for i, fpath in enumerate(npy_files):
                events = np.load(fpath)
                res = predictor.predict(events)
                depth = res["depth"]

                p2 = np.percentile(depth, 2)
                p98 = np.percentile(depth, 98)
                depth_norm = np.clip((depth - p2) / (p98 - p2 + 1e-6), 0, 1)

                colored = colormap(depth_norm)[:, :, :3]
                colored_bgr = (colored * 255.0)[:, :, ::-1].astype(np.uint8)
                proc.stdin.write(colored_bgr.tobytes())
                processed_frames += 1

                if (i + 1) % 50 == 0 or (i + 1) == num_frames:
                    print(f"  Processed {i+1}/{num_frames} frames (mean depth: {res['depth_mean']:.2f}m)")
        elif len(png_files) > 0:
            if target_num_frames:
                png_files = png_files[:target_num_frames]
            num_frames = len(png_files)
            print(f"[EventsDepth] Processing {num_frames} frames from event PNG images...")
            for i, fpath in enumerate(png_files):
                events = load_event_frame_from_png(fpath, target_shape=(360, 640))
                res = predictor.predict(events)
                depth = res["depth"]

                p2 = np.percentile(depth, 2)
                p98 = np.percentile(depth, 98)
                depth_norm = np.clip((depth - p2) / (p98 - p2 + 1e-6), 0, 1)

                colored = colormap(depth_norm)[:, :, :3]
                colored_bgr = (colored * 255.0)[:, :, ::-1].astype(np.uint8)
                proc.stdin.write(colored_bgr.tobytes())
                processed_frames += 1

                if (i + 1) % 50 == 0 or (i + 1) == num_frames:
                    print(f"  Processed {i+1}/{num_frames} frames (mean depth: {res['depth_mean']:.2f}m)")
        else:
            if not os.path.exists(raw_path):
                raise FileNotFoundError(f"No event_*.png, event_2ch_*.npy, or events_recording.raw found in {capture_dir}")
            print(f"[EventsDepth] Streaming events directly from {os.path.basename(raw_path)} (target: {target_num_frames} frames)...")
            it = EventsIterator(raw_path, delta_t=33333)
            for ev in it:
                if target_num_frames and processed_frames >= target_num_frames:
                    break
                events = events_to_2channel_grid(ev, target_shape=(360, 640), orig_shape=(720, 1280), accumulation_ms=10.0)
                res = predictor.predict(events)
                depth = res["depth"]

                p2 = np.percentile(depth, 2)
                p98 = np.percentile(depth, 98)
                depth_norm = np.clip((depth - p2) / (p98 - p2 + 1e-6), 0, 1)

                colored = colormap(depth_norm)[:, :, :3]
                colored_bgr = (colored * 255.0)[:, :, ::-1].astype(np.uint8)
                proc.stdin.write(colored_bgr.tobytes())
                processed_frames += 1

                if processed_frames % 50 == 0 or (target_num_frames and processed_frames == target_num_frames):
                    print(f"  Processed {processed_frames}/{target_num_frames or '?'} frames (mean depth: {res['depth_mean']:.2f}m)")

    proc.stdin.close()
    proc.wait()

    file_size_mb = os.path.getsize(out_video_path) / (1024 * 1024)
    print(f"[EventsDepth] Video successfully written: {out_video_path} ({file_size_mb:.2f} MB)")

    if not args.no_discord:
        send_video_to_discord(
            repo_root=repo_root,
            video_path=out_video_path,
            model_name="EventsDepth (EventDepth CNN)",
            dataset_name=f"{capture_short} (Event)",
            title=f"🎬 Benchmark: EventsDepth ({capture_short})",
            message=f"Monocular Event-to-Depth inference completed ({processed_frames} frames @ 30 FPS). Native turbo colormap."
        )

if __name__ == "__main__":
    main()
