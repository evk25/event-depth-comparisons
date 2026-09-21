#!/usr/bin/env python3
"""
F3 (Fast Feature Field) Inference Runner
Runs EventFFDepthAnythingV2 on the event stream with 20ms context window per frame.
Outputs native F3 disparity/depth video using magma colormap at 30 FPS.
"""

import os
import sys
import argparse
import subprocess
import cv2
import numpy as np
import torch
import matplotlib.cm as cm

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
f3_dir = os.path.join(repo_root, "models", "f3")
sys.path.insert(0, os.path.join(f3_dir, "src"))

from f3.tasks.depth.utils import init_depth_model
from metavision_core.event_io import EventsIterator
from runners.utils import resolve_capture, get_capture_frame_count, send_video_to_discord

def main():
    parser = argparse.ArgumentParser(description="F3 Inference Runner")
    parser.add_argument("--capture", "-c", type=str, default="example_dont_change", help="Capture name or folder")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--no-discord", action="store_true", help="Skip Discord upload")
    args = parser.parse_args()

    capture_dir, capture_short = resolve_capture(args.capture, repo_root)
    print(f"[F3] Target capture: {capture_short} ({capture_dir})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[F3] Using device: {device}")

    # Switch working directory to models/f3 so relative configs resolve correctly
    cwd = os.getcwd()
    os.chdir(f3_dir)

    config_path = "weights/dav2b_fullm3ed_pseudo_518x518x20/depth_config.yml"
    checkpoint_path = "weights/dav2b_fullm3ed_pseudo_518x518x20/best.pth"

    print(f"[F3] Initializing model from {config_path}...")
    model = init_depth_model(config_path).to(device)

    print(f"[F3] Loading weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, weights_only=True)
    state_dict = {k.replace("._orig_mod.", "."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    os.chdir(cwd)

    raw_path = os.path.join(capture_dir, "events_recording.raw")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw recording not found: {raw_path}")

    target_num_frames = get_capture_frame_count(capture_dir)
    if args.max_frames:
        target_num_frames = min(target_num_frames, args.max_frames) if target_num_frames else args.max_frames

    out_dir = os.path.join(repo_root, "outputs", "videos")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if capture_short == "example_dont_change" else f"_{capture_short}"
    out_video_path = os.path.join(out_dir, f"f3{suffix}.mp4")

    # Native resolution: 1280x720
    H, W = 720, 1280
    cmap = cm.get_cmap("magma")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "bgr24",
        "-r", "30",
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        out_video_path
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    delta_t_us = 33333
    it = EventsIterator(raw_path, delta_t=delta_t_us)

    print(f"[F3] Processing frames with EventFFDepthAnythingV2 (target: {target_num_frames or 'all'})...")
    frame_idx = 0
    with torch.no_grad():
        for ev in it:
            if target_num_frames and frame_idx >= target_num_frames:
                break

            N = len(ev)
            if N == 0:
                colored_bgr = np.zeros((H, W, 3), dtype=np.uint8)
            else:
                _used = min(N, 800000)
                sub_ev = ev[-_used:] # Take most recent events in context window
                t_end = sub_ev['t'][-1]

                ctx = np.empty((_used, 4), dtype=np.float32)
                ctx[:, 0] = sub_ev['x'] / 1280.0
                ctx[:, 1] = sub_ev['y'] / 720.0
                ctx[:, 2] = np.clip((t_end - sub_ev['t']) // 1000, 0, 19) / 20.0
                ctx[:, 3] = sub_ev['p']

                ctx_tensor = torch.from_numpy(ctx).to(device)
                totcnt = torch.tensor([_used], device=device)

                depth_pred, _ = model.infer_image(ctx_tensor, totcnt)
                depth_np = depth_pred.cpu().numpy()

                # Native F3 visualization using magma colormap
                d_min = np.percentile(depth_np, 2)
                d_max = np.percentile(depth_np, 98)
                if d_max > d_min:
                    depth_norm = np.clip((depth_np - d_min) / (d_max - d_min), 0, 1)
                else:
                    depth_norm = np.zeros_like(depth_np)

                colored = cmap(depth_norm)[:, :, :3]
                colored_bgr = (colored * 255.0)[:, :, ::-1].astype(np.uint8)

            proc.stdin.write(colored_bgr.tobytes())
            frame_idx += 1

            if frame_idx % 50 == 0 or (target_num_frames and frame_idx == target_num_frames):
                print(f"  Processed {frame_idx}/{target_num_frames or '?'} frames (events: {N})")

    proc.stdin.close()
    proc.wait()

    file_size_mb = os.path.getsize(out_video_path) / (1024 * 1024)
    print(f"[F3] Video successfully written: {out_video_path} ({file_size_mb:.2f} MB)")

    if not args.no_discord:
        send_video_to_discord(
            repo_root=repo_root,
            video_path=out_video_path,
            model_name="F3 (Fast Feature Field Depth)",
            dataset_name=f"{capture_short} (Event)",
            title=f"🎬 Benchmark: F3 ({capture_short})",
            message=f"Monocular Event-to-Depth inference completed ({frame_idx} frames @ 30 FPS). Native F3 magma colormap."
        )

if __name__ == "__main__":
    main()
