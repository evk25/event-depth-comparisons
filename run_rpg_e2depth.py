#!/usr/bin/env python3
"""
RPG E2Depth Inference Runner
Runs RPG E2Depth (E2VIDRecurrent with ConvLSTM) on 5-bin voxel grids.
Maintains temporal ConvLSTM states across the sequence.
Outputs native RPG E2Depth magma colormapped depth video at 30 FPS.
"""

import os
import sys
import argparse
import subprocess
import cv2
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.cm as cm

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "models", "rpg_e2depth"))
sys.path.insert(0, os.path.join(repo_root, "models", "rpg_e2depth", "e2depth"))

from utils.loading_utils import load_model
from utils.inference_utils import CropParameters
from metavision_core.event_io import EventsIterator
from runners.utils import resolve_capture, get_capture_frame_count, send_video_to_discord

def events_to_voxel_grid(events, num_bins, height, width):
    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
    if len(events) == 0:
        return voxel_grid.reshape((num_bins, height, width))
    last_stamp = events['t'][-1]
    first_stamp = events['t'][0]
    deltaT = float(last_stamp - first_stamp)
    if deltaT == 0:
        deltaT = 1.0
    ts = (num_bins - 1) * (events['t'] - first_stamp).astype(np.float32) / deltaT
    xs = (events['x'] // 2).astype(np.int64)
    ys = (events['y'] // 2).astype(np.int64)
    pols = np.where(events['p'] > 0, 1.0, -1.0).astype(np.float32)
    tis = ts.astype(np.int64)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts
    valid_left = tis < num_bins
    np.add.at(voxel_grid, xs[valid_left] + ys[valid_left] * width + tis[valid_left] * width * height, vals_left[valid_left])
    valid_right = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_right] + ys[valid_right] * width + (tis[valid_right] + 1) * width * height, vals_right[valid_right])
    return voxel_grid.reshape((num_bins, height, width))

def main():
    parser = argparse.ArgumentParser(description="RPG E2Depth Inference Runner")
    parser.add_argument("--capture", "-c", type=str, default="example_dont_change", help="Capture name or folder")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--no-discord", action="store_true", help="Skip Discord upload")
    args = parser.parse_args()

    capture_dir, capture_short = resolve_capture(args.capture, repo_root)
    print(f"[RPG E2Depth] Target capture: {capture_short} ({capture_dir})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RPG E2Depth] Using device: {device}")

    checkpoint_path = os.path.join(repo_root, "models", "rpg_e2depth", "pretrained", "E2DEPTH_si_grad_loss_mixed.pth.tar")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[RPG E2Depth] Loading weights from {checkpoint_path}...")
    model = load_model(checkpoint_path).to(device)
    model.eval()

    raw_path = os.path.join(capture_dir, "events_recording.raw")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw recording not found: {raw_path}")

    target_num_frames = get_capture_frame_count(capture_dir)
    if args.max_frames:
        target_num_frames = min(target_num_frames, args.max_frames) if target_num_frames else args.max_frames

    H, W = 360, 640
    crop = CropParameters(W, H, model.num_encoders)

    out_dir = os.path.join(repo_root, "outputs", "videos")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if capture_short == "example_dont_change" else f"_{capture_short}"
    out_video_path = os.path.join(out_dir, f"rpg_e2depth{suffix}.mp4")

    # Native colormap magma as per inference_utils.py
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

    states = None
    print(f"[RPG E2Depth] Processing frames with recurrent ConvLSTM (target: {target_num_frames or 'all'})...")
    
    frame_idx = 0
    with torch.no_grad():
        for ev in it:
            if target_num_frames and frame_idx >= target_num_frames:
                break
            
            # Voxel grid creation (5 bins)
            voxel = events_to_voxel_grid(ev, 5, H, W)
            
            # Non-zero normalization as per VoxelGridDataset
            mask = np.nonzero(voxel)
            if mask[0].size > 0:
                mean, stddev = voxel[mask].mean(), voxel[mask].std()
                if stddev > 0:
                    voxel[mask] = (voxel[mask] - mean) / stddev

            voxel_tensor = torch.from_numpy(voxel).unsqueeze(0).to(device)
            padded = crop.pad(voxel_tensor)

            pred_frame, states = model(padded, states)
            out = pred_frame[0, 0, crop.iy0:crop.iy1, crop.ix0:crop.ix1].cpu().numpy()

            # Native magma colormap visualization as in inference_utils.py:
            vmax = np.percentile(out, 95)
            vmin = out.min()
            if vmax > vmin:
                normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                normalizer = mpl.colors.Normalize(vmin=0, vmax=1)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
            colored = mapper.to_rgba(out)[:, :, :3] # RGB
            colored_bgr = (colored * 255.0)[:, :, ::-1].astype(np.uint8)

            proc.stdin.write(colored_bgr.tobytes())
            frame_idx += 1

            if frame_idx % 50 == 0 or (target_num_frames and frame_idx == target_num_frames):
                print(f"  Processed {frame_idx}/{target_num_frames or '?'} frames (depth range: {out.min():.3f} - {out.max():.3f})")

    proc.stdin.close()
    proc.wait()

    file_size_mb = os.path.getsize(out_video_path) / (1024 * 1024)
    print(f"[RPG E2Depth] Video successfully written: {out_video_path} ({file_size_mb:.2f} MB)")

    if not args.no_discord:
        send_video_to_discord(
            repo_root=repo_root,
            video_path=out_video_path,
            model_name="RPG E2Depth (E2VIDRecurrent)",
            dataset_name=f"{capture_short} (Event)",
            title=f"🎬 Benchmark: RPG E2Depth ({capture_short})",
            message=f"Monocular Event-to-Depth inference completed ({frame_idx} frames @ 30 FPS). Native RPG E2Depth recurrent magma_r colormap (closer objects bright)."
        )

if __name__ == "__main__":
    main()
