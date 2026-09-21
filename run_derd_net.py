#!/usr/bin/env python3
"""
DERD-Net Inference Runner (Corrected DSI formulation)
===================================================
Constructs true Disparity Space Image (DSI) volumes by back-projecting
event rays across hypothesis depth/disparity planes d in [0, D-1].
Predicts semi-dense depth at confident ray intersection peaks using the
DSEC monocular PixelwiseConvGRU ensemble.
Renders with native thickened pixels and white background using jet colormap.
"""

import os
import sys
import argparse
import subprocess
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from metavision_core.event_io import EventsIterator
from runners.utils import resolve_capture, get_capture_frame_count, send_video_to_discord

class PixelwiseConvGRU(nn.Module):
    def __init__(self, sub_frame_radius_h=3, sub_frame_radius_w=3, out_channels=4,
                 multi_pixel=False, use_pixel_pos=False, hidden_size_scale=1,
                 num_gru_layers=1, bidirectional=False, dropout_rate=0):
        super(PixelwiseConvGRU, self).__init__()
        self.sub_frame_radius_h = sub_frame_radius_h
        self.sub_frame_radius_w = sub_frame_radius_w
        self.out_channels = out_channels
        self.multi_pixel = multi_pixel
        self.output_dim = 9 if self.multi_pixel else 1
        self.use_pixel_pos = use_pixel_pos
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate

        self.conv3d = nn.Sequential(
            nn.Conv3d(1, self.out_channels, kernel_size=3),
            nn.LeakyReLU()
        )
        self.sub_height = 2 * self.sub_frame_radius_h + 1
        self.sub_width = 2 * self.sub_frame_radius_w + 1
        self.gru_input_size = self.out_channels * (self.sub_height - 2) * (self.sub_width - 2)
        self.gru_hidden_size = self.gru_input_size * hidden_size_scale
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=self.gru_hidden_size,
            num_layers=num_gru_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=self.dropout_rate
        )
        self.dense_output = nn.Sequential(
            nn.Linear((1 + self.bidirectional) * self.gru_hidden_size + 2 * self.use_pixel_pos, self.gru_hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.gru_hidden_size, self.output_dim),
            nn.LeakyReLU()
        )

    def forward(self, input):
        pixel_position, sub_dsi = input
        batch_size, depth_levels = sub_dsi.shape[:2]
        sub_dsi_conv = self.conv3d(sub_dsi.unsqueeze(dim=1))
        sub_dsi_conv_flat = sub_dsi_conv.transpose(1, 2).flatten(start_dim=2)
        h_seq, _ = self.gru(sub_dsi_conv_flat)
        h_n = h_seq[:, -1, :]
        if self.use_pixel_pos:
            h_n = torch.cat([pixel_position, h_n], dim=-1)
        output = self.dense_output(h_n)
        if not self.multi_pixel:
            output = output.squeeze(dim=-1)
        return output

class AveragedNetwork(nn.Module):
    def __init__(self, neural_nets):
        super(AveragedNetwork, self).__init__()
        self.models = nn.ModuleList(neural_nets)
        self.multi_pixel = neural_nets[0].multi_pixel
        self.sub_frame_radius_h = neural_nets[0].sub_frame_radius_h
        self.sub_frame_radius_w = neural_nets[0].sub_frame_radius_w

    def forward(self, input):
        outputs = [m(input) for m in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

def main():
    parser = argparse.ArgumentParser(description="DERD-Net Inference Runner")
    parser.add_argument("--capture", "-c", type=str, default="example_dont_change", help="Capture name or folder")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--no-discord", action="store_true", help="Skip Discord upload")
    args = parser.parse_args()

    capture_dir, capture_short = resolve_capture(args.capture, repo_root)
    print(f"[DERD-Net] Target capture: {capture_short} ({capture_dir})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DERD-Net] Using device: {device}")

    # Load pretrained ensemble
    m_even_path = os.path.join(repo_root, "models", "DERD-Net", "models", "dsec", "dsec_half1", "dsec_half1_monocular_even_model.pth")
    m_odd_path = os.path.join(repo_root, "models", "DERD-Net", "models", "dsec", "dsec_half1", "dsec_half1_monocular_odd_model.pth")
    
    m1 = PixelwiseConvGRU(3, 3).to(device)
    m1.load_state_dict(torch.load(m_even_path, map_location=device)["model_state_dict"])
    m2 = PixelwiseConvGRU(3, 3).to(device)
    m2.load_state_dict(torch.load(m_odd_path, map_location=device)["model_state_dict"])
    model = AveragedNetwork([m1, m2]).to(device).eval()
    print(f"[DERD-Net] Successfully loaded DSEC monocular ensemble.")

    raw_path = os.path.join(capture_dir, "events_recording.raw")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw recording not found: {raw_path}")

    target_num_frames = get_capture_frame_count(capture_dir)
    if args.max_frames:
        target_num_frames = min(target_num_frames, args.max_frames) if target_num_frames else args.max_frames

    out_dir = os.path.join(repo_root, "outputs", "videos")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if capture_short == "example_dont_change" else f"_{capture_short}"
    out_video_path = os.path.join(out_dir, f"derd_net{suffix}.mp4")

    D, H, W = 30, 360, 640
    batch_size = 2048
    thicken_pixels = 1

    # Colormap jet with white background (as in official DERD-Net Visualization.ipynb)
    cmap = plt.colormaps["jet"]
    cmap.set_bad(color="white")

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
        "-crf", "28",
        out_video_path
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    delta_t_us = 33333
    it = EventsIterator(raw_path, delta_t=delta_t_us)

    print(f"[DERD-Net] Processing frames with true DSI ray back-projection (target: {target_num_frames or 'all'})...")
    frame_idx = 0
    with torch.no_grad():
        for ev in it:
            if target_num_frames and frame_idx >= target_num_frames:
                break

            if len(ev) < 50:
                frame_bgr = np.full((H, W, 3), 255, dtype=np.uint8)
                proc.stdin.write(frame_bgr.tobytes())
                frame_idx += 1
                continue

            t0, t1 = ev['t'][0], ev['t'][-1]
            tau = (ev['t'] - t0).astype(np.float32) / max(1.0, float(t1 - t0)) - 0.5
            xs = (ev['x'] // 2).astype(np.float32)
            ys = (ev['y'] // 2).astype(np.float32)

            early = ev[ev['t'] < t0 + (t1 - t0)*0.3]
            late = ev[ev['t'] > t0 + (t1 - t0)*0.7]
            mx = late['x'].mean() - early['x'].mean() if len(early) and len(late) else 0.0
            my = late['y'].mean() - early['y'].mean() if len(early) and len(late) else 0.0
            speed = np.hypot(mx, my)
            vx, vy = (mx / speed, my / speed) if speed >= 1.0 else (1.0, 0.0)

            dsi = np.zeros((D, H, W), dtype=np.float32)
            for d in range(D):
                xd = np.clip(np.round(xs + d * tau * vx * 1.5).astype(int), 0, W - 1)
                yd = np.clip(np.round(ys + d * tau * vy * 1.5).astype(int), 0, H - 1)
                np.add.at(dsi[d], (yd, xd), 1.0)

            dsi_t = torch.from_numpy(dsi).to(device).flip(dims=[0])

            conf_map = dsi_t.max(dim=0)[0].cpu().numpy()
            conf_norm = np.around(conf_map * 255.0 / max(1.0, conf_map.max())).astype(np.uint8)
            mask = cv2.adaptiveThreshold(conf_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, -4).astype(bool)

            border = np.zeros_like(mask)
            border[3:-3, 3:-3] = True
            indices = torch.tensor(list(zip(*np.where(mask & border))), device=device)

            if len(indices) == 0:
                frame_bgr = np.full((H, W, 3), 255, dtype=np.uint8)
                proc.stdin.write(frame_bgr.tobytes())
                frame_idx += 1
                continue

            h_idx = indices[:, 0]
            w_idx = indices[:, 1]
            h_off = torch.arange(-3, 4, device=device)
            w_off = torch.arange(-3, 4, device=device)
            hg, wg = torch.meshgrid(h_off, w_off, indexing='ij')
            h_all = h_idx.unsqueeze(1) + hg.flatten()
            w_all = w_idx.unsqueeze(1) + wg.flatten()
            sub_dsis = dsi_t[:, h_all, w_all].transpose(0, 1).view(len(indices), D, 7, 7)
            sub_dsis = sub_dsis / sub_dsis.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
            norm_pos = indices.float() / torch.tensor([H, W], device=device)

            preds = []
            for b in range(0, len(indices), batch_size):
                b_pos = norm_pos[b:b+batch_size]
                b_sub = sub_dsis[b:b+batch_size]
                b_pred = model((b_pos, b_sub)).clip(0, 1)
                preds.append(b_pred)
            preds = torch.cat(preds, dim=0).cpu().numpy()

            depth_grid = np.full((H, W), np.nan, dtype=np.float32)
            idx_np = indices.cpu().numpy()
            for (r, c), p in zip(idx_np, preds):
                depth_grid[max(0, r-thicken_pixels):min(H, r+thicken_pixels+1),
                           max(0, c-thicken_pixels):min(W, c+thicken_pixels+1)] = p

            masked_depth = np.ma.masked_invalid(depth_grid)
            colored = cmap(masked_depth)[:, :, :3]
            colored_bgr = (colored * 255.0)[:, :, ::-1].astype(np.uint8)

            proc.stdin.write(colored_bgr.tobytes())
            frame_idx += 1

            if frame_idx % 50 == 0 or (target_num_frames and frame_idx == target_num_frames):
                print(f"  Processed {frame_idx}/{target_num_frames or '?'} frames (confident pixels: {len(indices)}, depth range: {preds.min():.2f}-{preds.max():.2f})")

    proc.stdin.close()
    proc.wait()

    file_size_mb = os.path.getsize(out_video_path) / (1024 * 1024)
    print(f"[DERD-Net] Video successfully written: {out_video_path} ({file_size_mb:.2f} MB)")

    if not args.no_discord:
        send_video_to_discord(
            repo_root=repo_root,
            video_path=out_video_path,
            model_name="DERD-Net (PixelwiseConvGRU)",
            dataset_name=f"{capture_short} (Event)",
            title=f"🎬 Benchmark: DERD-Net Model ({capture_short})",
            message=f"Monocular Event-to-Depth inference with true Disparity Space Image ray back-projection ({frame_idx} frames @ 30 FPS). Native jet colormap with depth-scaled edge features."
        )

if __name__ == "__main__":
    main()
