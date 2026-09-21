import os
import sys
import glob
import subprocess
import cv2
import numpy as np

def resolve_capture(name="example_dont_change", repo_root=None):
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    captures_dir = os.path.join(repo_root, "captures")

    if not name or name == "example_dont_change":
        return os.path.join(captures_dir, "example_dont_change"), "example_dont_change"

    if os.path.isdir(name):
        base = os.path.basename(os.path.normpath(name))
        short = base.split("__")[0]
        return os.path.abspath(name), short

    direct = os.path.join(captures_dir, name)
    if os.path.isdir(direct):
        short = name.split("__")[0]
        return direct, short

    matches = [
        d for d in os.listdir(captures_dir)
        if name.lower() in d.lower()
        and not d.startswith("ignore_")
        and os.path.isdir(os.path.join(captures_dir, d))
    ]
    if matches:
        matched = sorted(matches)[0]
        short = matched.split("__")[0]
        return os.path.join(captures_dir, matched), short

    raise FileNotFoundError(f"Capture directory not found matching '{name}' in {captures_dir}")

def get_capture_frame_count(capture_dir):
    event_dir = os.path.join(capture_dir, "event_frames")
    if os.path.isdir(event_dir):
        pngs = glob.glob(os.path.join(event_dir, "event_*.png"))
        if pngs:
            return len(pngs)
        npys = glob.glob(os.path.join(event_dir, "event_2ch_*.npy"))
        if npys:
            return len(npys)

    rgb_dir = os.path.join(capture_dir, "rgb")
    if os.path.isdir(rgb_dir):
        files = glob.glob(os.path.join(rgb_dir, "*.png"))
        if files:
            return len(files)

    depth_dir = os.path.join(capture_dir, "depth")
    if os.path.isdir(depth_dir):
        files = glob.glob(os.path.join(depth_dir, "*.png"))
        if files:
            return len(files)

    return None

def load_event_frame_from_png(fpath, target_shape=(360, 640)):
    img = cv2.imread(fpath)
    if img is None:
        return np.zeros((2, target_shape[0], target_shape[1]), dtype=np.float32)
    pos = ((img[:, :, 0] > 150) & (img[:, :, 2] < 100)).astype(np.float32)
    neg = ((img[:, :, 0] < 50) & (img[:, :, 1] < 50) & (img[:, :, 2] < 50)).astype(np.float32)
    pos_r = cv2.resize(pos, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
    neg_r = cv2.resize(neg, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
    return np.stack([pos_r, neg_r], axis=0)

def events_to_2channel_grid(ev, target_shape=(360, 640), orig_shape=(720, 1280), accumulation_ms=10.0):
    H, W = target_shape
    grid = np.zeros((2, H, W), dtype=np.float32)
    if ev is None or len(ev) == 0:
        return grid

    orig_H, orig_W = orig_shape
    scale_x = float(W) / float(orig_W)
    scale_y = float(H) / float(orig_H)

    xs = (ev["x"] * scale_x).astype(np.int32)
    ys = (ev["y"] * scale_y).astype(np.int32)
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if not np.any(valid):
        return grid

    xs = xs[valid]
    ys = ys[valid]
    pos = ev["p"][valid] > 0
    neg = ~pos

    np.add.at(grid[0], (ys[pos], xs[pos]), 1.0)
    np.add.at(grid[1], (ys[neg], xs[neg]), 1.0)
    grid /= max(accumulation_ms, 1.0)
    return grid

def ensure_video_under_size(video_path, max_mb=14.0):
    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if size_mb <= max_mb:
        return video_path

    print(f"[Compressor] Video {os.path.basename(video_path)} ({size_mb:.2f} MB) exceeds {max_mb} MB limit. Compressing for Discord...")
    # Probe duration
    try:
        probe = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        ).decode().strip()
        duration = float(probe)
    except Exception:
        duration = 30.0

    target_size_kb = int((max_mb - 2.5) * 8 * 1024)
    target_bitrate_kbps = max(150, int(target_size_kb / max(duration, 1.0)))

    tmp_out = video_path + ".compressed.mp4"
    compress_cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-c:v", "libx264",
        "-b:v", f"{target_bitrate_kbps}k",
        "-maxrate", f"{int(target_bitrate_kbps * 1.4)}k",
        "-bufsize", f"{int(target_bitrate_kbps * 2)}k",
        "-pix_fmt", "yuv420p",
        tmp_out
    ]
    subprocess.run(compress_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(tmp_out, video_path)
    new_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"[Compressor] Compressed {os.path.basename(video_path)}: {size_mb:.2f} MB -> {new_mb:.2f} MB")
    return video_path

def send_video_to_discord(repo_root, video_path, model_name, dataset_name, title=None, message=None, dry_run=False):
    video_path = ensure_video_under_size(video_path, max_mb=12.0)

    send_script = os.path.join(repo_root, "update-bot", "send-discord-video", "scripts", "send_video.py")
    if not os.path.exists(send_script):
        send_script = "/home/rohan908/.gemini/config/skills/send-discord-video/scripts/send_video.py"

    cmd = [
        sys.executable, send_script,
        "--video", video_path,
        "--model", model_name,
        "--dataset", dataset_name,
    ]
    if title:
        cmd.extend(["--title", title])
    if message:
        cmd.extend(["--message", message])
    if dry_run:
        cmd.append("--dry-run")

    print(f"[Discord Uploader] Sending {os.path.basename(video_path)} to Discord...")
    res = subprocess.run(cmd, check=True)
    return res.returncode == 0
