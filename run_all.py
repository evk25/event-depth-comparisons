#!/usr/bin/env python3
"""
Master Benchmark Runner
Runs depth estimation models across specified captures, generates output videos,
and posts evaluation videos and reports to Discord.
"""

import os
import sys
import time
import argparse
import subprocess

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

ALL_NEW_CAPTURES = [
    "goddard_straight",
    "institute_tapenav",
    "goddard_loop",
    "institute_training",
]

ALL_MODELS = [
    "depth_anything_v2",
    "events_depth",
    "evfly",
    "rpg_e2depth",
    "f3",
    "derd_net",
]

MODEL_SCRIPTS = {
    "depth_anything_v2": "runners/run_depth_anything_v2.py",
    "events_depth": "runners/run_events_depth.py",
    "evfly": "runners/run_evfly.py",
    "rpg_e2depth": "runners/run_rpg_e2depth.py",
    "f3": "runners/run_f3.py",
    "derd_net": "runners/run_derd_net.py",
}

def main():
    parser = argparse.ArgumentParser(description="Master Benchmark Runner")
    parser.add_argument("--captures", "-c", nargs="+", default=ALL_NEW_CAPTURES, help="Captures to run")
    parser.add_argument("--models", "-m", nargs="+", default=ALL_MODELS, help="Models to run")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per capture")
    parser.add_argument("--no-discord", action="store_true", help="Skip Discord upload")
    args = parser.parse_args()

    print("=" * 70)
    print("🎬 EVENT-TO-DEPTH BENCHMARK SUITE")
    print(f"Captures: {args.captures}")
    print(f"Models:   {args.models}")
    if args.max_frames:
        print(f"Max Frames: {args.max_frames}")
    print(f"Discord Upload: {not args.no_discord}")
    print("=" * 70)

    total_runs = len(args.captures) * len(args.models)
    current_run = 0
    start_total = time.time()
    results = []

    for cap in args.captures:
        for model_key in args.models:
            current_run += 1
            script = MODEL_SCRIPTS.get(model_key)
            if not script:
                print(f"[Error] Unknown model '{model_key}', skipping.")
                continue

            script_path = os.path.join(repo_root, script)
            cmd = [sys.executable, script_path, "--capture", cap]
            if args.max_frames:
                cmd.extend(["--max-frames", str(args.max_frames)])
            if args.no_discord:
                cmd.append("--no-discord")

            print(f"\n[{current_run}/{total_runs}] Starting {model_key} on {cap}...")
            t0 = time.time()
            try:
                subprocess.run(cmd, check=True)
                dur = time.time() - t0
                print(f"✓ Completed {model_key} on {cap} in {dur:.1f}s")
                results.append((cap, model_key, "SUCCESS", dur))
            except Exception as e:
                dur = time.time() - t0
                print(f"✗ Failed {model_key} on {cap} after {dur:.1f}s: {e}")
                results.append((cap, model_key, "FAILED", dur))

    total_time = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"BENCHMARK SUMMARY ({total_time:.1f}s total)")
    print("=" * 70)
    for cap, model, status, dur in results:
        sym = "✓" if status == "SUCCESS" else "✗"
        print(f"{sym} {cap:22s} | {model:20s} | {status:7s} ({dur:.1f}s)")
    print("=" * 70)

if __name__ == "__main__":
    main()
