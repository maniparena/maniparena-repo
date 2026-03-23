#!/usr/bin/env python3
"""Open-loop evaluation: replay LeRobot episodes through a running server, plot pred vs gt.

Usage:
    # Start your server first, then:
    python scripts/eval_openloop.py \
        --server ws://localhost:8000 \
        --dataset /path/to/lerobot_dataset \
        --episode 0 \
        --save-dir openloop_plots

Expected dataset layout (LeRobot):
    <dataset>/
        meta/tasks.jsonl              (optional, for task text)
        data/chunk-000/episode_000000.parquet
        videos/chunk-000/
            observation.images.faceImg/episode_000000.mp4
            observation.images.leftImg/episode_000000.mp4
            observation.images.rightImg/episode_000000.mp4
"""

from __future__ import annotations

import argparse
import base64
import json
import os

import av
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import msgpack
import msgpack_numpy as m
import numpy as np
import pandas as pd
import websockets.sync.client

m.patch()


# ── Data loading ──────────────────────────────────────────────────


def _read_task_text(dataset_root: str) -> str:
    path = os.path.join(dataset_root, "meta", "tasks.jsonl")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    return first.get("task", "")


def _load_episode(dataset_root: str, episode_idx: int) -> pd.DataFrame:
    chunk = episode_idx // 1000
    path = os.path.join(dataset_root, "data", f"chunk-{chunk:03d}", f"episode_{episode_idx:06d}.parquet")
    return pd.read_parquet(path)


def _load_video_frames(dataset_root: str, episode_idx: int, cam_key: str) -> list[np.ndarray]:
    chunk = episode_idx // 1000
    fname = f"episode_{episode_idx:06d}.mp4"
    chunk_dir = os.path.join(dataset_root, "videos", f"chunk-{chunk:03d}")
    path = os.path.join(chunk_dir, cam_key, fname)
    if not os.path.exists(path):
        # Try short name: observation.images.faceImg → faceImg
        short = cam_key.rsplit(".", 1)[-1] if "." in cam_key else cam_key
        path = os.path.join(chunk_dir, short, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {cam_key} under {chunk_dir}")

    container = av.open(path)
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    container.close()
    return frames


def _encode_jpeg(image_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ── Plotting ──────────────────────────────────────────────────────


def _plot(gt: np.ndarray, pred: np.ndarray, save_path: str, tag: str):
    dim = gt.shape[1]
    fig = plt.figure(figsize=(12, 3 * dim))
    for i in range(dim):
        ax = fig.add_subplot(dim, 1, i + 1)
        ax.plot(gt[:, i], label="Ground Truth", color="blue", linewidth=1.0)
        ax.plot(pred[:, i], label="Prediction", color="orange", linewidth=1.0)
        ax.set_title(f"{tag} - Dim {i + 1}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"[SAVED] {save_path}")


# ── Evaluation ────────────────────────────────────────────────────


def _parse_response(resp: dict) -> np.ndarray:
    """Server response → (T, 14) array."""
    for lk, rk in [("follow1_pos", "follow2_pos"), ("follow1_joints", "follow2_joints")]:
        if lk in resp and rk in resp:
            left = np.asarray(resp[lk], dtype=np.float32)
            right = np.asarray(resp[rk], dtype=np.float32)
            return np.concatenate([left, right], axis=1)
    raise RuntimeError(f"Unexpected response keys: {list(resp.keys())}")


def run(args: argparse.Namespace):
    print(f"[1/4] Loading episode {args.episode} from {args.dataset} ...")
    df = _load_episode(args.dataset, args.episode)
    task_text = args.instruction or _read_task_text(args.dataset)
    print(f"       Task: {task_text!r}  Frames: {len(df)}")

    print("       Loading videos ...")
    front = _load_video_frames(args.dataset, args.episode, "observation.images.faceImg")
    left = _load_video_frames(args.dataset, args.episode, "observation.images.leftImg")
    right = _load_video_frames(args.dataset, args.episode, "observation.images.rightImg")
    num_frames = min(len(df), len(front), len(left), len(right))
    if args.max_steps > 0:
        num_frames = min(num_frames, args.max_steps)
    print(f"       Usable frames: {num_frames}")

    print(f"\n[2/4] Connecting to {args.server} ...")
    ws = websockets.sync.client.connect(args.server, max_size=None, close_timeout=10)
    meta = msgpack.unpackb(ws.recv(), raw=False)
    print(f"       metadata = {meta}")

    print(f"\n[3/4] Running open-loop (chunk={args.action_chunk}) ...")
    gt_all, pred_all = [], []
    idx = 0
    infer_count = 0

    while idx < num_frames:
        state = np.array(df["observation.state"].iloc[idx], dtype=np.float32)
        payload = {
            "state": {
                "follow1_pos": state[:7].tolist(),
                "follow2_pos": state[7:14].tolist(),
            },
            "views": {
                "camera_front": _encode_jpeg(front[idx]),
                "camera_left": _encode_jpeg(left[idx]),
                "camera_right": _encode_jpeg(right[idx]),
            },
            "instruction": task_text,
        }

        ws.send(msgpack.packb(payload, use_bin_type=True))
        resp = msgpack.unpackb(ws.recv(), raw=False)
        if isinstance(resp, str):
            raise RuntimeError(f"Server error at step {idx}: {resp}")

        pred_chunk = _parse_response(resp)
        use_len = min(args.action_chunk, pred_chunk.shape[0], num_frames - idx)

        for k in range(use_len):
            gt_action = np.array(df["action"].iloc[idx + k], dtype=np.float32)
            D = min(pred_chunk.shape[1], gt_action.shape[0])
            gt_all.append(gt_action[:D])
            pred_all.append(pred_chunk[k, :D])

        infer_count += 1
        print(f"       infer #{infer_count}: idx={idx}, chunk={use_len}, total={len(pred_all)}/{num_frames}")
        idx += use_len

    ws.close()

    gt_arr = np.asarray(gt_all, dtype=np.float32)
    pred_arr = np.asarray(pred_all, dtype=np.float32)

    print(f"\n[4/4] Saving results ({infer_count} inferences, {len(gt_arr)} steps) ...")
    stem = os.path.join(args.save_dir, f"{args.tag}_ep{args.episode:03d}")
    _plot(gt_arr, pred_arr, f"{stem}.jpg", args.tag)
    np.savez(f"{stem}.npz", gt=gt_arr, pred=pred_arr)
    print(f"[SAVED] {stem}.npz")


def main():
    p = argparse.ArgumentParser(description="Open-loop evaluation (LeRobot dataset)")
    p.add_argument("--server", default="ws://localhost:8000")
    p.add_argument("--dataset", required=True, help="LeRobot dataset root")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--save-dir", default="openloop_plots")
    p.add_argument("--tag", default="openloop")
    p.add_argument("--instruction", default="", help="Override task instruction")
    p.add_argument("--max-steps", type=int, default=0, help="0 = all frames")
    p.add_argument("--action-chunk", type=int, default=32, help="Steps per inference")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
