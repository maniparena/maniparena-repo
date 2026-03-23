#!/usr/bin/env python3
"""Open-loop evaluation helper for ManipArena server (LeRobot-only).

Features:
- Reads LeRobot episodes (parquet + videos) and queries the server step-by-step.
- Saves `pred`/`gt` arrays to `.npz` for analysis.
- Plotting is optional and disabled by default (`--enable-plots`).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import cv2
import msgpack
import msgpack_numpy as m
import numpy as np
import websockets

m.patch()


def _to_1d_list(x: Any) -> list[float]:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    return arr.tolist()


def _extract_arm_7d(step: dict[str, Any], arm: str) -> list[float] | None:
    # Direct 7D keys
    if arm == "left":
        direct = ["follow1_pos"]
    else:
        direct = ["follow2_pos"]
    for k in direct:
        if k in step:
            v = _to_1d_list(step[k])
            if len(v) >= 7:
                return v[:7]

    # Component keys commonly used in logs
    if arm == "left":
        part_keys = [
            "follow_left_ee_cartesian_pos",
            "follow_left_ee_rotation",
            "follow_left_gripper",
        ]
    else:
        part_keys = [
            "follow_right_ee_cartesian_pos",
            "follow_right_ee_rotation",
            "follow_right_gripper",
        ]
    if all(k in step for k in part_keys):
        out: list[float] = []
        for k in part_keys:
            val = step[k]
            if isinstance(val, list):
                out.extend(_to_1d_list(val))
            else:
                out.extend(_to_1d_list([val]))
        if len(out) >= 7:
            return out[:7]
    return None


def _encode_image(img_rgb: np.ndarray) -> str:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", img_bgr)
    if not ok:
        raise ValueError("cv2.imencode(.jpg) failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _read_video_rgb(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return np.zeros((0, 64, 64, 3), dtype=np.uint8)
    return np.asarray(frames, dtype=np.uint8)


def _find_video(sample_dir: Path, names: list[str]) -> np.ndarray:
    for n in names:
        p = sample_dir / n
        if p.exists():
            return _read_video_rgb(p)
    return np.zeros((0, 64, 64, 3), dtype=np.uint8)


def _resolve_lerobot_paths(data_dir: Path) -> tuple[Path | None, Path | None]:
    # data_dir can be either:
    # 1) <dataset_root>/data
    # 2) <dataset_root>
    if (data_dir / "chunk-000").exists():
        data_root = data_dir
        videos_root = data_dir.parent / "videos" if (data_dir.parent / "videos").exists() else None
        return data_root, videos_root
    if (data_dir / "data").exists() and (data_dir / "videos").exists():
        return data_dir / "data", data_dir / "videos"
    return None, None


def _discover_samples(data_dir: Path) -> list[str]:
    # LeRobot parquet episodes only: data/chunk-xxx/episode_xxxxxx.parquet
    data_root, _ = _resolve_lerobot_paths(data_dir)
    if data_root is not None:
        out: list[str] = []
        for chunk_dir in sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("chunk-")]):
            for ep_file in sorted(chunk_dir.glob("episode_*.parquet")):
                out.append(f"{chunk_dir.name}/{ep_file.stem}")
        if out:
            return out

    raise FileNotFoundError(
        "No LeRobot episodes found. Expected either "
        "<dataset_root>/data/chunk-*/episode_*.parquet or <dataset_root>/chunk-*/episode_*.parquet."
    )


def _load_lerobot_case(data_dir: Path, sample_token: str) -> dict[str, Any]:
    data_root, videos_root = _resolve_lerobot_paths(data_dir)
    if data_root is None:
        raise FileNotFoundError(f"Cannot locate LeRobot data root from {data_dir}")

    if "/" not in sample_token:
        raise ValueError(f"Invalid LeRobot sample token: {sample_token}")
    chunk_name, episode_name = sample_token.split("/", 1)
    parquet_path = data_root / chunk_name / f"{episode_name}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if "observation.state" not in df.columns:
        raise KeyError(f"{parquet_path} missing observation.state")
    if "action" not in df.columns:
        raise KeyError(f"{parquet_path} missing action")

    states = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in df["observation.state"].tolist()], axis=0)
    acts = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in df["action"].tolist()], axis=0)

    if states.shape[1] < 14:
        raise ValueError(f"observation.state dim < 14 in {parquet_path}: {states.shape}")
    if acts.shape[1] < 14:
        raise ValueError(f"action dim < 14 in {parquet_path}: {acts.shape}")

    follow1_state = states[:, :7].tolist()
    follow2_state = states[:, 7:14].tolist()
    follow1_gt = acts[:, :7].tolist()
    follow2_gt = acts[:, 7:14].tolist()

    # Videos (if available)
    front = left = right = np.zeros((0, 64, 64, 3), dtype=np.uint8)
    if videos_root is not None:
        front_p = videos_root / chunk_name / "observation.images.faceImg" / f"{episode_name}.mp4"
        left_p = videos_root / chunk_name / "observation.images.leftImg" / f"{episode_name}.mp4"
        right_p = videos_root / chunk_name / "observation.images.rightImg" / f"{episode_name}.mp4"
        if front_p.exists():
            front = _read_video_rgb(front_p)
        if left_p.exists():
            left = _read_video_rgb(left_p)
        if right_p.exists():
            right = _read_video_rgb(right_p)

    return {
        "follow1_state": follow1_state,
        "follow2_state": follow2_state,
        "follow1_gt": follow1_gt,
        "follow2_gt": follow2_gt,
        "camera_front": front,
        "camera_left": left,
        "camera_right": right,
    }


def _concat_lr(left: list[list[float]], right: list[list[float]]) -> np.ndarray:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    n = min(a.shape[0], b.shape[0])
    return np.concatenate([a[:n], b[:n]], axis=1)


def _extract_resp_lr(resp: dict[str, Any]) -> tuple[list[list[float]], list[list[float]]]:
    if "follow1_pos" in resp and "follow2_pos" in resp:
        return resp["follow1_pos"], resp["follow2_pos"]
    if "follow1_joints" in resp and "follow2_joints" in resp:
        return resp["follow1_joints"], resp["follow2_joints"]
    raise KeyError("response missing follow1/follow2 trajectory keys")


def _plot_openloop(pred: np.ndarray, gt: np.ndarray, save_stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    if pred.shape != gt.shape:
        n = min(pred.shape[0], gt.shape[0])
        pred = pred[:n]
        gt = gt[:n]
    if pred.size == 0:
        return

    dim = pred.shape[1]
    fig, axes = plt.subplots(dim, 1, figsize=(12, 3 * dim), sharex=True)
    if dim == 1:
        axes = [axes]
    xs = np.arange(pred.shape[0])
    for i, ax in enumerate(axes):
        ax.plot(xs, gt[:, i], label="GroundTruth", color="blue")
        ax.plot(xs, pred[:, i], label="Prediction", color="orange")
        ax.set_ylabel(f"dim{i}")
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("t")
    fig.tight_layout()
    jpg_path = f"{save_stem}.jpg"
    fig.savefig(jpg_path, dpi=180)
    plt.close(fig)
    print(f"[SAVE] plot -> {jpg_path}")


async def run(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    samples = _discover_samples(data_dir)
    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]
    if not samples:
        print("[FAIL] No samples found.")
        return 1

    print(f"[OPENLOOP] connect {args.uri}")
    async with websockets.connect(args.uri, open_timeout=args.timeout_sec) as ws:
        first = await asyncio.wait_for(ws.recv(), timeout=args.timeout_sec)
        if isinstance(first, str):
            print(f"[FAIL] expected metadata msgpack, got text: {first}")
            return 1
        metadata = msgpack.unpackb(first, raw=False)
        print(f"[INFO] metadata={metadata}")

        for sample in samples:
            print(f"\n[CASE] {sample}")
            try:
                case = _load_lerobot_case(data_dir, sample)
            except Exception as exc:  # noqa: BLE001
                print(f"[SKIP] load case failed: {exc}")
                continue

            front = case["camera_front"]
            left = case["camera_left"]
            right = case["camera_right"]
            follow1_state = case.get("follow1_state", case["follow1_gt"])
            follow2_state = case.get("follow2_state", case["follow2_gt"])

            n_state = min(len(follow1_state), len(follow2_state), len(case["follow1_gt"]), len(case["follow2_gt"]))
            n_cam = min(
                [x.shape[0] for x in (front, left, right) if x.shape[0] > 0] or [n_state]
            )
            n_steps = min(n_state, n_cam)
            if n_steps < 2:
                print("[SKIP] not enough aligned steps")
                continue

            idx = int(max(0, min(n_steps - 1, args.start_ratio * n_steps)))
            pred_chunks: list[np.ndarray] = []
            gt_chunks: list[np.ndarray] = []

            while idx < n_steps and sum(c.shape[0] for c in pred_chunks) < args.max_pred_steps:
                payload = {
                    "state": {
                        "follow1_pos": follow1_state[idx],
                        "follow2_pos": follow2_state[idx],
                    },
                    "views": {
                        "camera_front": _encode_image(front[idx]) if front.shape[0] > idx else None,
                        "camera_left": _encode_image(left[idx]) if left.shape[0] > idx else None,
                        "camera_right": _encode_image(right[idx]) if right.shape[0] > idx else None,
                    },
                    "instruction": args.instruction,
                }

                await ws.send(msgpack.packb(payload, use_bin_type=True))
                msg = await asyncio.wait_for(ws.recv(), timeout=args.timeout_sec)
                if isinstance(msg, str):
                    raise RuntimeError(f"server returned text: {msg}")
                resp = msgpack.unpackb(msg, raw=False)
                if not isinstance(resp, dict):
                    raise RuntimeError(f"response type invalid: {type(resp).__name__}")

                left_pred, right_pred = _extract_resp_lr(resp)
                pred = _concat_lr(left_pred, right_pred)
                if pred.shape[0] == 0:
                    break

                gt_l = case["follow1_gt"][idx : idx + pred.shape[0]]
                gt_r = case["follow2_gt"][idx : idx + pred.shape[0]]
                gt = _concat_lr(gt_l, gt_r)

                n = min(pred.shape[0], gt.shape[0], args.action_chunk)
                pred_chunks.append(pred[:n])
                gt_chunks.append(gt[:n])
                idx += args.action_chunk

            if not pred_chunks:
                print("[SKIP] no predictions collected")
                continue

            pred_all = np.concatenate(pred_chunks, axis=0)
            gt_all = np.concatenate(gt_chunks, axis=0)
            n = min(pred_all.shape[0], gt_all.shape[0])
            pred_all = pred_all[:n]
            gt_all = gt_all[:n]

            out_stem = save_dir / sample
            out_stem.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                f"{out_stem}.npz",
                pred=pred_all.astype(np.float32),
                gt=gt_all.astype(np.float32),
                sample=sample,
                metadata=metadata,
            )
            print(f"[SAVE] npz -> {out_stem}.npz  shape={pred_all.shape}")

            if args.enable_plots:
                try:
                    _plot_openloop(pred_all, gt_all, out_stem)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] plot failed (npz already saved): {exc}")

    print("\n[DONE] open-loop evaluation completed.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run open-loop eval against ManipArena server (LeRobot-only).")
    p.add_argument("--uri", type=str, default="ws://127.0.0.1:8000", help="Server WebSocket URI.")
    p.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help=(
            "LeRobot dataset root or its data dir. "
            "Expected: <root>/data/chunk-*/episode_*.parquet (+ optional <root>/videos)."
        ),
    )
    p.add_argument("--save-dir", type=str, required=True, help="Output directory for npz/jpg.")
    p.add_argument("--instruction", type=str, default="self-check task", help="Instruction text sent to server.")
    p.add_argument("--sample-limit", type=int, default=1, help="How many samples to evaluate (<=0 means all).")
    p.add_argument("--start-ratio", type=float, default=0.1, help="Start index ratio in each sample [0,1].")
    p.add_argument("--action-chunk", type=int, default=10, help="Stride/chunk size when advancing index.")
    p.add_argument("--max-pred-steps", type=int, default=200, help="Max predicted steps per sample.")
    p.add_argument("--timeout-sec", type=float, default=10.0, help="WebSocket recv timeout.")
    p.add_argument("--enable-plots", action="store_true", help="Generate jpg plots (default off, npz only).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    code = asyncio.run(run(args))
    raise SystemExit(code)


if __name__ == "__main__":
    main()

