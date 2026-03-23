#!/usr/bin/env python3
"""Protocol/schema validator for ManipArena model server.

Sends a dummy observation and validates the response:
lowercase keys (`follow1_pos`, `follow2_pos`), List[List[float]] trajectories.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from typing import Any

import cv2
import msgpack
import msgpack_numpy as m
import numpy as np
import websockets

m.patch()


def _encode_jpeg_base64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("cv2.imencode(.jpg) failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.floating, np.integer))


def _validate_trajectory(obj: Any, key: str, dim: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, list):
        return [f"{key} must be List[List[float]], got {type(obj).__name__}"]
    if len(obj) == 0:
        return [f"{key} is empty"]
    for i, row in enumerate(obj):
        if not isinstance(row, list):
            errors.append(f"{key}[{i}] must be list, got {type(row).__name__}")
            continue
        if len(row) != dim:
            errors.append(f"{key}[{i}] dim mismatch: expected {dim}, got {len(row)}")
            continue
        bad = [j for j, v in enumerate(row) if not _is_number(v)]
        if bad:
            errors.append(f"{key}[{i}] contains non-numeric values at indices {bad[:5]}")
    return errors


def _build_payload() -> dict[str, Any]:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 1] = 180
    jpeg = _encode_jpeg_base64(img)
    return {
        "state": {
            "follow1_pos": [0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.5],
            "follow2_pos": [0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.5],
        },
        "views": {
            "camera_left": jpeg,
            "camera_front": jpeg,
            "camera_right": jpeg,
        },
        "instruction": "self-check",
    }


def _validate_response(resp: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("follow1_pos", "follow2_pos"):
        if key not in resp:
            errors.append(f"missing key: {key}")
        else:
            errors.extend(_validate_trajectory(resp[key], key, dim=7))
    return errors


async def run(uri: str, timeout_sec: float) -> int:
    print(f"[SCHEMA] Connecting to {uri}")
    async with websockets.connect(uri, open_timeout=timeout_sec) as ws:
        first = await asyncio.wait_for(ws.recv(), timeout=timeout_sec)
        if isinstance(first, str):
            print(f"[FAIL] expected metadata msgpack, got text: {first}")
            return 1
        metadata = msgpack.unpackb(first, raw=False)
        print("[INFO] metadata:", json.dumps(metadata, ensure_ascii=False, default=str))

        print("[CHECK] bimanual schema")
        try:
            payload = _build_payload()
            await ws.send(msgpack.packb(payload, use_bin_type=True))
            msg = await asyncio.wait_for(ws.recv(), timeout=timeout_sec)
            if isinstance(msg, str):
                raise RuntimeError(f"server returned text error: {msg}")
            resp = msgpack.unpackb(msg, raw=False)
            if not isinstance(resp, dict):
                raise RuntimeError(f"response is not dict: {type(resp).__name__}")
            errors = _validate_response(resp)
        except Exception as exc:
            print(f"[FAIL] {exc}")
            return 1

        if errors:
            print("[FAIL] Schema errors:")
            for e in errors:
                print(f"  - {e}")
            return 1

        print("[PASS] All schema checks passed.")
        return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate request/response schema against server.")
    p.add_argument("--uri", type=str, default="ws://127.0.0.1:8000", help="Server WebSocket URI.")
    p.add_argument("--timeout-sec", type=float, default=8.0, help="Connect/recv timeout in seconds.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    code = asyncio.run(run(uri=args.uri, timeout_sec=args.timeout_sec))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
