#!/usr/bin/env python3
"""Ping/handshake checker for ManipArena model server.

Checks:
1) WebSocket connect succeeds.
2) First server frame (metadata) is received.
3) Required metadata fields exist and have expected types.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import msgpack
import msgpack_numpy as m
import websockets

m.patch()


def _validate_metadata(meta: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    required = {
        "control_mode": str,
        "action_horizon": int,
        "state_dim": int,
    }
    for key, typ in required.items():
        if key not in meta:
            errors.append(f"missing key: {key}")
            continue
        if not isinstance(meta[key], typ):
            errors.append(
                f"invalid type for {key}: expected {typ.__name__}, got {type(meta[key]).__name__}"
            )

    if "control_mode" in meta and isinstance(meta["control_mode"], str):
        if meta["control_mode"] not in {"joints", "end_pose"}:
            errors.append(
                "control_mode should be one of joints/end_pose"
            )
    if "action_horizon" in meta and isinstance(meta["action_horizon"], int):
        if meta["action_horizon"] <= 0:
            errors.append("action_horizon must be > 0")
    if "state_dim" in meta and isinstance(meta["state_dim"], int):
        if meta["state_dim"] != 14:
            errors.append("state_dim should be 14")
    return errors


async def run(uri: str, timeout_sec: float) -> int:
    print(f"[PING] Connecting to {uri}")
    try:
        async with websockets.connect(uri, open_timeout=timeout_sec) as ws:
            first = await asyncio.wait_for(ws.recv(), timeout=timeout_sec)
            if isinstance(first, str):
                print("[FAIL] First frame is text, expected msgpack metadata.")
                print(f"       payload={first!r}")
                return 1

            try:
                meta = msgpack.unpackb(first, raw=False)
            except Exception as exc:  # noqa: BLE001
                print(f"[FAIL] Cannot decode metadata as msgpack: {exc}")
                return 1

            if not isinstance(meta, dict):
                print(f"[FAIL] Metadata is not a dict: {type(meta).__name__}")
                return 1

            errors = _validate_metadata(meta)
            if errors:
                print("[FAIL] Metadata check failed:")
                for e in errors:
                    print(f"  - {e}")
                print("[INFO] Received metadata:")
                print(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
                return 1

            print("[PASS] WebSocket handshake + metadata check succeeded.")
            print(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
            return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Connection or handshake error: {exc}")
        return 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check server ping/metadata handshake."
    )
    p.add_argument("--uri", type=str, default="ws://127.0.0.1:8000", help="Server WebSocket URI.")
    p.add_argument("--timeout-sec", type=float, default=5.0, help="Connect/recv timeout in seconds.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    code = asyncio.run(run(args.uri, args.timeout_sec))
    raise SystemExit(code)


if __name__ == "__main__":
    main()

