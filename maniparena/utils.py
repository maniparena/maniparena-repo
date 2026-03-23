"""Observation / action conversion helpers for Bimanual (14D) task."""

import base64
import logging
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ── Low-level converters ──────────────────────────────────────────


def to_numpy_1d(x: Any, *, name: str = "unknown") -> np.ndarray:
    """Coerce *x* to a 1-D float32 array (handles list, ndarray, msgpack dict)."""
    if x is None:
        return np.zeros((0,), dtype=np.float32)
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False).reshape(-1)
    if isinstance(x, dict) and "data" in x and "shape" in x:
        return np.array(x["data"], dtype=np.float32).reshape(tuple(x["shape"])).reshape(-1)
    if isinstance(x, (list, tuple)):
        return np.array(x, dtype=np.float32).reshape(-1)
    raise TypeError(f"{name}: unsupported type {type(x)}")


def decode_jpeg(v: Union[str, bytes, None], *, name: str = "unknown") -> Optional[np.ndarray]:
    """Decode base64-JPEG / raw bytes / numpy passthrough → RGB ndarray."""
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, str):
        try:
            v = base64.b64decode(v)
        except Exception:
            return None
    if not isinstance(v, (bytes, bytearray, memoryview)):
        return None
    import cv2
    buf = np.frombuffer(v, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to decode JPEG for {name}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def normalize_joints_to_7d(joints: np.ndarray, control_mode: str) -> np.ndarray:
    """8-D (7 joints + gripper) → 7-D (6 joints + gripper); 7-D pass-through."""
    if control_mode != "joints":
        return joints
    if joints.size == 8:
        return np.concatenate([joints[:6], joints[-1:]])
    if joints.size == 7:
        return joints
    raise ValueError(f"Expected 7D or 8D joints, got {joints.size}D")


# ── Instruction extraction ────────────────────────────────────────


def _extract_instruction(obs: Dict[str, Any]) -> str:
    """Extract text instruction from observation.

    Handles key variants (``instruction`` / ``INSTRUCTION`` / ``prompt``),
    plain strings, ``np.ndarray``, ``bytes``, and the msgpack_numpy
    serialized dict that appears when ``np.object_`` arrays are not
    properly deserialized on the wire.
    """
    raw = None
    for key in ("instruction", "INSTRUCTION", "prompt", "PROMPT"):
        raw = obs.get(key)
        if raw is not None:
            break
    if raw is None:
        return ""

    if isinstance(raw, np.ndarray):
        return str(raw.flat[0]) if raw.size > 0 else ""

    if isinstance(raw, dict):
        import pickle
        data = raw.get("data", raw.get(b"data"))
        if data is not None:
            try:
                arr = pickle.loads(data)
                if isinstance(arr, np.ndarray) and arr.size > 0:
                    return str(arr.flat[0])
                return str(arr)
            except Exception:
                pass
        return ""

    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")

    return str(raw) if raw else ""


# ── Observation → model input ─────────────────────────────────────


def convert_observation_to_model_input(
    obs: Dict[str, Any],
    control_mode: str,
    decode_images: bool = True,
) -> Dict[str, Any]:
    """Parse Bimanual observation into a model-friendly dict.

    Returns::

        {
            "left": ndarray,  "front": ndarray,  "right": ndarray,
            "state": ndarray (14,),
            "instruction": str,
        }
    """
    views = obs.get("views", {})
    images: Dict[str, Any] = {}
    for src, dst in (("camera_left", "left"), ("camera_front", "front"), ("camera_right", "right")):
        raw = views.get(src)
        if raw is not None:
            images[dst] = decode_jpeg(raw, name=src) if decode_images else raw

    state_dict = obs.get("state", {})
    if control_mode == "joints":
        f1 = to_numpy_1d(state_dict.get("follow1_joints", state_dict.get("follow1_pos")), name="follow1")
        f2 = to_numpy_1d(state_dict.get("follow2_joints", state_dict.get("follow2_pos")), name="follow2")
        f1 = normalize_joints_to_7d(f1, control_mode)
        f2 = normalize_joints_to_7d(f2, control_mode)
    else:
        f1 = to_numpy_1d(state_dict.get("follow1_pos"), name="follow1_pos")
        f2 = to_numpy_1d(state_dict.get("follow2_pos"), name="follow2_pos")

    if f1.size < 7 or f2.size < 7:
        raise ValueError(f"Expected ≥7D per arm, got left={f1.size}, right={f2.size}")

    state14 = np.concatenate([f1[:7], f2[:7]]).astype(np.float32)
    instruction = _extract_instruction(obs)

    return {**images, "state": state14, "instruction": instruction}


# ── Model output → server response ───────────────────────────────


def convert_model_output_to_action(
    actions: np.ndarray,
    control_mode: str,
    action_horizon: int,
) -> Dict[str, Any]:
    """Convert (T, 14) action array → Bimanual response dict (lowercase keys).

    IMPORTANT: all values are Python lists (``.tolist()``), NOT numpy arrays.
    The client does ``[current_pos] + actions``; if ``actions`` is numpy,
    ``+`` silently broadcasts instead of concatenating.
    """
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected shape (T, 14), got {actions.shape}")
    if actions.shape[0] != action_horizon:
        logger.warning(f"action_horizon mismatch: expected {action_horizon}, got {actions.shape[0]}")

    left = np.concatenate([actions[:, :6], actions[:, 6:7]], axis=1)
    right = np.concatenate([actions[:, 7:13], actions[:, 13:14]], axis=1)

    if control_mode == "joints":
        return {
            "follow1_joints": left.tolist(),
            "follow2_joints": right.tolist(),
            "follow1_pos": left.tolist(),
            "follow2_pos": right.tolist(),
        }
    return {
        "follow1_pos": left.tolist(),
        "follow2_pos": right.tolist(),
    }
