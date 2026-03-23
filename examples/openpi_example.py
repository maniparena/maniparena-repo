#!/usr/bin/env python3
"""OpenPI policy example — Bimanual 14D (end-effector).

Prerequisites:
    pip install openpi    # or install from the openpi repo

Usage:
    python serve.py \
        --checkpoint /path/to/openpi/checkpoints/step_10000 \
        --control-mode end_pose \
        --port 8000

    Replace `MyPolicy` import in serve.py or copy this file to examples/my_policy.py.

Set OPENPI_CONFIG_NAME below to match your trained OpenPI config.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict

import cv2
import numpy as np

from maniparena.policy import ModelPolicy

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────

OPENPI_CONFIG_NAME = "your_openpi_config_name"  # e.g. "pi0_your_task_ee"
DEFAULT_PROMPT = "pick up the banana"
ACTION_END_RATIO = 0.8  # keep first 80% of predicted actions


# ── OpenPI Camera key mapping ────────────────────────────────────

_CAM_MAP = {
    "camera_front": "observation.images.faceImg",
    "camera_left": "observation.images.leftImg",
    "camera_right": "observation.images.rightImg",
}


def _decode_image(v: Any) -> np.ndarray:
    """base64 JPEG string or numpy array → RGB uint8 ndarray."""
    if isinstance(v, np.ndarray):
        return v.astype(np.uint8) if v.dtype != np.uint8 else v
    raw = base64.b64decode(v) if isinstance(v, str) else bytes(v)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Policy ────────────────────────────────────────────────────────


class MyPolicy(ModelPolicy):
    """OpenPI policy adapter for ManipArena bimanual (14D EE)."""

    def load_model(self, checkpoint_path: str, device: str) -> Any:
        from openpi.policies import policy_config as pc
        from openpi.training import config as train_config

        cfg = train_config.get_config(OPENPI_CONFIG_NAME)
        policy = pc.create_trained_policy(
            cfg, checkpoint_path,
            default_prompt=DEFAULT_PROMPT,
            pytorch_device=device,
        )
        logger.info(f"OpenPI model loaded: config={OPENPI_CONFIG_NAME}")
        return policy

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """ManipArena observation → OpenPI observation dict."""
        state_dict = obs.get("state", {})
        f1 = np.asarray(state_dict.get("follow1_pos", np.zeros(7)), dtype=np.float32)[:7]
        f2 = np.asarray(state_dict.get("follow2_pos", np.zeros(7)), dtype=np.float32)[:7]
        # OpenPI state: [left_pos(3), left_euler(3), left_grip(1), right_pos(3), right_euler(3), right_grip(1)]
        state = np.concatenate([f1[:6], f1[6:7], f2[:6], f2[6:7]]).astype(np.float32)

        openpi_obs: Dict[str, Any] = {
            "observation.state": state,
            "prompt": obs.get("instruction", DEFAULT_PROMPT) or DEFAULT_PROMPT,
        }

        views = obs.get("views", {})
        for client_key, openpi_key in _CAM_MAP.items():
            raw = views.get(client_key)
            if raw is not None:
                openpi_obs[openpi_key] = _decode_image(raw)
            else:
                openpi_obs[openpi_key] = np.zeros((480, 640, 3), dtype=np.uint8)

        return openpi_obs

    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        result = self.model.infer(model_input)
        return np.asarray(result["actions"], dtype=np.float32)

    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        """OpenPI actions (T, 14) → ManipArena response dict.

        NOTE: values must be Python lists (.tolist()), not numpy arrays.
        """
        actions = model_output
        end_idx = max(2, int(ACTION_END_RATIO * actions.shape[0]))
        actions = actions[:end_idx]

        left = actions[:, :7]
        right = actions[:, 7:14]

        return {
            "follow1_pos": left.tolist(),
            "follow2_pos": right.tolist(),
        }
