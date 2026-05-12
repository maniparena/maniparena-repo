#!/usr/bin/env python3
"""OpenPI policy example — Mobile Manipulation (20D = 14D EE + 6D mobile extras).

This example targets the **mobile manipulation track** (semi-finals). The
observation/action keys are UPPERCASE and the policy must additionally output
chassis / lift / head trajectories. See docs/mobile_manipulation.md for the
full protocol and safety limits.

Key responsibilities of this adapter:

1. Detect mobile-protocol observations (UPPERCASE keys, decoded RGB arrays).
2. Convert `CAR_POSE` (pose) into body-frame velocity to match the training
   data convention `velocity_decomposed_odom = [vx, vy, vyaw]`. We keep the
   previous pose across inference calls in `self._last_car_pose`.
3. Convert the model's predicted velocity back into an absolute pose
   trajectory, anchored at the current observed chassis pose.

Prerequisites:
    pip install openpi    # or install from the openpi repo

Usage:
    Copy this file to examples/my_policy.py, then launch:

        python serve.py \\
            --checkpoint /path/to/openpi/checkpoints/step_10000 \\
            --control-mode end_pose \\
            --port 8000

Set OPENPI_CONFIG_NAME below to match your trained mobile-manipulation config.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np

from maniparena.policy import ModelPolicy

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────

OPENPI_CONFIG_NAME = "your_mobile_config_name"   # e.g. "pi0_mobile_pick_place"
DEFAULT_PROMPT = "pick up the bottle and place it on the shelf"
ACTION_END_RATIO = 0.8       # keep first 80% of predicted actions
CONTROL_DT = 0.05            # 20 Hz training data

# UPPERCASE camera keys (mobile protocol)
_CAM_MAP = {
    "CAMERA_FRONT": "observation.images.faceImg",
    "CAMERA_LEFT":  "observation.images.leftImg",
    "CAMERA_RIGHT": "observation.images.rightImg",
}


# ── Helpers ──────────────────────────────────────────────────────


def _decode_image(v: Any) -> np.ndarray:
    """Mobile cameras arrive as decoded RGB ndarrays; tolerate base64 fallback."""
    if isinstance(v, np.ndarray):
        return v.astype(np.uint8) if v.dtype != np.uint8 else v
    if isinstance(v, str):
        v = base64.b64decode(v)
    arr = np.frombuffer(v, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _pose_to_velocity(
    current_pose: np.ndarray,
    last_pose: Optional[np.ndarray],
    dt: float = CONTROL_DT,
) -> np.ndarray:
    """CAR_POSE (global frame) → body-frame velocity [vx, vy, vyaw]."""
    if last_pose is None:
        return np.zeros(3, dtype=np.float32)
    dx = current_pose[0] - last_pose[0]
    dy = current_pose[1] - last_pose[1]
    dtheta = current_pose[2] - last_pose[2]
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    theta = last_pose[2]
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array([
        ( dx * cos_t + dy * sin_t) / dt,
        (-dx * sin_t + dy * cos_t) / dt,
        dtheta / dt,
    ], dtype=np.float32)


def _integrate_velocity_to_pose(
    velocities: np.ndarray,
    start_pose: np.ndarray,
    dt: float = CONTROL_DT,
) -> np.ndarray:
    """Body-frame [vx, vy, vyaw] trajectory → global-frame [x, y, yaw] trajectory.

    Sign convention of the deployment controller: flip vy and vyaw.
    """
    poses = np.zeros((len(velocities), 3), dtype=np.float32)
    x, y, yaw = float(start_pose[0]), float(start_pose[1]), float(start_pose[2])
    for i in range(len(velocities)):
        vx = float(velocities[i, 0])
        vy = -float(velocities[i, 1])
        vyaw = -float(velocities[i, 2])
        cos_t, sin_t = np.cos(yaw), np.sin(yaw)
        x += (vx * cos_t - vy * sin_t) * dt
        y += (vx * sin_t + vy * cos_t) * dt
        yaw += vyaw * dt
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        poses[i] = [x, y, yaw]
    return poses


# ── Policy ────────────────────────────────────────────────────────


class MyPolicy(ModelPolicy):
    """OpenPI policy adapter for mobile manipulation (20D state/action)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_car_pose: Optional[np.ndarray] = None
        self._current_car_pose: Optional[np.ndarray] = None

    def reset(self):
        super().reset()
        self._last_car_pose = None
        self._current_car_pose = None

    def load_model(self, checkpoint_path: str, device: str) -> Any:
        from openpi.policies import policy_config as pc
        from openpi.training import config as train_config

        cfg = train_config.get_config(OPENPI_CONFIG_NAME)
        policy = pc.create_trained_policy(
            cfg, checkpoint_path,
            default_prompt=DEFAULT_PROMPT,
            pytorch_device=device,
        )
        logger.info(f"OpenPI mobile model loaded: config={OPENPI_CONFIG_NAME}")
        return policy

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Mobile observation (UPPERCASE) → OpenPI observation dict.

        Builds the 20D state vector that matches the LeRobot dataset layout:
            [left_pos(3), left_euler(3), left_grip(1),
             right_pos(3), right_euler(3), right_grip(1),
             head_actions(2), height(1), velocity_decomposed_odom(3)]
        """
        f1 = np.asarray(obs["ACTION_FOLLOW1_POS"], dtype=np.float32)[:7]
        f2 = np.asarray(obs["ACTION_FOLLOW2_POS"], dtype=np.float32)[:7]

        head = np.asarray(obs.get("HEAD_POS", [0.0, 0.0]), dtype=np.float32)[:2]
        lift = np.asarray(obs.get("LIFT", [0.0]), dtype=np.float32)[:1]

        # CAR_POSE (pose) → velocity (matches velocity_decomposed_odom in training data).
        car_pose = np.asarray(obs.get("CAR_POSE", [0.0, 0.0, 0.0]), dtype=np.float32)[:3]
        car_vel = _pose_to_velocity(car_pose, self._last_car_pose, dt=CONTROL_DT)
        self._last_car_pose = car_pose
        self._current_car_pose = car_pose

        state = np.concatenate([
            f1[:6], f1[6:7],   # left arm: pos+euler, gripper
            f2[:6], f2[6:7],   # right arm
            head, lift, car_vel,
        ]).astype(np.float32)

        prompt_raw = obs.get("INSTRUCTION", DEFAULT_PROMPT)
        if isinstance(prompt_raw, np.ndarray) and prompt_raw.size > 0:
            prompt = str(prompt_raw.flat[0])
        else:
            prompt = str(prompt_raw) if prompt_raw else DEFAULT_PROMPT

        openpi_obs: Dict[str, Any] = {
            "observation.state": state,
            "prompt": prompt,
        }
        for client_key, openpi_key in _CAM_MAP.items():
            raw = obs.get(client_key)
            if raw is not None:
                openpi_obs[openpi_key] = _decode_image(raw)
            else:
                openpi_obs[openpi_key] = np.zeros((480, 640, 3), dtype=np.uint8)

        return openpi_obs

    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        result = self.model.infer(model_input)
        return np.asarray(result["actions"], dtype=np.float32)

    def convert_output(self, actions: np.ndarray) -> Dict[str, Any]:
        """OpenPI actions (T, 20) → mobile response dict.

        Layout of the action vector (same as training data):
            [left_7D, right_7D, head_2D, lift_1D, base_velocity_3D]
        """
        end_idx = max(2, int(ACTION_END_RATIO * actions.shape[0]))
        actions = actions[:end_idx]

        left = actions[:, :7]
        right = actions[:, 7:14]
        head_traj = actions[:, 14:16]
        lift_traj = actions[:, 16:17]
        car_vel_traj = actions[:, 17:20]

        # Integrate predicted body-frame velocity back into absolute pose trajectory,
        # anchored at the current observed chassis pose.
        start_pose = self._current_car_pose if self._current_car_pose is not None \
            else np.zeros(3, dtype=np.float32)
        car_pose_traj = _integrate_velocity_to_pose(car_vel_traj, start_pose, dt=CONTROL_DT)

        return {
            "FOLLOW1_POS":  left.tolist(),
            "FOLLOW2_POS":  right.tolist(),
            "HEAD_POS":     head_traj.tolist(),
            "LIFT_OUT":     lift_traj.tolist(),
            "CAR_POSE_OUT": car_pose_traj.tolist(),
        }
