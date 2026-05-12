# Mobile Manipulation Protocol

This document describes the **mobile manipulation** track (semi-finals). The platform is a wheeled bimanual robot with two arms, a chassis, a lift, and a pan-tilt head. The protocol is **different from the tabletop track** — keys are UPPERCASE and additional fields cover chassis / lift / head.

> Your policy server must accept the observation format and return the action format described below. The evaluation platform supplies the client that talks to your server.

---

## Observation (client &rarr; server)

The client sends a flat msgpack dict (no nested `state` / `views`):

```python
{
    # ── Cameras ─────────────────────────────────────────────
    "CAMERA_LEFT":  np.ndarray,   # (H, W, 3) uint8 RGB, left wrist camera
    "CAMERA_FRONT": np.ndarray,   # (H, W, 3) uint8 RGB, front camera
    "CAMERA_RIGHT": np.ndarray,   # (H, W, 3) uint8 RGB, right wrist camera

    # ── Arm end-effector state (euler) ──────────────────────
    "ACTION_FOLLOW1_POS": [x, y, z, roll, pitch, yaw, gripper],  # 7D, left
    "ACTION_FOLLOW2_POS": [x, y, z, roll, pitch, yaw, gripper],  # 7D, right

    # ── Arm joint efforts ───────────────────────────────────
    "ACTION_FOLLOW1_JOINTS_CUR": [...],   # joint currents/efforts, left
    "ACTION_FOLLOW2_JOINTS_CUR": [...],   # joint currents/efforts, right

    # ── Mobile base ─────────────────────────────────────────
    "CAR_POSE":  [x, y, yaw],   # 3D, chassis pose RELATIVE to a virtual zero
    "LIFT":      [h],           # 1D, lift mechanism height (m)
    "HEAD_POS":  [yaw, pitch],  # 2D, head rotation

    # ── Instruction ─────────────────────────────────────────
    "INSTRUCTION": np.array([str], dtype=np.object_),
}
```

### Notes

- `CAR_POSE` is a **pose**, not a velocity. The virtual zero is fixed to the chassis pose at the moment the episode starts, so all subsequent values are relative to that starting frame.
- Cameras arrive as already-decoded RGB `np.ndarray` (different from tabletop, which sends base64 JPEG).
- Coordinate convention: `+x` forward, `+y` left, `+z` up.

---

## Action (server &rarr; client)

```python
{
    # ── Arm trajectories (full sequence is executed step-by-step) ──
    "FOLLOW1_POS": [[x, y, z, r, p, y, grip], ...],   # (T, 7), left arm
    "FOLLOW2_POS": [[x, y, z, r, p, y, grip], ...],   # (T, 7), right arm

    # ── Mobile base targets (ONLY the LAST step is executed) ───────
    "CAR_POSE_OUT": [[x, y, yaw], ...],                # (T, 3), chassis relative pose
    "LIFT_OUT":     [[h], ...],                        # (T, 1), lift target
    "HEAD_POS":     [[yaw, pitch], ...],               # (T, 2), head target
}
```

> [!CAUTION]
> All values must be Python lists (`.tolist()`), not numpy arrays.

### Execution semantics

| Field | Behavior |
|---|---|
| `FOLLOW1_POS` / `FOLLOW2_POS` | Client iterates the **full trajectory** and sends each step. Euler rotation is converted to quaternion before sending to the arms. Last element is gripper. |
| `CAR_POSE_OUT` | Only the **last frame** is used as the chassis goal (relative to the virtual zero). |
| `LIFT_OUT` | Only the **last frame** is used as the lift target height. |
| `HEAD_POS` | Only the **last frame** is used. Output the two elements in the same `[yaw, pitch]` order as the observation `HEAD_POS` field — no axis swapping needed. |

After arm execution, the client **waits up to 3s** for chassis and lift to reach their targets (with tolerance), then proceeds to the next observation step.

### Hard safety limits (enforced by the client)

| Limit | Value | Effect |
|---|---|---|
| `position_max_tolerance` | **0.60 m** | If the requested chassis displacement exceeds this, the client **rejects** the command. Plan accordingly: each chunk should move the base at most ~60 cm. |
| `position_tolerance` | 0.05 m | Used to decide whether the chassis has arrived. |
| `yaw_tolerance` | ~0.174 rad (~10°) | Used to decide whether the chassis has arrived. |
| `lift_tolerance` | 0.10 m | Used to decide whether the lift has arrived. |
| `max_wait_time` | 3.0 s | Maximum wait before moving on, even if not arrived. |

---

## Important: Pose vs. Velocity Mismatch

The mobile client sends the chassis state as a **pose** (`CAR_POSE = [x, y, yaw]`), but the training dataset records it as a **body-frame velocity** (`velocity_decomposed_odom = [vx, vy, vyaw]`). If your model is trained on the LeRobot dataset, you **must** bridge this gap on the server side.

### Input: pose &rarr; velocity (differentiation)

Differentiate consecutive `CAR_POSE` observations into body-frame velocity:

```python
def pose_to_velocity(current_pose, last_pose, dt=0.05):
    """current_pose, last_pose: [x, y, yaw] in the global (virtual-zero) frame."""
    if last_pose is None:
        return np.zeros(3, dtype=np.float32)

    dx     = current_pose[0] - last_pose[0]
    dy     = current_pose[1] - last_pose[1]
    dtheta = current_pose[2] - last_pose[2]
    # Wrap angle to [-pi, pi)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

    # Global -> body frame using last pose's yaw
    theta = last_pose[2]
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    vx   = ( dx * cos_t + dy * sin_t) / dt
    vy   = (-dx * sin_t + dy * cos_t) / dt
    vyaw = dtheta / dt

    return np.array([vx, vy, vyaw], dtype=np.float32)
```

Cache `last_pose` between inference calls in your policy state (see `reset()`).

### Output: velocity &rarr; pose (integration)

If your model predicts a body-frame velocity sequence, integrate it back into a pose trajectory before populating `CAR_POSE_OUT`:

```python
def integrate_velocity_to_pose(velocities, dt=0.05, start_pose=None):
    """velocities: (T, 3) body-frame [vx, vy, vyaw].
    Returns (T, 3) global-frame poses [x, y, yaw] starting from start_pose."""
    if start_pose is None:
        start_pose = np.zeros(3, dtype=np.float32)

    poses = np.zeros((len(velocities), 3), dtype=np.float32)
    x, y, yaw = start_pose

    for i, (vx, vy, vyaw) in enumerate(velocities):
        # Sign convention of the deployment controller: flip vy and vyaw.
        vy_body, vyaw_body = -vy, -vyaw
        cos_t, sin_t = np.cos(yaw), np.sin(yaw)
        x   += (vx * cos_t - vy_body * sin_t) * dt
        y   += (vx * sin_t + vy_body * cos_t) * dt
        yaw += vyaw_body * dt
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi   # wrap to [-pi, pi)
        poses[i] = [x, y, yaw]

    return poses
```

Pass the most recent `CAR_POSE` from the observation as `start_pose` so the integrated trajectory is anchored to where the robot actually is.

### Why `dt = 0.05`?

The training data was recorded at **20 Hz** (`dt = 1/20 = 0.05 s`). If you up-sample the action chunk (interpolation), divide `dt` by the same factor.

---

## State Dimension Reminder

Mobile manipulation tasks use the **62-D** state (vs. 56-D tabletop). The extra 6 dims map to:

| Index | Field | Dim | Description |
|---|---|---|---|
| 56–57 | `head_actions` | 2 | Head rotation (yaw, pitch) |
| 58 | `height` | 1 | Lift mechanism height |
| 59–61 | `velocity_decomposed_odom` | 3 | Chassis body-frame velocity (vx, vy, angular velocity) |

See the [main README](../README.md#data-fields-summary) and [DATASET_CARD.md](../DATASET_CARD.md) for the full layout.

---

## Reference Implementation

For a working server that handles all of the above (UPPERCASE key parsing, pose↔velocity conversion, 20D state assembly, integrated chassis trajectory output), see [`examples/openpi_mobile_example.py`](../examples/openpi_mobile_example.py).

The tabletop-only OpenPI example lives at [`examples/openpi_example.py`](../examples/openpi_example.py) — use the mobile one for this track.
