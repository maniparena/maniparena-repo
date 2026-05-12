<div align="center">

# ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation

[![arXiv](https://img.shields.io/badge/arXiv-2603.28545-b31b1b.svg)](https://arxiv.org/abs/2603.28545)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://maniparena.x2robot.com)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/ManipArena/maniparena-dataset)
[![Simulation](https://img.shields.io/badge/Simulation-ManipArena--Sim-green)](https://github.com/maniparena/maniparena-sim)
[![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey.svg)](LICENSE)
[![CVPR 2026](https://img.shields.io/badge/CVPR%202026-Embodied%20AI%20Workshop-purple)](https://embodied-ai.org/cvpr2026/)

[**Paper**](https://arxiv.org/abs/2603.28545) | [**Project Page**](https://maniparena.x2robot.com) | [**Dataset**](https://huggingface.co/datasets/ManipArena/maniparena-dataset) | [**Simulation**](https://github.com/maniparena/maniparena-sim)

</div>

**ManipArena** is a real-robot benchmark and competition for bimanual manipulation, hosted at the [CVPR 2026 Embodied AI Workshop](https://embodied-ai.org/cvpr2026/). It features **20 real-robot tasks** across three categories:

| Category | Tasks | Challenge |
|----------|:-----:|-----------|
| **Execution Reasoning** | 10 | The goal is clear; the challenge is precise motor execution |
| **Semantic Reasoning** | 5 | The execution is straightforward; the challenge is understanding *what* to do |
| **Mobile Manipulation** | 5 | Long-horizon navigation and manipulation |

Participants serve their model **remotely** via WebSocket — no robot hardware needed. The organizers' infrastructure handles all robot control, data collection, and scoring.

<p align="center">
  <img src="docs/task_overview.png" width="100%" alt="ManipArena Task Overview" />
</p>

---

## How It Works

```
Your Model Server (this repo)          ManipArena Evaluation Platform
┌─────────────────────────┐            ┌──────────────────────────┐
│  load_model()           │            │  Robot sends observation │
│  convert_input()        │◄──────────►│  (images + state + task) │
│  run_inference()        │  WebSocket │                          │
│  convert_output()       │  (msgpack) │  Robot executes actions  │
└─────────────────────────┘            └──────────────────────────┘
```

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Edit examples/my_policy.py (implement load_model / run_inference / convert_output)

# 3. Launch
python serve.py --checkpoint /path/to/ckpt --control-mode end_pose --port 8000
```

## Implement Your Policy

Subclass `ModelPolicy` and implement three methods:

```python
from maniparena.policy import ModelPolicy
from maniparena.utils import convert_observation_to_model_input, convert_model_output_to_action

class MyPolicy(ModelPolicy):
    def load_model(self, checkpoint_path, device):
        ...                             # return your model

    def run_inference(self, model_input):
        ...                             # return (T, 14) numpy array

    def convert_output(self, model_output):
        return convert_model_output_to_action(model_output, self.control_mode, self.action_horizon)
```

See [`examples/my_policy.py`](examples/my_policy.py) for the full template,
[`examples/pytorch_example.py`](examples/pytorch_example.py) for a PyTorch reference,
and [`examples/openpi_example.py`](examples/openpi_example.py) for a ready-to-run
[OpenPI](https://github.com/Physical-Intelligence/openpi) example.

## Dataset

Training data is hosted on [HuggingFace](https://huggingface.co/datasets/ManipArena/maniparena-dataset) in [LeRobot](https://github.com/huggingface/lerobot) format. Beyond standard end-effector trajectories, ManipArena provides **joint positions, velocities, currents, camera views, and mobile-base states** — giving participants the freedom to explore diverse input representations.

```
dataset/
    meta/tasks.jsonl                                  # task instructions
    data/chunk-000/episode_000000.parquet             # state + action
    videos/chunk-000/
        observation.images.faceImg/episode_000000.mp4 # front camera
        observation.images.leftImg/episode_000000.mp4 # left wrist camera
        observation.images.rightImg/episode_000000.mp4# right wrist camera
```

See [DATASET_CARD.md](DATASET_CARD.md) for the full dimension layout, task list, and usage examples.

### Data Fields (Summary)

Each episode provides multiple data types. You can choose which fields to use for your model.

Tabletop tasks = **56D**, Mobile Manipulation tasks = **62D** (56D + 6D mobile extras).

> [!IMPORTANT]
> The **mobile manipulation track (semi-finals)** uses a different protocol — UPPERCASE keys plus extra chassis / lift / head fields. See [`docs/mobile_manipulation.md`](docs/mobile_manipulation.md) for the full protocol, safety limits, and the required pose ↔ velocity conversion.

**End-effector (index 0–13, 14D):**

| Index | Field | Dim | Description |
|-------|-------|-----|-------------|
| 0–2 | `follow_left_ee_cartesian_pos` | 3 | Left arm position (x, y, z) |
| 3–5 | `follow_left_ee_rotation` | 3 | Left arm rotation (roll, pitch, yaw) |
| 6 | `follow_left_gripper` | 1 | Left gripper open/close |
| 7–9 | `follow_right_ee_cartesian_pos` | 3 | Right arm position (x, y, z) |
| 10–12 | `follow_right_ee_rotation` | 3 | Right arm rotation (roll, pitch, yaw) |
| 13 | `follow_right_gripper` | 1 | Right gripper open/close |

> Coordinate system: +x forward, +y left, +z up.

**Joint data (index 14–55, 42D):**

| Index | Field | Dim | Description |
|-------|-------|-----|-------------|
| 14–20 | `follow_left_arm_joint_pos` | 7 | Left arm joint positions (6 joints + gripper) |
| 21–27 | `follow_left_arm_joint_dev` | 7 | Left arm joint velocities (6 joints + gripper) |
| 28–34 | `follow_left_arm_joint_cur` | 7 | Left arm joint currents (6 joints + gripper) |
| 35–41 | `follow_right_arm_joint_pos` | 7 | Right arm joint positions (6 joints + gripper) |
| 42–48 | `follow_right_arm_joint_dev` | 7 | Right arm joint velocities (6 joints + gripper) |
| 49–55 | `follow_right_arm_joint_cur` | 7 | Right arm joint currents (6 joints + gripper) |

**Mobile manipulation extras (index 56–61, mobile tasks only, 6D):**

| Index | Field | Dim | Description |
|-------|-------|-----|-------------|
| 56–57 | `head_actions` | 2 | Head rotation (yaw, pitch) |
| 58 | `height` | 1 | Lift mechanism height |
| 59–61 | `velocity_decomposed_odom` | 3 | Chassis velocity (vx, vy, angular velocity) |

**Camera views:**

| Field | Description |
|-------|-------------|
| `observation.images.faceImg` | Front camera (third-person view) |
| `observation.images.leftImg` | Left wrist camera |
| `observation.images.rightImg` | Right wrist camera |

## Self-Check Before Submission

We provide scripts to validate your server **before** you submit.

### Step 1 — Ping & handshake

```bash
python scripts/mock_ping.py --uri ws://127.0.0.1:8000
```

### Step 2 — Request/response schema

```bash
python scripts/mock_schema_check.py --uri ws://127.0.0.1:8000
```

### Step 3 — Open-loop evaluation (recommended)

We recommend running open-loop evaluation before submission to visually check whether your model's predictions align with the ground truth trajectories.

```bash
python scripts/eval_openloop.py \
    --server ws://127.0.0.1:8000 \
    --dataset /path/to/lerobot_dataset \
    --episode 0 \
    --save-dir openloop_plots \
    --action-chunk 32
```

### Step 4 — Simulation testing (optional)

Test your policy in our simulation environment before running on the real robot:

> **[ManipArena-Sim](https://github.com/maniparena/maniparena-sim)** — Isaac Lab based simulation with 3 tabletop tasks, teleoperation, and closed-loop policy evaluation.

```bash
# In the ManipArena-Sim repo:
python scripts/eval.py --task sort_blocks --config configs/eval/robot.yaml
```

---

## Observation & Action Format

ManipArena supports two control modes: **`end_pose`** (end-effector) and **`joints`**. Set via `--control-mode` when launching.

### Observation (client &rarr; server)

```python
{
    "state": {
        "follow1_pos": [x, y, z, r, p, y, gripper],   # 7D, left arm end-effector
        "follow2_pos": [x, y, z, r, p, y, gripper],   # 7D, right arm end-effector
    },
    "views": {
        "camera_left":  "<base64 JPEG>",
        "camera_front": "<base64 JPEG>",
        "camera_right": "<base64 JPEG>",
    },
    "instruction": "Pick up the cup.",
}
```

### Action (server &rarr; client)

**`end_pose` mode:**

```python
{
    "follow1_pos": [[x, y, z, r, p, y, grip], ...],   # List[List[float]], T steps, left arm
    "follow2_pos": [[x, y, z, r, p, y, grip], ...],   # List[List[float]], T steps, right arm
}
```

**`joints` mode:**

```python
{
    "follow1_pos": [[j1, j2, j3, j4, j5, j6, grip], ...],  # List[List[float]], T steps, left arm
    "follow2_pos": [[j1, j2, j3, j4, j5, j6, grip], ...],  # List[List[float]], T steps, right arm
}
```

> [!CAUTION]
> Values **must** be Python lists (`.tolist()`), not numpy arrays. The client does `[current_pos] + actions` — numpy `+` silently broadcasts instead of concatenating, corrupting the trajectory.

## Protocol

```
Client connects → Server sends metadata (msgpack)
Loop:  Client sends observation (msgpack) → Server returns actions (msgpack)
```

Metadata example: `{"control_mode": "end_pose", "action_horizon": 50, "state_dim": 14}`

## CLI Arguments

```bash
python serve.py --help
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | (required) | Model checkpoint path |
| `--control-mode` | `end_pose` | `end_pose` or `joints` |
| `--action-horizon` | `50` | Action sequence length (T) |
| `--device` | `cuda:0` | `cuda:0`, `cpu`, etc. |
| `--port` | `8000` | Server port |
| `--host` | `0.0.0.0` | Server host |

## Project Structure

```
serve.py                   # ← start here: python serve.py --checkpoint ...
maniparena/                # core framework (do not modify)
    policy.py              #   ModelPolicy base class
    server.py              #   WebSocket server
    utils.py               #   observation/action conversion helpers
    launch.py              #   CLI entry point
examples/                  # participant code
    my_policy.py           #   ← edit this file
    pytorch_example.py     #   PyTorch reference
    openpi_example.py      #   OpenPI ready-to-run example
scripts/                   # self-check & evaluation tools
    mock_ping.py           #   Step 1: handshake check
    mock_schema_check.py   #   Step 2: schema validation
    mock_openloop_eval.py  #   Quick open-loop check
    eval_openloop.py       #   Full open-loop eval with plots
```

## Citation

If you find ManipArena useful in your research, please consider citing:

```bibtex
@article{maniparena2026,
    title={ManipArena: A Benchmark for Bimanual Manipulation},
    author={ManipArena Team},
    journal={arXiv preprint arXiv:2603.28545},
    year={2026},
    url={https://arxiv.org/abs/2603.28545},
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
