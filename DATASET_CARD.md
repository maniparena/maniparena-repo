---
license: apache-2.0
task_categories:
  - robotics
tags:
  - maniparena
  - bimanual
  - manipulation
  - lerobot
  - cvpr2026
pretty_name: ManipArena Dataset
gated: auto
extra_gated_heading: "Access ManipArena Dataset"
extra_gated_description: "Please provide the following information to access the ManipArena dataset. Your request will be approved automatically."
extra_gated_button_content: "Submit and get access"
extra_gated_fields:
  Full Name: text
  Organization / Affiliation: text
  Country: country
  Email: text
  I plan to use this dataset for:
    type: select
    options:
      - Competition participation
      - Academic research
      - Education
      - label: Other
        value: other
  I agree to use this dataset only for research and competition purposes: checkbox
---

# ManipArena Dataset

Training dataset for [**ManipArena**](https://maniparena.x2robot.com), a real-robot benchmark and competition for bimanual manipulation at the [CVPR 2026 Embodied AI Workshop](https://embodied-ai.org/cvpr2026/).

This dataset provides rich multi-modal demonstrations in [LeRobot](https://github.com/huggingface/lerobot) format, covering **20 real-robot tasks** and **3 simulation tasks**. Beyond standard end-effector trajectories, we provide **joint positions, velocities, currents, camera views, and mobile-base states** — giving participants the freedom to explore diverse input representations.

## Dataset Structure

```
maniparena-dataset/
├── real/
│   ├── execution_reasoning/        (10 tasks, ~5,000 episodes)
│   ├── semantic_reasoning/         (5 tasks, ~2,800 episodes)
│   └── mobile_manipulation/        (5 tasks, ~2,900 episodes)
└── sim/
    ├── press_button_in_order/      (60 episodes)
    ├── put_blocks_to_color/        (50 episodes)
    └── pick_fruits_into_basket/    (50 episodes)
```

Each task folder follows LeRobot format:

```
<task>/
    meta/info.json
    meta/tasks.jsonl
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/
        observation.images.faceImg/episode_000000.mp4
        observation.images.leftImg/episode_000000.mp4
        observation.images.rightImg/episode_000000.mp4
```

---

## Real Robot Data

Tabletop tasks (Execution Reasoning + Semantic Reasoning) have **56-dimensional** state/action vectors.
Mobile Manipulation tasks have **62-dimensional** state/action vectors (56D + 6D mobile extras).

### Dimension Layout

**End-effector (index 0–13, 14D):**

| Index | Key | Dim | Description |
|-------|-----|-----|-------------|
| 0–2 | `follow_left_ee_cartesian_pos` | 3 | Left arm position (x, y, z) |
| 3–5 | `follow_left_ee_rotation` | 3 | Left arm rotation (roll, pitch, yaw) |
| 6 | `follow_left_gripper` | 1 | Left gripper open/close |
| 7–9 | `follow_right_ee_cartesian_pos` | 3 | Right arm position (x, y, z) |
| 10–12 | `follow_right_ee_rotation` | 3 | Right arm rotation (roll, pitch, yaw) |
| 13 | `follow_right_gripper` | 1 | Right gripper open/close |

> Coordinate system: **+x** forward, **+y** left, **+z** up.

**Joint — left arm (index 14–34, 21D):**

| Index | Key | Dim | Description |
|-------|-----|-----|-------------|
| 14–20 | `follow_left_arm_joint_pos` | 7 | Left arm joint positions (6 joints + gripper) |
| 21–27 | `follow_left_arm_joint_dev` | 7 | Left arm joint velocities (6 joints + gripper) |
| 28–34 | `follow_left_arm_joint_cur` | 7 | Left arm joint currents (6 joints + gripper) |

**Joint — right arm (index 35–55, 21D):**

| Index | Key | Dim | Description |
|-------|-----|-----|-------------|
| 35–41 | `follow_right_arm_joint_pos` | 7 | Right arm joint positions (6 joints + gripper) |
| 42–48 | `follow_right_arm_joint_dev` | 7 | Right arm joint velocities (6 joints + gripper) |
| 49–55 | `follow_right_arm_joint_cur` | 7 | Right arm joint currents (6 joints + gripper) |

> The last element (index 20, 27, 34, 41, 48, 55) in each 7D joint group is the gripper value.

**Mobile manipulation extras (index 56–61, mobile tasks only, 6D):**

| Index | Key | Dim | Description |
|-------|-----|-----|-------------|
| 56–57 | `head_actions` | 2 | Head rotation (yaw, pitch) |
| 58 | `height` | 1 | Lift mechanism height |
| 59–61 | `velocity_decomposed_odom` | 3 | Chassis velocity (vx, vy, angular velocity) |

> **Tabletop tasks** = 56D (index 0–55).
> **Mobile Manipulation tasks** = 62D (index 0–61).

### Task List — Real Robot

**Execution Reasoning (10 tasks):**

| Task | Episodes | Key Challenge |
|------|----------|---------------|
| `arrange_cup_inverted_triangle` | 528 | Multi-object spatial planning |
| `put_spoon_to_bowl` | 525 | Precision grasping, varied shapes |
| `put_glasses_on_woodshelf` | 513 | Fragile object handling |
| `put_ring_onto_rod` | 517 | Sub-cm insertion precision |
| `put_items_into_drawer` | 510 | Multi-object coordination |
| `pick_items_into_basket` | 532 | Adaptive grasping |
| `pour_water_from_bottle` | 526 | Force control, liquid dynamics |
| `insert_wireline` | 530 | Contact-rich, mm-level accuracy |
| `put_stationery_in_case` | 390 | Multi-object organization |
| `put_blocks_to_color` | 451 | Color-zone matching |

**Semantic Reasoning (5 tasks):**

| Task | Episodes | Key Challenge |
|------|----------|---------------|
| `sort_headphone` | 515 | Recognize headphone type |
| `classify_items_as_shape` | 545 | Map objects to shape categories |
| `press_button_in_order` | 538 | Color-button mapping + sequence |
| `pair_up_items` | 540 | Match pairs by pattern |
| `pick_fruits_into_basket` | 645 | Fruit vs. non-fruit distinction |

**Mobile Manipulation (5 tasks):**

| Task | Episodes | Key Challenge |
|------|----------|---------------|
| `put_clothes_in_hamper` | 540 | Navigate + pick clothes |
| `hang_up_picture` | 576 | Navigate to wall + hang |
| `organize_shoes` | 595 | Navigate + arrange on rack |
| `put_bottle_on_woodshelf` | 630 | Navigate to shelf + place |
| `take_and_set_tableware` | 531 | Navigate + set table |

---

## Simulation Data (28D)

Simulation demonstrations contain **28-dimensional** `observation.state` and `action` vectors, combining end-effector (14D) and joint (14D) data from the same trajectories.

### Dimension Layout

**End-effector (index 0–13):**

| Index | Key | Dim | Description |
|-------|-----|-----|-------------|
| 0–2 | `ee_left_xyz` | 3 | Left arm EE position (x, y, z) |
| 3–5 | `ee_left_rpy` | 3 | Left arm EE rotation (roll, pitch, yaw) |
| 6 | `ee_left_gripper` | 1 | Left gripper |
| 7–9 | `ee_right_xyz` | 3 | Right arm EE position (x, y, z) |
| 10–12 | `ee_right_rpy` | 3 | Right arm EE rotation (roll, pitch, yaw) |
| 13 | `ee_right_gripper` | 1 | Right gripper |

**Joint (index 14–27):**

| Index | Key | Dim | Description |
|-------|-----|-----|-------------|
| 14–19 | `joint_left_pos` | 6 | Left arm joint positions |
| 20 | `joint_left_gripper` | 1 | Left joint gripper |
| 21–26 | `joint_right_pos` | 6 | Right arm joint positions |
| 27 | `joint_right_gripper` | 1 | Right joint gripper |

> The first 14 dimensions (EE) are directly compatible with real-robot index 0–13.

### Task List — Simulation

| Task | Episodes | Real-robot Counterpart |
|------|----------|----------------------|
| `press_button_in_order` | 60 | press_button_in_order |
| `put_blocks_to_color` | 50 | put_blocks_to_color |
| `pick_fruits_into_basket` | 50 | pick_fruits_into_basket |

---

## Camera Views

All tasks include **3 synchronized camera streams** at 480×640 resolution:

| Camera | Key | Description |
|--------|-----|-------------|
| Front | `observation.images.faceImg` | Third-person overhead view |
| Left wrist | `observation.images.leftImg` | Left arm wrist-mounted camera |
| Right wrist | `observation.images.rightImg` | Right arm wrist-mounted camera |

## Recording Frequency

All data is recorded at **20 Hz**.

## Quick Usage

```python
import pandas as pd
import numpy as np

# Load one episode
df = pd.read_parquet("real/execution_reasoning/put_blocks_to_color/data/chunk-000/episode_000000.parquet")

state = np.stack(df["observation.state"].tolist())  # (T, 56) for real, (T, 28) for sim
action = np.stack(df["action"].tolist())

# EE data (first 14 dims — same layout for real and sim)
ee_left = state[:, 0:7]    # xyz(3) + rpy(3) + gripper(1)
ee_right = state[:, 7:14]

# Joint data (real: index 14–51, sim: index 14–27)
left_joint_pos = state[:, 14:20]   # 6 joint positions
right_joint_pos = state[:, 33:39]  # real only (sim uses 21:27)
```

## Citation

```bibtex
@misc{maniparena2026,
    title={ManipArena: A Benchmark for Bimanual Manipulation},
    year={2026},
    url={https://maniparena.x2robot.com},
}
```

## License

Apache License 2.0
