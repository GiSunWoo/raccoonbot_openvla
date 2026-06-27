# Raccoonbot_OpenVLA
2021741045_JeeSunWoo

# Raccoonbot_Openvla — Chess Pick-and-Place Extension

> **Course**: Physical AI
>

---

## Overview

This project extends the existing [Raccoonbot_Openvla](https://github.com/KWU-FAIR-LAB/Raccoonbot_Openvla) pipeline into a **chess piece pick-and-place task**.

The original pipeline only performed a simple grasping task on colored cylinders. This project simultaneously satisfies three dataset extension conditions:

| Item | Original | This Project |
|------|----------|--------------|
| Object | Colored cylinder | Chess pieces (rook, bishop) |
| Task | Grasp only | Pick-and-place + capture |
| Language command | `"grasp the {color} cylinder"` | `"Move the white rook from a2 to a3."` |

---

## Changes

### 1. Dataset Extension
- **New objects**: White rook, black bishop — on a 3×3 mini chessboard (usable squares: a2–c3)
- **New task**: Chess-rule-based piece movement + capture (126 scenarios total, 4 categories)
- **New language commands**: Natural-language commands based on chess algebraic notation
- **Camera adjustment**: Moved closer to a top-view to resolve the issue of front pieces occluding rear pieces

### 2. Code Improvement ④ — Pitch-Aware Inverse Kinematics (`raccoon_env_pitch_real.py`)
- The original `raccoon_env.py` fixes Joint 4 horizontally (`th4 = -(th2+th3) - 90°`)
- Added a pitch parameter:
  `th4 = -(th2+th3) - 90° - pitch`
- Enables top-down grasping when horizontal approach is blocked
- `lockp(pitch_deg)`: Maintains pitch throughout the lift / sweep / place stages
- J4 joint limit (±105°) applied within the IK solver

### 3. Code Improvement ⑥ — Visualization (`chess_eval_plot.py`)
- Per-category success-rate bar chart
- Per-scenario success-rate heatmap
- Place error box plot (successful episodes)
- Episode rollout GIF (consistent frame capture across VLA and rule-based segments)

---

## Scenario Categories

| Category | Description | # of Scenarios |
|----------|-------------|----------------|
| Cat 1 | General pick-and-place (no occlusion, no capture) | 80 |
| Cat 2 | Top-down pitch required (horizontal approach blocked) | 20 |
| Cat 3 | Capture → graveyard (no pitch needed) | 23 |
| Cat 4 | Capture → graveyard + pitch required | 3 |

---

## Demos

### Cat 1 — Rook Pick-and-Place
![cat1_rook](figures/cat1_rook.gif)
<img width="256" height="256" alt="4" src="https://github.com/user-attachments/assets/6ce58056-8d37-419b-9059-b8da8e387419" />

### Cat 1 — Bishop Pick-and-Place
![cat1_bishop](figures/cat1_bishop.gif)
<img width="256" height="256" alt="20" src="https://github.com/user-attachments/assets/b31a9c53-3a22-41f5-9430-57d23ce8ef56" />

### Cat 2 — Bishop Top-down (Pitch)
![cat2_bishop](figures/cat2_bishop.gif)
<img width="256" height="256" alt="2" src="https://github.com/user-attachments/assets/d22ca1ef-f55d-40ed-9bb2-e4fbffb1420c" />

### Cat 3 — Capture
![cat3_capture](figures/cat3_capture.gif)
<img width="256" height="256" alt="0" src="https://github.com/user-attachments/assets/1405b9f3-8323-4f69-92cb-12e3e1c9ed66" />

---

## Evaluation Results

126 scenarios in total, 10 rollouts each (1,260 episodes total).

| Category | Description | # of Scenarios | Success / Total | Success Rate |
|----------|-------------|----------------|-----------------|--------------|
| Cat 1 | General pick-and-place | 80 | 725 / 800 | 90.6% |
| Cat 2 | Pitch top-down | 20 | 149 / 200 | 74.5% |
| Cat 3 | Capture | 23 | 196 / 230 | 85.2% |
| Cat 4 | Capture + pitch | 3 | 20 / 30 | 66.7% |
| **Total** | | **126** | **1090 / 1260** | **86.5%** |

<img width="1200" height="750" alt="image" src="https://github.com/user-attachments/assets/260068a0-83f7-48fa-a2b2-c2c3be231de6" />

<img width="1200" height="750" alt="place_err_boxplot" src="https://github.com/user-attachments/assets/ae3f1f9a-1664-4aae-8ef0-b235c72e567e" />

<img width="1401" height="730" alt="image" src="https://github.com/user-attachments/assets/cc9d203b-9ad6-4a77-940b-2c93216a1e06" />

---

## How to Run

### 1. Generate the Dataset
```bash
cd Mujoco/
python chess_pick_place_dataset.py
```

### 2. Build RLDS / TFDS
```bash
cd Mujoco/rlds_dataset_builder/raccoon_pick_place
tfds build --overwrite
```

### 3. Fine-tuning
```bash
cd openvla/
WANDB_MODE=disabled torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /data/Raccoonbot_Openvla/tensorflow_datasets \
  --dataset_name raccoon_chess \
  --lora_rank 32 --batch_size 8 --grad_accumulation_steps 4 \
  --learning_rate 5e-4 --max_steps 30000 --save_steps 10000 \
  --run_id_note chess-top-k2
```

### 4. Run the OpenVLA Server
```bash
cd openvla/
CUDA_VISIBLE_DEVICES=0 python openvla_server.py \
  --model_path openvla-runs/<checkpoint> \
  --default-unnorm-key raccoon_chess \
  --host 0.0.0.0 --port 8000
```

### 5. Evaluation
```bash
cd Mujoco/
python chess_eval.py \
  --server_url http://127.0.0.1:8000 \
  --xml_path B_Raccoon_chess_board_col.xml \
  --n_per 3 --output_dir chess_eval_out
```

### 6. Visualization
```bash
python chess_eval_plot.py \
  --summary chess_eval_out/eval_summary.json \
  --out_dir chess_eval_out
```

---

## Training Information

- **Base model**: OpenVLA-7B
- **Training method**: LoRA fine-tuning (rank 32)
- **Dataset**: 1,008 episodes / 63,805 transitions
- **Training steps**: 30,000
- **Hardware**: A100 80GB
- **Wall-clock time**: ~19 hours

---

## Repository Structure

```
Raccoonbot_Openvla/
├── README.md
├── report.pdf
├── logs/
│   └── train_log.txt
└── Mujoco/
    ├── raccoon_env_pitch_real.py
    ├── chess_pick_place_dataset.py
    ├── chess_eval.py
    ├── chess_eval_plot.py
    └── B_Raccoon_chess_board_col.xml
```

---
