# raccoonbot_openvla
2021741045_JeeSunWoo

# Raccoonbot_Openvla — 체스 픽앤플레이스 확장

> **과목**: Physical AI 
>

---

## 개요

기존 [Raccoonbot_Openvla](https://github.com/KWU-FAIR-LAB/Raccoonbot_Openvla) 파이프라인을 **체스 기물 픽앤플레이스 태스크**로 확장하였다.

기존 파이프라인은 컬러 실린더를 단순 파지하는 태스크만 수행하였다. 본 프로젝트는 세 가지 데이터셋 확장 조건을 동시에 충족한다:

| 항목 | 기존 | 본 프로젝트 |
|------|------|-------------|
| 오브젝트 | 컬러 실린더 | 체스 기물 (룩, 비숍) |
| 태스크 | 파지(Grasp)만 | 픽앤플레이스 + 캡처 |
| 언어 명령 | `"grasp the {color} cylinder"` | `"Move the white rook from a2 to a3."` |

---

## 변경 사항

### 1. 데이터셋 확장
- **새로운 오브젝트**: 흰색 룩, 검정 비숍 — 3×3 미니 체스판 위 (사용 가능 칸: a2~c3)
- **새로운 태스크**: 체스 규칙 기반 기물 이동 + 캡처 (총 126개 시나리오, 4개 카테고리)
- **새로운 언어 명령**: 체스 대수 표기법 기반 자연어 명령
- **카메라 조정**: 앞쪽 기물이 뒤쪽 기물을 가리는 문제 해결을 위해 top-view에 가깝게 조정

### 2. 코드 개선 ④ — Pitch 인식 역운동학 (`raccoon_env_pitch_real.py`)
- 원본 `raccoon_env.py`는 Joint 4를 수평으로 고정 (`th4 = -(th2+th3) - 90°`)
- Pitch 파라미터 추가:  
  `th4 = -(th2+th3) - 90° - pitch`
- 수평 접근이 차단된 경우 top-down 파지 가능
- `lockp(pitch_deg)`: 리프트/스윕/플레이스 전 과정에서 pitch 유지
- J4 관절 제한 (±105°) IK 솔버에서 적용

### 3. 코드 개선 ⑥ — 시각화 (`chess_eval_plot.py`)
- 카테고리별 성공률 막대그래프
- 시나리오별 성공률 히트맵
- Place error 박스플롯 (성공 에피소드)
- 에피소드 롤아웃 GIF (VLA·룰베이스 구간 일관된 프레임 캡처)

---

## 시나리오 카테고리

| 카테고리 | 설명 | 시나리오 수 |
|----------|------|-------------|
| Cat 1 | 일반 픽앤플레이스 (막힘 없음, 캡처 없음) | 80 |
| Cat 2 | Top-down pitch 필요 (수평 접근 차단) | 20 |
| Cat 3 | 캡처 → 묘지 (pitch 불필요) | 23 |
| Cat 4 | 캡처 → 묘지 + pitch 필요 | 3 |

---

## 데모

### Cat 1 — 룩 픽앤플레이스
![cat1_rook](figures/cat1_rook.gif)
<img width="256" height="256" alt="4" src="https://github.com/user-attachments/assets/6ce58056-8d37-419b-9059-b8da8e387419" />

### Cat 1 — 비숍 픽앤플레이스
![cat1_bishop](figures/cat1_bishop.gif)
<img width="256" height="256" alt="20" src="https://github.com/user-attachments/assets/81a781d6-1f56-4dab-88d8-7d3032ac457e" />


### Cat 2 — 비숍 Top-down (Pitch)
![cat2_bishop](figures/cat2_bishop.gif)
<img width="256" height="256" alt="2" src="https://github.com/user-attachments/assets/d22ca1ef-f55d-40ed-9bb2-e4fbffb1420c" />

### Cat 3 — 캡처
![cat3_capture](figures/cat3_capture.gif)

---

## 평가 결과

전체 126개 시나리오, 각 10회 롤아웃 (총 1,260 에피소드).

| 카테고리 | 설명 | 시나리오 수 | 성공/전체 | 성공률 |
|----------|------|-------------|-----------|--------|
| Cat 1 | 일반 픽앤플레이스 | 80 | 725 / 800 | 90.6% |
| Cat 2 | Pitch top-down | 20 | 149 / 200 | 74.5% |
| Cat 3 | 캡처 | 23 | 196 / 230 | 85.2% |
| Cat 4 | 캡처 + pitch | 3 | 20 / 30 | 66.7% |
| **전체** | | **126** | **1090 / 1260** | **86.5%** |

![success rate](figures/cat_success_rate.png)
<img width="1200" height="750" alt="image" src="https://github.com/user-attachments/assets/260068a0-83f7-48fa-a2b2-c2c3be231de6" />

성공 에피소드의 place error 분포 (카테고리별):

![place error](figures/place_err_boxplot.png)
<img width="1200" height="750" alt="cat_success_rate" src="https://github.com/user-
시작/목적지 셀별 성공률:

![cell heatmap](figures/scenario_success_rate.png)
<img width="1401" height="730" alt="image" src="https://github.com/user-attachments/assets/cc9d203b-9ad6-4a77-940b-2c93216a1e06" />

---

## 실행 방법

### 1. 데이터셋 생성
```bash
cd Mujoco/
python chess_pick_place_dataset.py
```

### 2. RLDS / TFDS 빌드
```bash
cd Mujoco/rlds_dataset_builder/raccoon_pick_place
tfds build --overwrite
```

### 3. 파인튜닝
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

### 4. OpenVLA 서버 실행
```bash
cd openvla/
CUDA_VISIBLE_DEVICES=0 python openvla_server.py \
  --model_path openvla-runs/<checkpoint> \
  --default-unnorm-key raccoon_chess \
  --host 0.0.0.0 --port 8000
```

### 5. 평가
```bash
cd Mujoco/
python chess_eval.py \
  --server_url http://127.0.0.1:8000 \
  --xml_path B_Raccoon_chess_board_col.xml \
  --n_per 3 --output_dir chess_eval_out
```

### 6. 시각화
```bash
python chess_eval_plot.py \
  --summary chess_eval_out/eval_summary.json \
  --out_dir chess_eval_out
```

---

## 학습 정보

- **베이스 모델**: OpenVLA-7B
- **학습 방법**: LoRA 파인튜닝 (rank 32)
- **데이터셋**: 1,008 에피소드 / 63,805 transition
- **학습 스텝**: 30,000
- **하드웨어**: A100 80GB
- **소요 시간**: 약 19시간

---

## 레포지토리 구조

```
Raccoonbot_Openvla/
├── README.md
├── report.pdf
├── logs/
│   └── train_log.txt
├── figures/
│   ├── cat1_rook.gif
│   ├── cat1_bishop.gif
│   ├── cat2_bishop.gif
│   └── cat3_capture.gif
└── Mujoco/
    ├── raccoon_env_pitch_real.py
    ├── chess_pick_place_dataset.py
    ├── chess_eval.py
    ├── chess_eval_plot.py
    └── B_Raccoon_chess_board_col.xml
```

---

