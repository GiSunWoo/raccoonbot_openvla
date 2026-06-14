"""
chess_pick_place_dataset_top.py - 체스 픽앤플레이스 VLA 데이터 생성기
====================================================================

raccoon_grasp_multicolor_scene_dataset.py(실린더 수집기)의 출력 포맷/루프를
그대로 따르되, 체스 로직을 얹은 버전. chess_pick_place_pitch_test.py 의
검증된 동작(조건부 top-down pitch, lockp/lockh)을 PitchRealRaccoonEnv 로 재사용한다.

수집 대상:
  - 사용 칸: a2,b2,c2,a3,b3,c3 (1번 행은 배치/이동 모두 금지)
  - 기물: 흰색 룩 / 검정색 비숍 (둘 다 질량 0.04 로 고정)
  - 룩=직선, 비숍=대각선, 1행 이동 금지, 상대 기물 잡기(capture) 가능
  - 전체 126 시나리오 = 룩 이동 86 + 비숍 이동 40
      (1) 일반 픽앤플레이스(막힘X,캡처X) 80
      (2) 피치 필요(막힘O,캡처X)        20
      (3) 캡처+묘지(피치X)              23
      (4) 캡처+묘지+피치                3
  - 시나리오당 8 에피소드(성공 저장 기준) -> 총 1008

저장 방식(중요):
  - episode_XX 슬롯에는 '성공한 에피소드'만 저장된다.
  - 슬롯 0..N-1 중 비었거나 실패한 칸만, 랜덤성을 바꿔가며 성공할 때까지 재시도해 채운다.
  - 누적 시도 수는 N을 넘어 9,10,11... 로 가도 되며, N개가 다 찰 때까지 반복한다.
  - 성공이 하나도 없는데 give_up_no_success_after(기본 10)회를 넘기면 그 시나리오는 건너뛴다.
  - 무한 루프 방지용으로 시나리오당 총 시도 상한(max_attempts_per_scenario)도 둔다.

핵심 설계:
  - 캡처면 leg 2개: (잡힐말 -> 묘지) 다음에 (잡는말 src -> dst). 비캡처면 leg 1개.
  - 막힘은 항상 row3 타깃. 그 leg 만 top-down(45~50도), 나머지는 수평(0도).
  - 피치는 5번째 action 차원으로 기록: action = [x_m, y_m, z_m, pitch_deg, grip].
  - 모든 waypoint 를 _calc_inv_kinematics 로 사전검증 -> 안 닿으면 파라미터 재추출.
  - b2(정중앙)는 손목 J4 한계 때문에 오프셋이 좁고 grasp z 가 높게 자동 조정됨.
  - 실패(grasp 미스/기물 쓰러짐/충돌로 다른 기물 밀림)는 폐기하고 재생성.
  - [수정] 수평(pitch=0) grasp 는 TCP 목표를 베이스 쪽으로 HORIZ_GRASP_BACKOFF_M 만큼
    당겨, 손목이 기물에 닿지 않고 손끝으로 잡게 한다. top-down(pitch>0)에는 적용하지 않는다.
  - [수정] 비숍은 top-down(피치) 잡기에서 칼라(r 11.6mm, z 42.9~50.1mm)·어깨(r 8.5mm,
    z 21.4~28.6mm) 사이의 '목'(r 5.7~5.9mm, z 28.6~42.9mm)만 잡아야 한다. 오프셋 방식은
    창 양끝이 모두 실패 구간이라 간헐 실패했다. 그래서 비숍 피치 leg 는 grasp z 를
    BISHOP_TOPDOWN_GRASP_Z_RANGE(목 하단 절대 z) 로, 피치를 BISHOP_TOPDOWN_PITCH_GRID(45 고정)
    로 강제한다. 45 초과로 가면 접촉 밴드가 세로로 길어져 칼라에 더 잘 걸리므로 손해다.
    룩/수평 잡기에는 영향 없다.

출력 폴더 구조:
  chess_data/
    scenario_000_cat1_rook_a2-c2_rk_a2_bp_b2/
      episode_00/ frame_000000.png ... meta.json episode.gif
      episode_01/ ...
      ...(8개)
    ...(126개)
    scenarios_index.json

실행: python chess_pick_place_dataset.py
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import math
import json
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
import mujoco

from raccoon_env_pitch_real import PitchRealRaccoonEnv


# ======================================================================
# 상수
# ======================================================================
XML_PATH = "B_Raccoon_chess_board_col.xml"
CAMERA_NAME = "front_view"
IMAGE_SIZE = (256, 256)

# 보드 칸 좌표(월드, m) - XML cell_* geom 과 동일
CELLS = {
    "a1": (-0.06, 0.08), "b1": (0.00, 0.08), "c1": (0.06, 0.08),
    "a2": (-0.06, 0.14), "b2": (0.00, 0.14), "c2": (0.06, 0.14),
    "a3": (-0.06, 0.20), "b3": (0.00, 0.20), "c3": (0.06, 0.20),
}
USABLE = ["a2", "b2", "c2", "a3", "b3", "c3"]   # row-major(결정적)
COLS = {"a": 0, "b": 1, "c": 2}
ICOL = {0: "a", 1: "b", 2: "c"}
ROW_STEP = 0.06

# 묘지(표시 없음): 잡힌 기물을 보드 밖으로 - 가까운 쪽으로
GRAVEYARDS = {"L": (-0.12, 0.14), "R": (0.12, 0.14)}

# 기물
PIECE_BODIES = ("rook", "bishop")
PIECE_DESC = {"rook": "white rook", "bishop": "black bishop"}
PIECE_MASS = 0.04
SPAWN_Z = 0.02
EE_LINK_BODY = "Link4"

# 동작 파라미터
SAFE_Z_CM = 13.5
SPEED = 150                       # degree_to 내부에서 100%로 클립됨
HZ = 10
SETTLE_PER_ACTION = 0.8
GRASP_SETTLE_SEC = 0.2
HOLD_SEC = 0.2
INITIAL_SETTLE = 0.3
SQUEEZE = 0.20

# 들기/내려놓기 속도 배율.
# 0.4 = 약 2.5배 느림, 0.2 = 약 5배 느림.
# 너무 느리면 전체 생성 시간이 많이 늘어나므로 먼저 0.4 권장.
LIFT_PLACE_SPEED_SCALE = 1.0

# 다양성 범위(룩/수평 잡기 기본)
GRASP_Z_RANGE = (3.5, 4.5)

# [수정] 비숍 top-down(피치) 잡기는 목이 아니라 머리 위 '꼭지(상단 홈)'를 잡는다.
# 목 잡기는 바로 위 칼라(r 11.6mm)에 패드가 걸려 목(r 5.7mm)을 못 조이고 슬립이 났다.
# 비숍 충돌 실린더 기준 z 구간(mm):
#   어깨   r 8.5  : z 21.4 ~ 28.6
#   목     r 5.7~5.9 : z 28.6 ~ 42.9   (예전 목표, 칼라에 걸려 실패)
#   칼라   r 11.6 : z 42.9 ~ 50.1
#   머리   r ~10  : z 57.2 ~ 71.5
#   홈     r 3.6  : z 71.5 ~ 78.7      <- 새 목표(가장 얇음)
#   꼭지   r 4.7  : z 78.6 ~ 85.8      <- 홈 바로 위(간섭원이자 걸쇠)
# grasp z 는 홈 하단~중앙(7.4~7.5cm)에 둔다. 7.4 아래로 가면 패드가 머리(71.5mm)에
# 닿으므로 7.4 가 하한이다. 룩/수평 잡기에는 적용하지 않는다.
BISHOP_TOPDOWN_GRASP_Z_RANGE = (7.4, 7.5)

# [수정] 비숍 꼭지 잡기는 '낮은' 피치(15~25도)를 쓴다. 잡는 점(z~7.4cm)이 이미
# 룩 꼭대기(6.05cm)보다 높아 낮은 피치로도 룩을 회피한다. 피치가 클수록 패드가
# 위쪽 꼭지(r 4.7mm)로 다가가 걸리므로 45도는 쓰지 않는다(권장 20도).
BISHOP_TOPDOWN_PITCH_GRID = [18.0, 20.0, 22.0]

# 꼭지 잡기는 무게중심(z~2.5cm)에서 ~5cm 위를 잡아 이동 중 흔들림이 크다.
# 이 leg 의 lift/place down 만 더 느리게 움직여 진자 흔들림을 줄인다.
# 비숍 피치 leg 에만 적용되며, 그 외에는 LIFT_PLACE_SPEED_SCALE 을 그대로 쓴다.
BISHOP_KNOB_LIFT_PLACE_SPEED_SCALE = 0.3

YAW_RANGE = (-math.pi / 6, math.pi / 6)
IN_CELL_R_DEFAULT = 0.005
IN_CELL_R_B2 = 0.004
GRAVE_JITTER_R = 0.010

# [수정] 수평(pitch=0) grasp 에서 손목이 기물에 닿지 않도록, TCP 목표를
# 베이스(원점) 쪽으로 이만큼(m) 당긴다 -> 팔이 0.5cm 뒤로 빠져 손끝으로 잡게 됨.
# top-down(pitch>0)에는 적용하지 않는다.
HORIZ_GRASP_BACKOFF_M = 0.005

# top-down(막힘) 피치 - 룩(및 일반)용
TOPDOWN_PITCH_FLOOR = 45.0
TOPDOWN_PITCH_GRID = [45.0, 46.0, 47.0, 48.0, 49.0, 50.0]

# [수정] 잡기 직후 어떤 충돌 실린더가 그리퍼에 닿았는지 출력(디버그).
# True 로 켜면 episode 마다 grasp 접촉 geom 의 local z / 반경을 찍는다.
# 비숍 목 잡기 검증용: z=3.2/3.9cm(r5.7/5.9) 만 보이면 성공, z=4.65(r11.6)면 칼라 물림.
DEBUG_GRASP_CONTACTS = False

# 성공/검증 허용치
PLACE_TOL_M = 0.018
UPRIGHT_TOL_M = 0.020
GRAVE_TOL_M = 0.020
KNOCK_TOL_M = 0.015
LIFT_DELTA_M = 0.030

HOME_TCP_CM = (0.0, 14.74, 11.44)
HOME_FALLBACK_DEG = [0.0, -10.0, -140.0, 60.0]
HOME_XY_M = (HOME_TCP_CM[0] / 100.0, HOME_TCP_CM[1] / 100.0)

DEFAULT_STARTS_CELLS = {"rook": "a2", "bishop": "c3"}

GIF_FPS = 12
GIF_STRIDE = 2

N_PER_SCENARIO = 3
CAT_NAMES = {
    1: "plain_pick_place",
    2: "pitch_topdown",
    3: "capture_to_graveyard",
    4: "capture_to_graveyard_topdown",
}


# ======================================================================
# 순수 체스 로직
# ======================================================================
def _cr(cell):
    return COLS[cell[0]], int(cell[1])


def _cell(ci, r):
    return f"{ICOL[ci]}{r}"


def rook_moves(src, occupied):
    """룩: 같은 행/열 직선, 점프 불가, 상대 잡기 가능, 1행 금지."""
    ci, r = _cr(src)
    dests = []
    for step in (-1, +1):
        c = ci + step
        while 0 <= c <= 2:
            d = _cell(c, r)
            if d in occupied:
                dests.append(d)
                break
            dests.append(d)
            c += step
    for step in (-1, +1):
        nr = r + step
        if 2 <= nr <= 3:
            dests.append(_cell(ci, nr))
    return dests


def bishop_moves(src, occupied):
    """비숍: 대각선(2행짜리 보드라 1스텝), 1행 금지, 상대 잡기 가능."""
    ci, r = _cr(src)
    dests = []
    for dc in (-1, +1):
        for dr in (-1, +1):
            nc, nr = ci + dc, r + dr
            if 0 <= nc <= 2 and 2 <= nr <= 3:
                dests.append(_cell(nc, nr))
    return dests


def behind(blocker_cell, target_cell):
    """blocker 가 target 의 '팔 쪽 뒤'(같은 열, y 한 칸 작음)에 있어 수평 접근을 막는가."""
    bx, by = CELLS[blocker_cell]
    tx, ty = CELLS[target_cell]
    return (abs(bx - tx) < 1e-6) and (by < ty) and ((ty - by) <= ROW_STEP + 1e-6)


def _make_scenario(rook, bishop, mover, other, src, dst):
    other_cell = bishop if other == "bishop" else rook
    capture = (dst == other_cell)
    s = {
        "rook_cell": rook,
        "bishop_cell": bishop,
        "mover": mover,
        "other": other,
        "src": src,
        "dst": dst,
        "other_cell": other_cell,
        "capture": capture,
    }
    if not capture:
        blocked = behind(other_cell, src) or behind(other_cell, dst)
        s["legs"] = [{"blocked": blocked}]
        s["category"] = 2 if blocked else 1
    else:
        leg1_blocked = behind(src, dst)
        s["legs"] = [{"blocked": leg1_blocked}, {"blocked": False}]
        s["category"] = 4 if leg1_blocked else 3
    return s


def enumerate_scenarios():
    """126 시나리오를 결정적 순서로 생성."""
    scs = []
    for rook in USABLE:
        for bishop in USABLE:
            if rook == bishop:
                continue
            occ = {rook, bishop}
            for d in rook_moves(rook, occ):
                scs.append(_make_scenario(rook, bishop, "rook", "bishop", rook, d))
            for d in bishop_moves(bishop, occ):
                scs.append(_make_scenario(rook, bishop, "bishop", "rook", bishop, d))
    for i, s in enumerate(scs):
        s["idx"] = i
    return scs


def scenario_folder_name(s):
    return (
        f"scenario_{s['idx']:03d}_cat{s['category']}_{s['mover']}_"
        f"{s['src']}-{s['dst']}_rk_{s['rook_cell']}_bp_{s['bishop_cell']}"
    )


def scenario_instruction(s):
    base = f"Move the {PIECE_DESC[s['mover']]} from {s['src']} to {s['dst']}"
    if s["capture"]:
        return base + f" to capture the {PIECE_DESC[s['other']]}."
    return base + "."


# ======================================================================
# 샘플링 + IK 검증
# ======================================================================
def cell_offset_radius(cell):
    return IN_CELL_R_B2 if cell == "b2" else IN_CELL_R_DEFAULT


def sample_offset_m(rng, max_r):
    """중심 편향(절단 가우시안). 대부분 중심 근처, 최대 max_r 이내."""
    if max_r <= 0:
        return 0.0, 0.0
    sigma = max_r / 2.5
    for _ in range(200):
        dx, dy = rng.normal(0.0, sigma, size=2)
        if dx * dx + dy * dy <= max_r * max_r:
            return float(dx), float(dy)
    return 0.0, 0.0


def _backoff_toward_base(xy, delta_m):
    """수평(pitch=0) grasp용: TCP 목표를 베이스(원점) 쪽으로 delta_m 당긴다(반경 방향).

    pad가 손목보다 앞에 있어 룩이 throat 안쪽에 박히면 손목이 닿으므로,
    목표점을 베이스 쪽으로 살짝 당겨 손끝으로 잡게 한다. 같은 양이 place 에도
    적용돼 잡은 기물이 원래 목표 칸에 떨어진다.
    """
    x, y = xy
    r = math.hypot(x, y)
    if r < 1e-9 or delta_m <= 0.0:
        return (float(x), float(y))
    s = (r - delta_m) / r
    return (float(x * s), float(y * s))


def choose_graveyard(rng, cap_xy_m):
    """잡힌 기물 위치에서 가까운 묘지. b열(x~0)이면 랜덤."""
    x = cap_xy_m[0]
    if abs(x) < 1e-6:
        return GRAVEYARDS[rng.choice(["L", "R"])]
    return GRAVEYARDS["L"] if x < 0 else GRAVEYARDS["R"]


def _strat_uniform(rng, lo, hi, k, n):
    """[lo,hi]를 n구간으로 나눠 k구간에서 균등 추출. lo==hi 면 lo 반환."""
    if hi <= lo + 1e-9:
        return lo
    f = (k + rng.random()) / n
    return lo + f * (hi - lo)


def grasp_z_range_for(carry, blocked):
    """기물/접근 방식별 grasp z 탐색 범위(cm).

    [수정] 비숍 피치 잡기는 목(neck) 하단 절대 창으로 강제한다. 오프셋이 아니라
    절대 z 라서 GRASP_Z_RANGE 와 무관하게 항상 꼭지(상단 홈)만 노린다.
    """
    if blocked and carry == "bishop":
        return BISHOP_TOPDOWN_GRASP_Z_RANGE
    return GRASP_Z_RANGE


def feasible_grasp_z_interval(ik, pick_cm, place_cm, pitch, lo=None, hi=None, step=0.05):
    lo = GRASP_Z_RANGE[0] if lo is None else lo
    hi = GRASP_Z_RANGE[1] if hi is None else hi
    zs = []
    z = lo
    while z <= hi + 1e-9:
        zr = round(z, 3)
        if (
            ik(pick_cm[0], pick_cm[1], zr, pitch) is not None
            and ik(place_cm[0], place_cm[1], zr, pitch) is not None
        ):
            zs.append(zr)
        z += step
    if not zs:
        return None
    return min(zs), max(zs)


def feasible_leg_pitches(ik, pick_cm, place_cm, blocked, z_lo=None, z_hi=None, carry=None):
    """이 leg(pick,place 둘 다, SAFE_Z 포함)에서 가능한 pitch 목록.

    [수정] 비숍 피치 leg 는 꼭지용 저피치 그리드를 시도한다(BISHOP_TOPDOWN_PITCH_GRID, 15~25도).
    """
    if not blocked:
        cands = [0.0]
    elif carry == "bishop":
        cands = list(BISHOP_TOPDOWN_PITCH_GRID)
    else:
        cands = list(TOPDOWN_PITCH_GRID)

    out = []
    for p in cands:
        if ik(pick_cm[0], pick_cm[1], SAFE_Z_CM, p) is None:
            continue
        if ik(place_cm[0], place_cm[1], SAFE_Z_CM, p) is None:
            continue
        if feasible_grasp_z_interval(ik, pick_cm, place_cm, p, lo=z_lo, hi=z_hi) is None:
            continue
        out.append(float(p))
    return out


def _make_leg(ik, rng, pick_xy, place_xy, blocked, carry, place_center, ep_idx, n_per):
    # [수정] 수평(pitch=0) leg 에만 backoff: 팔을 베이스 쪽으로 0.5cm 빼서 손끝으로 잡게.
    # top-down(blocked=True, pitch>0)에는 적용하지 않는다.
    if not blocked:
        pick_xy = _backoff_toward_base(pick_xy, HORIZ_GRASP_BACKOFF_M)
        place_xy = _backoff_toward_base(place_xy, HORIZ_GRASP_BACKOFF_M)

    pick_cm = (pick_xy[0] * 100.0, pick_xy[1] * 100.0)
    place_cm = (place_xy[0] * 100.0, place_xy[1] * 100.0)

    # [수정] 비숍 피치 잡기는 grasp z 탐색 범위를 꼭지(상단 홈) 절대 창으로 강제.
    z_lo, z_hi = grasp_z_range_for(carry, blocked)
    bishop_knob = bool(blocked) and (carry == "bishop")

    pitches = feasible_leg_pitches(
        ik, pick_cm, place_cm, blocked, z_lo=z_lo, z_hi=z_hi, carry=carry,
    )
    if not pitches:
        return None

    if not blocked:
        pitch = 0.0
        zint = feasible_grasp_z_interval(ik, pick_cm, place_cm, pitch, lo=z_lo, hi=z_hi)
        if zint is None:
            return None
    else:
        if bishop_knob:
            # 비숍 꼭지 잡기: 저피치 그리드(15~25도) 그대로 사용.
            # TOPDOWN_PITCH_FLOOR(=45) 클램프를 적용하면 안 된다(꼭지 간섭).
            lo = min(pitches)
            hi = max(pitches)
        else:
            # 룩 등은 기존대로 45~50 에서 층화 추출.
            lo = max(TOPDOWN_PITCH_FLOOR, min(pitches))
            hi = max(pitches)
        pitch = _strat_uniform(rng, lo, hi, ep_idx, n_per)
        zint = feasible_grasp_z_interval(ik, pick_cm, place_cm, pitch, lo=z_lo, hi=z_hi)
        if zint is None:
            for p in sorted(pitches):
                z2 = feasible_grasp_z_interval(ik, pick_cm, place_cm, p, lo=z_lo, hi=z_hi)
                if z2 is not None:
                    pitch, zint = p, z2
                    break
            if zint is None:
                return None

    gz = _strat_uniform(rng, zint[0], zint[1], ep_idx, n_per)
    return {
        "pick_xy": (float(pick_xy[0]), float(pick_xy[1])),
        "place_xy": (float(place_xy[0]), float(place_xy[1])),
        "place_center": (float(place_center[0]), float(place_center[1])),
        "pitch_deg": float(pitch),
        "grasp_z_cm": float(gz),
        "carry": carry,
        "blocked": bool(blocked),
        # 비숍 꼭지 잡기 leg 표시: 흔들림 억제용 추가 감속에 사용.
        "bishop_knob": bishop_knob,
    }


def leg_waypoints(leg):
    """한 leg 의 9 waypoint. 0번은 x,y=None(현재 EE 위치에서 들어올림)."""
    px, py = leg["pick_xy"]
    qx, qy = leg["place_xy"]
    gz = leg["grasp_z_cm"] / 100.0
    sz = SAFE_Z_CM / 100.0
    carry = leg["carry"]
    return [
        {"x": None, "y": None, "z": sz, "grip": 0},                                  # 0 raise
        {"x": px, "y": py, "z": sz, "grip": 0},                                      # 1 over pick
        {"x": px, "y": py, "z": gz, "grip": 0},                                      # 2 down
        {"x": px, "y": py, "z": gz, "grip": 1, "is_grasp": True, "carry": carry},    # 3 close
        {"x": px, "y": py, "z": sz, "grip": 1, "is_lift": True,
         "carry": carry, "slow_motion": True},                                       # 4 lift
        {"x": qx, "y": qy, "z": sz, "grip": 1},                                      # 5 sweep
        {"x": qx, "y": qy, "z": gz, "grip": 1,
         "is_place_down": True, "slow_motion": True},                                # 6 place down
        {"x": qx, "y": qy, "z": gz, "grip": 0, "is_release": True},                  # 7 release
        {"x": qx, "y": qy, "z": sz, "grip": 0},                                      # 8 retreat
    ]


def validate_legs(ik, legs):
    """현재-EE 체인(leg1=home, leg2=직전 place)으로 전 waypoint 도달성 검증."""
    cur = HOME_XY_M
    for leg in legs:
        for wp in leg_waypoints(leg):
            x = cur[0] if wp["x"] is None else wp["x"]
            y = cur[1] if wp["y"] is None else wp["y"]
            if ik(x * 100.0, y * 100.0, wp["z"] * 100.0, leg["pitch_deg"]) is None:
                return False
        cur = leg["place_xy"]
    return True


def sample_valid_params(ik, rng, scenario, ep_idx, n_per=N_PER_SCENARIO, max_tries=80):
    """IK 로 도달 가능한 에피소드 파라미터 1세트를 뽑는다. 실패 시 None."""
    for _ in range(max_tries):
        starts = {}
        for piece, cell in (("rook", scenario["rook_cell"]), ("bishop", scenario["bishop_cell"])):
            cx, cy = CELLS[cell]
            dx, dy = sample_offset_m(rng, cell_offset_radius(cell))
            starts[piece] = {
                "xy": (cx + dx, cy + dy),
                "yaw": float(rng.uniform(*YAW_RANGE)),
                "cell": cell,
            }

        legs = []
        graveyard_xy = None

        if not scenario["capture"]:
            mover, dst = scenario["mover"], scenario["dst"]
            pick_xy = starts[mover]["xy"]
            dcx, dcy = CELLS[dst]
            ddx, ddy = sample_offset_m(rng, cell_offset_radius(dst))
            place_xy = (dcx + ddx, dcy + ddy)
            leg = _make_leg(
                ik, rng, pick_xy, place_xy,
                scenario["legs"][0]["blocked"], mover, CELLS[dst],
                ep_idx, n_per,
            )
            if leg is None:
                continue
            legs = [leg]
        else:
            captured = scenario["other"]
            cap_xy = starts[captured]["xy"]
            grave = choose_graveyard(rng, cap_xy)
            gjx, gjy = sample_offset_m(rng, GRAVE_JITTER_R)
            grave_pt = (grave[0] + gjx, grave[1] + gjy)
            graveyard_xy = grave

            leg1 = _make_leg(
                ik, rng, cap_xy, grave_pt,
                scenario["legs"][0]["blocked"], captured, grave,
                ep_idx, n_per,
            )
            if leg1 is None:
                continue

            mover, dst = scenario["mover"], scenario["dst"]
            pick_xy = starts[mover]["xy"]
            dcx, dcy = CELLS[dst]
            ddx, ddy = sample_offset_m(rng, cell_offset_radius(dst))
            place_xy = (dcx + ddx, dcy + ddy)
            leg2 = _make_leg(
                ik, rng, pick_xy, place_xy,
                False, mover, CELLS[dst],
                ep_idx, n_per,
            )
            if leg2 is None:
                continue
            legs = [leg1, leg2]

        if not validate_legs(ik, legs):
            continue

        return {"starts": starts, "legs": legs, "graveyard_xy": graveyard_xy}
    return None


# ======================================================================
# 환경
# ======================================================================
class ChessDataEnv(PitchRealRaccoonEnv):
    def reset_episode(self, starts=None, *args, **kwargs):
        """부모 __init__ 이 reset_episode(0.15,0.15,0.0) 로 부르는 경우를 흡수."""
        mujoco.mj_resetData(self.model, self.data)

        ha = self._calc_inv_kinematics(HOME_TCP_CM[0], HOME_TCP_CM[1], HOME_TCP_CM[2], pitch=0.0)
        home = np.radians(ha if ha is not None else HOME_FALLBACK_DEG)
        for i in range(4):
            self.data.qpos[i] = home[i]
            self.data.ctrl[i] = home[i]
            self.current_setpoints[i] = home[i]
            self.target_angles[i] = home[i]
            self.joint_control_mode[i] = self.MODE_POSITION
        self.data.qvel[:] = 0.0

        self.data.qpos[4] = self.GRIP_OPEN
        self.data.ctrl[4] = self.GRIP_OPEN
        self.current_setpoints[4] = self.GRIP_OPEN
        self.gripper_target = self.GRIP_OPEN
        self.gripper_mode = self.GRIP_MODE_FREE

        valid = (
            isinstance(starts, dict)
            and "rook" in starts
            and isinstance(starts.get("rook"), dict)
        )
        if not valid:
            starts = {
                p: {"xy": CELLS[c], "yaw": 0.0, "cell": c}
                for p, c in DEFAULT_STARTS_CELLS.items()
            }

        for piece in PIECE_BODIES:
            x, y = starts[piece]["xy"]
            self.reset_object_pose(piece, x=x, y=y, z=SPAWN_Z, yaw=starts[piece]["yaw"])
            b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, piece)
            self.model.body_mass[b] = PIECE_MASS

        mujoco.mj_forward(self.model, self.data)
        self.step_n(20)

    def firm_grip(self):
        target = max(self.GRIP_CLOSE, self.current_setpoints[4] - SQUEEZE)
        if target == self.GRIP_CLOSE:
            target = self.GRIP_CLOSE + 1e-4
        self.gripper_target = target


# ======================================================================
# 잡기 접촉 진단(디버그)
# ======================================================================
def log_grasp_contacts(env, piece, tag=""):
    """잡기 직후 grasp 한 기물의 어떤 충돌 geom 이 그리퍼에 닿았는지 출력.

    비숍 목 잡기 검증용. local z 와 반경(mm)으로 어느 부위인지 알 수 있다.
      z~3.2/3.9cm (r 5.7/5.9mm) = 목  -> 성공
      z~4.65cm   (r 11.6mm)    = 칼라 -> 아직 칼라 물림(실패 가능)
      z~2.5cm    (r 8.5mm)     = 어깨 -> 너무 낮음
    """
    m, d = env.model, env.data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, piece)
    if bid == -1:
        return
    seen = []
    for i in range(d.ncon):
        c = d.contact[i]
        for g in (c.geom1, c.geom2):
            if m.geom_bodyid[g] == bid:
                zc = float(m.geom_pos[g][2]) * 100.0   # cm (body-local)
                rr = float(m.geom_size[g][0]) * 1000.0  # mm
                seen.append((zc, rr))
    if not seen:
        print(f"    [grasp{(' ' + tag) if tag else ''}] {piece}: 접촉 geom 없음")
        return
    seen.sort()
    parts = ", ".join(f"z={z:.2f}cm r={r:.1f}mm" for z, r in seen)
    print(f"    [grasp{(' ' + tag) if tag else ''}] {piece}: {parts}")


# ======================================================================
# 로거
# ======================================================================
class ChessEpisodeLogger:
    def __init__(self, scenario_dir):
        self.scenario_dir = Path(scenario_dir)
        self.scenario_dir.mkdir(parents=True, exist_ok=True)
        self.ep_dir = None
        self.meta = None
        self.frames = None

    def start(self, ep_idx, header):
        self.ep_dir = self.scenario_dir / f"episode_{ep_idx:02d}"
        if self.ep_dir.exists():
            shutil.rmtree(self.ep_dir, ignore_errors=True)
        self.ep_dir.mkdir(parents=True, exist_ok=True)
        self.meta = dict(header)
        self.meta["steps"] = []
        self.frames = []

    def log(self, t, img, joints, grip, mover_pose, other_pose, ee, action,
            leg_index, is_first, is_last):
        fn = f"frame_{t:06d}.png"
        Image.fromarray(img).save(self.ep_dir / fn)
        self.frames.append(img)
        self.meta["steps"].append({
            "t": int(t),
            "image_file": fn,
            "leg_index": int(leg_index),
            "joint_angles": [float(x) for x in joints],
            "gripper_state": float(grip),
            "object_pose": [float(x) for x in mover_pose],
            "other_object_pose": [float(x) for x in other_pose],
            "ee_pose": [float(x) for x in ee],
            "action": [float(x) for x in action],
            "is_first": bool(is_first),
            "is_last": bool(is_last),
        })

    def finalize(self, success, fail_info=None):
        self.meta["success"] = bool(success)
        if fail_info is not None:
            self.meta["fail_info"] = fail_info
        with open(self.ep_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)

        if self.frames:
            imgs = [Image.fromarray(f) for f in self.frames[::GIF_STRIDE]]
            if imgs:
                imgs[0].save(
                    self.ep_dir / "episode.gif",
                    save_all=True,
                    append_images=imgs[1:],
                    duration=int(1000 / GIF_FPS),
                    loop=0,
                )

    def abort(self):
        if self.ep_dir is not None and self.ep_dir.exists():
            shutil.rmtree(self.ep_dir, ignore_errors=True)


# ======================================================================
# 한 에피소드 실행 + 기록
# ======================================================================
def _link4_ee(env):
    lid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, EE_LINK_BODY)
    if lid != -1:
        p = env.data.xpos[lid]
        return [float(p[0]), float(p[1]), float(p[2])]
    return [0.0, 0.0, 0.0]


def _observe(env, scenario):
    img = env.render_rgb()
    joints = [float(env.data.qpos[i]) for i in range(4)]
    grip = float(env.data.qpos[4])
    mp = env.get_object_pose(scenario["mover"])
    op = env.get_object_pose(scenario["other"])
    return {
        "img": img,
        "joints": joints,
        "grip": grip,
        "mover_pose": [float(v) for v in mp],
        "other_pose": [float(v) for v in op],
        "ee": [float(v) for v in env.get_ee_pose()[:3]],
    }


def episode_eval(env, scenario, params, mover_z0, other_xy0):
    """(성공여부, 진단정보) 반환."""
    mp = env.get_object_pose(scenario["mover"])
    dcx, dcy = CELLS[scenario["dst"]]
    place_err = math.hypot(float(mp[0]) - dcx, float(mp[1]) - dcy)
    dz = abs(float(mp[2]) - mover_z0)
    placed = place_err <= PLACE_TOL_M
    upright = dz <= UPRIGHT_TOL_M

    info = {
        "place_err_cm": round(place_err * 100, 2),
        "place_tol_cm": PLACE_TOL_M * 100,
        "upright_dz_cm": round(dz * 100, 2),
        "upright_tol_cm": UPRIGHT_TOL_M * 100,
    }

    if scenario["capture"]:
        cp = env.get_object_pose(scenario["other"])
        gx, gy = params["graveyard_xy"]
        grave_err = math.hypot(float(cp[0]) - gx, float(cp[1]) - gy)
        cleared = grave_err <= GRAVE_TOL_M
        info["grave_err_cm"] = round(grave_err * 100, 2)
        info["grave_tol_cm"] = GRAVE_TOL_M * 100
        info["checks"] = {"placed": placed, "upright": upright, "cleared": cleared}
        ok = placed and upright and cleared
    else:
        op = env.get_object_pose(scenario["other"])
        sx, sy = other_xy0
        knock = math.hypot(float(op[0]) - sx, float(op[1]) - sy)
        undisturbed = knock <= KNOCK_TOL_M
        info["knock_cm"] = round(knock * 100, 2)
        info["knock_tol_cm"] = KNOCK_TOL_M * 100
        info["checks"] = {"placed": placed, "upright": upright, "undisturbed": undisturbed}
        ok = placed and upright and undisturbed

    return bool(ok), info


def _finish(logger, keep_failed, success, info):
    """성공이면 저장. 실패는 keep_failed 일 때만 저장, 아니면 폐기."""
    if success or keep_failed:
        logger.finalize(success, info)
    else:
        logger.abort()
    return success, info


def run_one_episode(env, logger, ep_idx, scenario, params, keep_failed=False):
    env.reset_episode(params["starts"])
    env.settle_steps(INITIAL_SETTLE)

    header = {
        "episode_id": int(ep_idx),
        "scenario_id": int(scenario["idx"]),
        "category": int(scenario["category"]),
        "category_name": CAT_NAMES[scenario["category"]],
        "instruction": scenario_instruction(scenario),
        "task_type": "capture_pick_place" if scenario["capture"] else "pick_place",
        "moved_piece": scenario["mover"],
        "src": scenario["src"],
        "dst": scenario["dst"],
        "dst_center_xy": [
            float(CELLS[scenario["dst"]][0]),
            float(CELLS[scenario["dst"]][1]),
        ],
        "capture": bool(scenario["capture"]),
        "captured_piece": scenario["other"] if scenario["capture"] else None,
        "graveyard_xy": (
            [float(params["graveyard_xy"][0]), float(params["graveyard_xy"][1])]
            if params["graveyard_xy"] is not None else None
        ),
        "rook_init_cell": scenario["rook_cell"],
        "bishop_init_cell": scenario["bishop_cell"],
        "all_object_init_poses": {
            p: {
                "xy": [
                    float(params["starts"][p]["xy"][0]),
                    float(params["starts"][p]["xy"][1]),
                ],
                "yaw": float(params["starts"][p]["yaw"]),
                "cell": params["starts"][p]["cell"],
            }
            for p in PIECE_BODIES
        },
        "legs": [
            {
                "pick_xy": list(leg["pick_xy"]),
                "place_xy": list(leg["place_xy"]),
                "pitch_deg": leg["pitch_deg"],
                "grasp_z_cm": leg["grasp_z_cm"],
                "carry": leg["carry"],
                "blocked": leg["blocked"],
                "bishop_knob": leg.get("bishop_knob", False),
            }
            for leg in params["legs"]
        ],
        "action_format": "[x_m, y_m, z_m, pitch_deg, gripper(0=open,1=close)]",
        "fps": HZ,
    }
    logger.start(ep_idx, header)

    dt = 1.0 / HZ
    n_move = max(1, int(SETTLE_PER_ACTION * HZ))
    n_grasp = max(1, int(GRASP_SETTLE_SEC * HZ))
    n_hold = max(1, int(HOLD_SEC * HZ))

    base_rot_scale = getattr(env, "BASE_ROT_SPEED_SCALE", 1.0)
    if base_rot_scale <= 0.0:
        base_rot_scale = 1.0

    state = {"t": 0, "obs": _observe(env, scenario)}
    mover_z0 = float(env.get_object_pose(scenario["mover"])[2])
    _op0 = env.get_object_pose(scenario["other"])
    other_xy0 = (float(_op0[0]), float(_op0[1]))

    def settle_and_log(action, leg_idx, count):
        for _ in range(count):
            o = state["obs"]
            logger.log(
                state["t"],
                o["img"],
                o["joints"],
                o["grip"],
                o["mover_pose"],
                o["other_pose"],
                o["ee"],
                action,
                leg_idx,
                is_first=(state["t"] == 0),
                is_last=False,
            )
            env.settle_steps(dt)
            state["obs"] = _observe(env, scenario)
            state["t"] += 1

    def joints_reached(tol_deg=1.0):
        """
        Joint1~3 이 목표 각도에 도착했는지 확인한다.
        Joint4 는 lockh()/lockp()에서 별도 공식으로 제어되므로 제외한다.
        """
        tol = math.radians(tol_deg)
        for j in range(3):
            if abs(env.target_angles[j] - env.current_setpoints[j]) > tol:
                return False
        return True

    def calc_max_move_count(motion_scale):
        """
        Joint1은 BASE_ROT_SPEED_SCALE로 느려져 있고,
        lift/place down에서는 motion_scale이 추가 적용된다.
        """
        if motion_scale <= 0.0:
            motion_scale = 1.0

        effective_scale = min(base_rot_scale * motion_scale, motion_scale)
        if effective_scale <= 0.0:
            effective_scale = 1.0

        return max(n_move, int(n_move / effective_scale) + 5)

    def settle_and_log_until_reached(action, leg_idx, min_count, max_count):
        """
        최소 min_count만큼 진행하고,
        이후 Joint1~3이 목표 각도에 도착하면 바로 다음 단계로 넘어간다.
        """
        for step_i in range(max_count):
            o = state["obs"]
            logger.log(
                state["t"],
                o["img"],
                o["joints"],
                o["grip"],
                o["mover_pose"],
                o["other_pose"],
                o["ee"],
                action,
                leg_idx,
                is_first=(state["t"] == 0),
                is_last=False,
            )
            env.settle_steps(dt)
            state["obs"] = _observe(env, scenario)
            state["t"] += 1

            if step_i + 1 >= min_count and joints_reached():
                break

    def set_env_motion_scale(scale):
        """
        raccoon_env_pitch_real.py의 set_motion_speed_scale()과 연결된다.
        없는 경우에도 에러 없이 무시한다.
        """
        if hasattr(env, "set_motion_speed_scale"):
            env.set_motion_speed_scale(scale)

    try:
        pre_grasp_z = None
        last_xy = (HOME_XY_M[0], HOME_XY_M[1])
        grip_cmd = 0.0

        for li, leg in enumerate(params["legs"]):
            pitch = leg["pitch_deg"]
            # 비숍 꼭지 잡기 leg 는 무게중심에서 멀어 흔들림이 크므로 더 느리게.
            slow_scale = (
                BISHOP_KNOB_LIFT_PLACE_SPEED_SCALE
                if leg.get("bishop_knob")
                else LIFT_PLACE_SPEED_SCALE
            )

            if pitch > 0.0:
                env.lockp(pitch)
            else:
                env.lockh()

            for wp in leg_waypoints(leg):
                if wp["x"] is None:
                    x, y = last_xy
                else:
                    x, y = wp["x"], wp["y"]

                z = wp["z"]
                last_xy = (x, y)
                target_grip = float(wp["grip"])

                motion_scale = slow_scale if wp.get("slow_motion") else 1.0
                set_env_motion_scale(motion_scale)

                env.move_to(
                    x * 100.0,
                    y * 100.0,
                    z * 100.0,
                    pitch=pitch,
                    speed=SPEED,
                )

                # 이동 중에는 기존 그리퍼 상태 유지.
                # 이동 완료 후에만 close/open을 실행한다.
                move_action = [x, y, z, pitch, grip_cmd]

                max_move = calc_max_move_count(motion_scale)
                settle_and_log_until_reached(move_action, li, n_move, max_move)

                set_env_motion_scale(1.0)

                # 목표 위치에 완전히 도착한 뒤에만 그리퍼 명령 변경.
                if target_grip != grip_cmd:
                    if target_grip >= 0.5:
                        env.close_gripper()
                    else:
                        env.open_gripper()

                    grip_cmd = target_grip
                    grip_action = [x, y, z, pitch, grip_cmd]

                    if wp.get("is_grasp"):
                        settle_and_log(grip_action, li, n_grasp)
                        env.firm_grip()
                        settle_and_log(grip_action, li, n_hold)
                        pre_grasp_z = float(env.get_object_pose(wp["carry"])[2])
                        # [수정] 어느 부위를 물었는지 진단(기본 off).
                        if DEBUG_GRASP_CONTACTS:
                            log_grasp_contacts(
                                env, wp["carry"],
                                tag=f"leg{li} pitch{pitch:.0f}",
                            )
                    elif wp.get("is_release"):
                        settle_and_log(grip_action, li, n_grasp)
                    else:
                        settle_and_log(grip_action, li, n_grasp)

                if wp.get("is_lift") and pre_grasp_z is not None:
                    lifted = float(env.get_object_pose(wp["carry"])[2]) - pre_grasp_z
                    if lifted < LIFT_DELTA_M:
                        info = {
                            "fail_stage": "grasp_lift",
                            "leg_index": li,
                            "carry": wp["carry"],
                            "lifted_cm": round(lifted * 100, 2),
                            "need_cm": round(LIFT_DELTA_M * 100, 2),
                        }
                        return _finish(logger, keep_failed, False, info)

        env.settle_steps(0.5)
        state["obs"] = _observe(env, scenario)

        last = params["legs"][-1]
        last_action = [
            last["place_xy"][0],
            last["place_xy"][1],
            SAFE_Z_CM / 100.0,
            last["pitch_deg"],
            0.0,
        ]

        o = state["obs"]
        logger.log(
            state["t"],
            o["img"],
            o["joints"],
            o["grip"],
            o["mover_pose"],
            o["other_pose"],
            o["ee"],
            last_action,
            len(params["legs"]) - 1,
            is_first=False,
            is_last=True,
        )

        ok, info = episode_eval(env, scenario, params, mover_z0, other_xy0)
        if not ok:
            info["fail_stage"] = "final_check"
        return _finish(logger, keep_failed, ok, info)

    except ValueError as e:
        set_env_motion_scale(1.0)
        return _finish(
            logger,
            keep_failed,
            False,
            {"fail_stage": "unreachable_move", "msg": str(e)},
        )

    except Exception:
        set_env_motion_scale(1.0)
        logger.abort()
        raise


# ======================================================================
# 드라이버
# ======================================================================
def _slot_done(scenario_dir, k):
    """해당 슬롯(episode_kk)에 success=true 메타가 이미 저장돼 있는지."""
    m = Path(scenario_dir) / f"episode_{k:02d}" / "meta.json"
    if not m.exists():
        return False
    try:
        with open(m, encoding="utf-8") as f:
            return bool(json.load(f).get("success"))
    except Exception:
        return False


def collect_dataset(
    xml_path=XML_PATH,
    dataset_root="Top_chess_data",
    n_per_scenario=N_PER_SCENARIO,
    camera_name=CAMERA_NAME,
    use_viewer=False,
    seed=0,
    only_categories=None,
    only_scenarios=None,
    resume=True,
    keep_failed=False,
    give_up_no_success_after=10,     # 성공 0개인데 이 횟수를 넘기면 시나리오 포기
    max_attempts_per_scenario=None,  # 시나리오당 총 시도 상한(None이면 자동 산정)
):
    all_scenarios = enumerate_scenarios()
    scenarios = all_scenarios

    if only_categories is not None:
        scenarios = [s for s in scenarios if s["category"] in set(only_categories)]
    if only_scenarios is not None:
        scenarios = [s for s in scenarios if s["idx"] in set(only_scenarios)]

    if max_attempts_per_scenario is None:
        # 슬롯마다 몇 번씩 실패해도 N개를 채울 여유를 둔다.
        max_attempts_per_scenario = max(60, n_per_scenario * 8)

    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)

    index = [{
        "idx": s["idx"],
        "category": s["category"],
        "category_name": CAT_NAMES[s["category"]],
        "instruction": scenario_instruction(s),
        "mover": s["mover"],
        "src": s["src"],
        "dst": s["dst"],
        "capture": s["capture"],
        "rook_cell": s["rook_cell"],
        "bishop_cell": s["bishop_cell"],
        "folder": scenario_folder_name(s),
    } for s in all_scenarios]

    with open(root / "scenarios_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    env = ChessDataEnv(
        xml_path=xml_path,
        image_size=IMAGE_SIZE,
        camera_name=camera_name,
        use_viewer=use_viewer,
    )
    ik = env._calc_inv_kinematics

    total_ok = 0
    skipped_scenarios = 0

    try:
        for s in scenarios:
            sdir = root / scenario_folder_name(s)
            logger = ChessEpisodeLogger(sdir)

            # 이미 성공 저장된 슬롯은 건너뛰고, 비었거나 실패한 슬롯만 채운다.
            # (중간이 빠진 데이터도 그 칸만 골라 메워서 0..N-1 이 모두 차게 한다.)
            slots_to_fill = [
                k for k in range(n_per_scenario)
                if not (resume and _slot_done(sdir, k))
            ]
            already_done = n_per_scenario - len(slots_to_fill)

            if not slots_to_fill:
                print(f"[scenario {s['idx']:03d}] 이미 {n_per_scenario}개 완료(skip)")
                continue

            attempt = 0                          # 이 시나리오의 누적 시도 수(랜덤성 시드로도 사용)
            successes_this_run = 0
            proven_possible = already_done > 0   # 기존에 성공이 있으면 '가능한 시나리오'로 간주
            skip_scenario = False

            for slot in slots_to_fill:
                ok = False
                while not ok:
                    # (1) 성공이 한 번도 없는데 너무 많이 시도하면 이 시나리오는 포기.
                    no_success_yet = (not proven_possible) and successes_this_run == 0
                    if no_success_yet and attempt >= give_up_no_success_after:
                        skip_scenario = True
                        break
                    # (2) 안전장치: 전체 시도 상한 초과 시 포기.
                    if attempt >= max_attempts_per_scenario:
                        skip_scenario = True
                        break

                    # 시도마다 다른 시드 -> 다른 랜덤성. 층화(stratify)는 slot 기준으로 유지.
                    rng = np.random.default_rng([seed, s["idx"], attempt])
                    attempt += 1

                    params = sample_valid_params(
                        ik, rng, s, ep_idx=slot, n_per=n_per_scenario,
                    )
                    if params is None:
                        # IK로 도달 가능한 파라미터를 못 뽑음 -> 다른 랜덤성으로 재시도
                        print(
                            f"[scenario {s['idx']:03d}] slot {slot:02d} "
                            f"시도 {attempt} 실패: no_feasible_params"
                        )
                        continue

                    try:
                        ok, info = run_one_episode(
                            env, logger, slot, s, params, keep_failed=keep_failed,
                        )
                    except Exception as e:
                        ok, info = False, {"fail_stage": "exception", "msg": str(e)}
                        print(f"[scenario {s['idx']:03d}] slot {slot:02d} 예외: {e}")

                    if ok:
                        successes_this_run += 1
                        total_ok += 1
                        print(
                            f"[scenario {s['idx']:03d} cat{s['category']}] "
                            f"{s['mover']} {s['src']}->{s['dst']} "
                            f"slot {slot:02d} OK (누적 시도 {attempt})"
                        )
                    else:
                        print(
                            f"[scenario {s['idx']:03d}] slot {slot:02d} "
                            f"시도 {attempt} 실패: {info}"
                        )

                if skip_scenario:
                    break

            if skip_scenario:
                skipped_scenarios += 1
                print(
                    f"[scenario {s['idx']:03d}] 시도 {attempt}회에도 {n_per_scenario}개를 "
                    f"못 채워 건너뜀 (이번 실행 성공 {successes_this_run}, 기존 {already_done})"
                )

    finally:
        env.close()

    print(f"\n완료: 저장 성공 {total_ok}개, 건너뛴 시나리오 {skipped_scenarios}개")
    print(f"출력: {root}/ (대상 시나리오 {len(scenarios)}개)")


if __name__ == "__main__":
    collect_dataset(
        xml_path=XML_PATH,
        dataset_root="Top_chess_data",
        n_per_scenario=8,
        seed=0,
        # only_categories=[4],
        resume=True,
        give_up_no_success_after=10,   # 성공 0개로 10회 초과 시 시나리오 건너뜀
    )