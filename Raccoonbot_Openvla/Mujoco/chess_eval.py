"""
chess_eval.py - 체스 VLA 평가 (서버 헤드리스)
====================================================================

학습한 체스 OpenVLA 체크포인트를 실제로 굴려서 시나리오별 성공률을 잰다.
데이터 생성기(chess_pick_place_dataset.py)의 시나리오 배치 / 성공 판정 / GIF 로직을
그대로 재사용하고, '정해진 waypoint 추종' 부분만 '모델이 뱉은 액션 추종'으로 바꾼다.

핵심 (데이터 인코딩과 정확히 대칭으로 적용):
  - 서버가 주는 action = [dx_m, dy_m, dz_m, 0, pitch_abs_deg, 0, grip]
  - xyz 는 '현재 EE(get_ee_pose, m) + delta' 를 절대좌표로 만들어 move_to(cm)
  - pitch(action[4]) 는 '절대 피치(deg)' → lockp(pitch) 후 그 피치로 이동
  - grip(action[6]) 는 0.5 기준 close/open
  - move_to 는 비블로킹이므로 호출 후 settle_steps 로 도착을 기다린다.
  - 성공 판정 / 기물 배치 / 명령 문자열 / GIF 는 데이터 생성기 함수를 그대로 import.

실행 (서버 안, openvla_server.py 가 localhost:8000 에 떠 있어야 함):
  # 단발 1개 시나리오 (움직임 확인용 GIF)
  python chess_eval.py --only_scenarios 0 --n_per 1 --max_steps 60

  # 카테고리별 몇 개만 빠르게
  python chess_eval.py --only_categories 1 2 --n_per 3

  # 전체 시나리오 성공률 (오래 걸림)
  python chess_eval.py --n_per 5
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import io
import json
import time
import base64
import argparse
import math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import requests
from PIL import Image

# 데이터 생성기에서 환경/시나리오/판정/상수를 그대로 가져온다.
import chess_pick_place_dataset as C
from chess_pick_place_dataset import (
    ChessDataEnv,
    enumerate_scenarios,
    scenario_instruction,
    scenario_folder_name,
    episode_eval,
    sample_valid_params,
    CELLS,
    SAFE_Z_CM,
    HZ,
    CAT_NAMES,
)


# ----------------------------------------------------------------------
# 서버 통신
# ----------------------------------------------------------------------
def image_to_b64(img_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def request_action(server_url, instruction, img_rgb, unnorm_key, timeout=60.0):
    payload = {
        "instruction": instruction,
        "image_b64": image_to_b64(img_rgb),
        "unnorm_key": unnorm_key,
        "do_sample": False,
    }
    r = requests.post(f"{server_url.rstrip('/')}/predict", json=payload, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"server {r.status_code}: {r.text}")
    return r.json()["action"]


# ----------------------------------------------------------------------
# 모델 액션 1스텝 적용
# ----------------------------------------------------------------------
def apply_action(env, action, speed, settle_dt, settle_steps,
                 max_delta_xyz=None, pitch_thresh=1.0, delta_scale=1.0):
    a = [float(x) for x in action]
    dx, dy, dz = a[0]*delta_scale, a[1]*delta_scale, a[2]*delta_scale
    pitch = a[4] if len(a) > 4 else 0.0
    grip  = a[6] if len(a) > 6 else (a[-1] if a else 0.0)

    if max_delta_xyz is not None:
        n = math.sqrt(dx*dx+dy*dy+dz*dz)
        if n > max_delta_xyz and n > 1e-9:
            s = max_delta_xyz/n; dx,dy,dz = dx*s,dy*s,dz*s

    ee = env.get_ee_pose()[:3]
    tx,ty,tz = (ee[0]+dx)*100, (ee[1]+dy)*100, (ee[2]+dz)*100

    if pitch > pitch_thresh: env.lockp(pitch)
    else:                    env.lockh()

    env.move_to(tx, ty, tz, pitch=pitch, speed=speed)

    # grip 먼저 반영 (데이터는 close/open 명령 후 그 상태로 settle)
    if grip >= 0.5: env.close_gripper()
    else:           env.open_gripper()

    # 핵심: settle을 딱 1틱만. 데이터 1프레임 = dt 한 번.
    env.settle_steps(settle_dt)
# ----------------------------------------------------------------------
# 한 에피소드 롤아웃
# ----------------------------------------------------------------------
def rollout_episode(env, server_url, scenario, params, unnorm_key,
                    out_dir, max_steps, speed, settle_dt, settle_steps,
                    max_delta_xyz, save_gif, request_timeout, delta_scale=1.0,
                    idle_eps_m=0.002, idle_patience=15):
    out_dir.mkdir(parents=True, exist_ok=True)
    instruction = scenario_instruction(scenario)

    # 시나리오 기물 배치(데이터 생성기와 동일)
    env.reset_episode(params["starts"])
    env.settle_steps(C.INITIAL_SETTLE)

    # 성공 판정 기준값(데이터 생성기와 동일)
    mover_z0 = float(env.get_object_pose(scenario["mover"])[2])
    _op0 = env.get_object_pose(scenario["other"])
    other_xy0 = (float(_op0[0]), float(_op0[1]))

    frames = []
    img = env.render_rgb()
    frames.append(img)

    fail_reason = None
    idle_count = 0
    stuck = 0                      # grip 닫힌 채 z≈0 연속 틱
    nudged = False                 # 이번 에피소드에서 이미 nudge 했는지
    prev_ee = np.array(env.get_ee_pose()[:3])
    prev_grip = None
    for step_idx in range(max_steps):
        try:
            action = request_action(server_url, instruction, img, unnorm_key, request_timeout)
        except Exception as exc:
            fail_reason = f"server_error: {exc}"
            break

        cur_grip = 1.0 if (len(action) > 6 and float(action[6]) >= 0.5) else 0.0
        try:
            apply_action(
                env, action, speed=speed, settle_dt=settle_dt,
                settle_steps=settle_steps, max_delta_xyz=max_delta_xyz,
                delta_scale=delta_scale,
            )
        except ValueError as exc:
            fail_reason = f"unreachable: {exc}"

        # --- [nudge] grip 닫힌 채 z 명령이 거의 0으로 멈춰 있으면 정지 루프로 보고
        #     강제로 살짝 들어올려 '들린 화면'을 모델에 보여준다(데이터의 lift waypoint 흉내). ---
        z_cmd = abs(float(action[2])) if len(action) > 2 else 0.0
        if cur_grip >= 0.5 and z_cmd < 1.5e-4:
            stuck += 1
        else:
            stuck = 0
        if cur_grip >= 0.5 and stuck >= 5 and not nudged:
            pitch = float(action[4]) if len(action) > 4 else 0.0
            if pitch > 1.0:
                env.lockp(pitch)
            else:
                env.lockh()
            ee = env.get_ee_pose()[:3]
            # 데이터의 lift 목표(SAFE_Z)까지 한 번에 끌어올린다.
            env.move_to(ee[0] * 100.0, ee[1] * 100.0, C.SAFE_Z_CM,
                        pitch=pitch, speed=speed)
            for _ in range(12):
                env.settle_steps(settle_dt)
            nudged = True
            stuck = 0
            fail_reason = (fail_reason or "") + f" | nudged@{step_idx+1}"

        img = env.render_rgb()
        frames.append(img)

        # 조기 종료: EE 가 거의 안 움직이고 그리퍼 상태도 안 변하면 동작 끝난 것으로 보고 중단.
        # (max_steps 까지 헛도는 시간 낭비 방지)
        cur_ee = np.array(env.get_ee_pose()[:3])
        moved = float(np.linalg.norm(cur_ee - prev_ee))
        if moved < idle_eps_m and (prev_grip is None or cur_grip == prev_grip):
            idle_count += 1
        else:
            idle_count = 0
        prev_ee = cur_ee
        prev_grip = cur_grip
        if idle_count >= idle_patience:
            fail_reason = (fail_reason or "") + f" | early_stop@{step_idx+1} (idle)"
            break

    ok, info = episode_eval(env, scenario, params, mover_z0, other_xy0)
    info["instruction"] = instruction
    info["steps_run"] = step_idx + 1
    if fail_reason is not None:
        info["last_issue"] = fail_reason

    if save_gif and frames:
        imgs = [Image.fromarray(f) for f in frames[::C.GIF_STRIDE]]
        if imgs:
            imgs[0].save(
                out_dir / "rollout.gif", save_all=True, append_images=imgs[1:],
                duration=int(1000 / C.GIF_FPS), loop=0,
            )
    Image.fromarray(frames[-1]).save(out_dir / "final.png")
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump({"success": bool(ok), **info}, f, indent=2, ensure_ascii=False)

    return bool(ok), info


# ----------------------------------------------------------------------
# 드라이버
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_url", type=str, default="http://127.0.0.1:8000")
    ap.add_argument("--unnorm_key", type=str, default="raccoon_chess")
    ap.add_argument("--xml_path", type=str, default=C.XML_PATH)
    ap.add_argument("--camera_name", type=str, default=C.CAMERA_NAME)
    ap.add_argument("--output_dir", type=str, default="chess_eval_out")
    ap.add_argument("--n_per", type=int, default=3, help="시나리오당 롤아웃 횟수")
    ap.add_argument("--max_steps", type=int, default=80, help="에피소드당 최대 액션 스텝")
    ap.add_argument("--speed", type=int, default=70)
    ap.add_argument("--settle_dt", type=float, default=1.0 / HZ)
    ap.add_argument("--settle_steps", type=int, default=5,
                    help="move_to 후 도착 대기용 settle 반복(비블로킹 보정). 작을수록 빠름")
    ap.add_argument("--delta_scale", type=float, default=3.0,
                    help="모델 xyz delta 증폭 배율. 클수록 한 스텝에 멀리 가 추론 횟수↓(빠름). 너무 크면 오버슈트")
    ap.add_argument("--idle_patience", type=int, default=15,
                    help="EE가 거의 안 움직이는 스텝이 이만큼 연속되면 조기 종료")
    ap.add_argument("--max_delta_xyz", type=float, default=0.03,
                    help="스텝당 xyz delta 최대치(m). 증폭 후 상한 클립.")
    ap.add_argument("--only_scenarios", type=int, nargs="*", default=None)
    ap.add_argument("--only_categories", type=int, nargs="*", default=None)
    ap.add_argument("--no_gif", action="store_true")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--request_timeout", type=float, default=60.0)
    args = ap.parse_args()

    # 서버 헬스체크
    try:
        h = requests.get(f"{args.server_url.rstrip('/')}/health", timeout=10)
        print(f"[server] health={h.json()}")
    except Exception as exc:
        print(f"[ERROR] 서버에 연결 못 함({args.server_url}). openvla_server.py 떠 있는지 확인.\n  {exc}")
        sys.exit(1)

    scenarios = enumerate_scenarios()
    if args.only_categories is not None:
        scenarios = [s for s in scenarios if s["category"] in set(args.only_categories)]
    if args.only_scenarios is not None:
        scenarios = [s for s in scenarios if s["idx"] in set(args.only_scenarios)]
    print(f"[eval] 대상 시나리오 {len(scenarios)}개, 시나리오당 {args.n_per}회")

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    env = ChessDataEnv(
        xml_path=args.xml_path,
        image_size=C.IMAGE_SIZE,
        camera_name=args.camera_name,
        use_viewer=False,
    )
    ik = env._calc_inv_kinematics

    per_cat = defaultdict(lambda: [0, 0])   # cat -> [success, total]
    per_scen = {}
    rng = np.random.default_rng(args.seed)
    rows = []

    try:
        for s in scenarios:
            sdir = root / scenario_folder_name(s)
            succ = 0
            for k in range(args.n_per):
                # 데이터 생성기와 같은 방식으로 유효 배치 1세트 추출
                params = sample_valid_params(ik, rng, s, ep_idx=k, n_per=args.n_per)
                if params is None:
                    print(f"[scenario {s['idx']:03d}] rollout {k}: 배치 샘플 실패(skip)")
                    continue

                ep_dir = sdir / f"rollout_{k:02d}"
                t0 = time.time()
                ok, info = rollout_episode(
                    env, args.server_url, s, params, args.unnorm_key,
                    ep_dir, args.max_steps, args.speed, args.settle_dt,
                    args.settle_steps, args.max_delta_xyz,
                    save_gif=not args.no_gif, request_timeout=args.request_timeout,
                    delta_scale=args.delta_scale,
                    idle_patience=args.idle_patience,
                )
                dt = time.time() - t0
                succ += int(ok)
                per_cat[s["category"]][0] += int(ok)
                per_cat[s["category"]][1] += 1
                rows.append({
                    "scenario": s["idx"], "category": s["category"],
                    "rollout": k, "success": bool(ok),
                    "place_err_cm": info.get("place_err_cm"),
                    "instruction": info.get("instruction"),
                })
                print(
                    f"[scenario {s['idx']:03d} cat{s['category']}] "
                    f"rollout {k}: {'OK' if ok else 'FAIL'} "
                    f"({dt:.1f}s) place_err={info.get('place_err_cm')}cm"
                )
            per_scen[s["idx"]] = [succ, args.n_per]
    finally:
        env.close()

    # 요약
    print("\n================ 카테고리별 성공률 ================")
    for cat in sorted(per_cat):
        sc, tot = per_cat[cat]
        rate = (sc / tot * 100) if tot else 0
        print(f"  cat{cat} ({CAT_NAMES[cat]}): {sc}/{tot} = {rate:.1f}%")
    total_s = sum(v[0] for v in per_cat.values())
    total_t = sum(v[1] for v in per_cat.values())
    print(f"  ------------------------------------------------")
    print(f"  전체: {total_s}/{total_t} = {(total_s/total_t*100) if total_t else 0:.1f}%")

    summary = {
        "unnorm_key": args.unnorm_key,
        "n_per": args.n_per,
        "max_steps": args.max_steps,
        "per_category": {str(c): per_cat[c] for c in per_cat},
        "per_scenario": {str(i): per_scen[i] for i in per_scen},
        "rows": rows,
    }
    with open(root / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n요약 저장: {root}/eval_summary.json")


if __name__ == "__main__":
    main()