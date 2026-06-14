"""
chess_eval_plot.py - eval_summary.json 을 읽어 성공률 표/그래프 생성
=====================================================================

사용법:
  python chess_eval_plot.py --summary chess_eval_out/eval_summary.json
  python chess_eval_plot.py --summary chess_eval_out/eval_summary.json --out_dir report_figures

출력:
  1. 터미널: 카테고리별 / 시나리오별 성공률 표
  2. cat_success_rate.png    - 카테고리별 성공률 막대그래프
  3. scenario_success_rate.png - 시나리오별 성공률 heatmap
  4. place_err_boxplot.png   - 카테고리별 place error 박스플롯
  5. success_summary.csv     - 시나리오별 raw 데이터
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

CAT_NAMES = {
    1: "Plain\nPick&Place",
    2: "Pitch\nTop-down",
    3: "Capture\n(no pitch)",
    4: "Capture\n+Pitch",
}
CAT_COLORS = {1: "#4C72B0", 2: "#DD8452", 3: "#55A868", 4: "#C44E52"}


def load_summary(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_table(summary):
    rows = summary["rows"]

    # 카테고리별 집계
    cat_stat = defaultdict(lambda: [0, 0])  # [succ, total]
    for r in rows:
        cat_stat[r["category"]][0] += int(r["success"])
        cat_stat[r["category"]][1] += 1

    print("\n========== 카테고리별 성공률 ==========")
    print(f"{'Cat':>4} {'이름':<25} {'성공':>4} {'전체':>4} {'성공률':>7}")
    print("-" * 50)
    total_s, total_t = 0, 0
    for cat in sorted(cat_stat):
        s, t = cat_stat[cat]
        name = CAT_NAMES[cat].replace("\n", " ")
        print(f"  {cat:>2} {name:<25} {s:>4} {t:>4} {s/t*100:>6.1f}%")
        total_s += s; total_t += t
    print("-" * 50)
    print(f"{'전체':<29} {total_s:>4} {total_t:>4} {total_s/total_t*100:>6.1f}%")

    # 시나리오별 집계
    scen_stat = defaultdict(lambda: {"s": 0, "t": 0, "cat": 0, "errs": []})
    for r in rows:
        d = scen_stat[r["scenario"]]
        d["s"] += int(r["success"])
        d["t"] += 1
        d["cat"] = r["category"]
        if r.get("place_err_cm") is not None:
            d["errs"].append(r["place_err_cm"])

    print("\n========== 시나리오별 성공률 (상위/하위 10개) ==========")
    scen_list = [(sid, d) for sid, d in scen_stat.items()]
    scen_list.sort(key=lambda x: x[1]["s"] / max(x[1]["t"], 1), reverse=True)

    print(f"{'Scen':>5} {'Cat':>4} {'성공':>4} {'전체':>4} {'성공률':>7} {'avg_err_cm':>10}")
    print("-" * 50)
    for sid, d in scen_list[:10]:
        rate = d["s"] / d["t"] * 100 if d["t"] else 0
        avg_err = np.mean(d["errs"]) if d["errs"] else float("nan")
        print(f"  {sid:>3} cat{d['cat']:>1} {d['s']:>4} {d['t']:>4} {rate:>6.1f}% {avg_err:>9.2f}")
    print("  ...")
    for sid, d in scen_list[-5:]:
        rate = d["s"] / d["t"] * 100 if d["t"] else 0
        avg_err = np.mean(d["errs"]) if d["errs"] else float("nan")
        print(f"  {sid:>3} cat{d['cat']:>1} {d['s']:>4} {d['t']:>4} {rate:>6.1f}% {avg_err:>9.2f}")

    return cat_stat, scen_stat


def plot_cat_bar(cat_stat, out_path):
    cats = sorted(cat_stat)
    rates = [cat_stat[c][0] / cat_stat[c][1] * 100 for c in cats]
    succs = [cat_stat[c][0] for c in cats]
    totals = [cat_stat[c][1] for c in cats]
    colors = [CAT_COLORS[c] for c in cats]
    labels = [CAT_NAMES[c] for c in cats]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, rates, color=colors, edgecolor="white", linewidth=1.2, width=0.55)

    for bar, s, t, r in zip(bars, succs, totals, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{s}/{t}\n({r:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 115)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Chess VLA — Success Rate by Category", fontsize=13, fontweight="bold")
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")


def plot_scenario_heatmap(scen_stat, out_path):
    """시나리오별 성공률을 카테고리 그룹으로 나눠 heatmap."""
    cats = sorted(set(d["cat"] for d in scen_stat.values()))

    fig, axes = plt.subplots(1, len(cats), figsize=(14, 5),
                              gridspec_kw={"wspace": 0.35})
    if len(cats) == 1:
        axes = [axes]

    for ax, cat in zip(axes, cats):
        sids = sorted([sid for sid, d in scen_stat.items() if d["cat"] == cat])
        rates = [scen_stat[sid]["s"] / scen_stat[sid]["t"] * 100 for sid in sids]

        # n×5 grid
        cols = 5
        rows = math.ceil(len(rates) / cols)
        grid = np.full((rows, cols), np.nan)
        for i, r in enumerate(rates):
            grid[i // cols, i % cols] = r

        im = ax.imshow(grid, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
        ax.set_title(f"Cat {cat}\n{CAT_NAMES[cat].replace(chr(10), ' ')}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        for i, (sid, r) in enumerate(zip(sids, rates)):
            rr, cc = i // cols, i % cols
            ax.text(cc, rr, f"{sid}\n{r:.0f}%",
                    ha="center", va="center", fontsize=7,
                    color="black" if 20 < r < 80 else "white")

    fig.colorbar(im, ax=axes[-1], label="Success Rate (%)", shrink=0.8)
    fig.suptitle("Chess VLA — Per-Scenario Success Rate", fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def plot_place_err_boxplot(rows, out_path):
    """카테고리별 place error 박스플롯 (성공 에피소드만)."""
    import math
    cats = sorted(set(r["category"] for r in rows))
    data = []
    labels = []
    colors = []
    for cat in cats:
        errs = [r["place_err_cm"] for r in rows
                if r["category"] == cat
                and r["success"]
                and r.get("place_err_cm") is not None]
        if errs:
            data.append(errs)
            labels.append(CAT_NAMES[cat].replace("\n", " "))
            colors.append(CAT_COLORS[cat])

    if not data:
        print("[skip] place_err_boxplot: 성공 데이터 없음")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Place Error (cm)", fontsize=12)
    ax.set_title("Chess VLA — Place Error by Category (Success Only)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")


def save_csv(scen_stat, rows, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "category", "rollout", "success",
                         "place_err_cm", "instruction"])
        for r in sorted(rows, key=lambda x: (x["scenario"], x["rollout"])):
            writer.writerow([r["scenario"], r["category"], r["rollout"],
                             int(r["success"]),
                             r.get("place_err_cm", ""),
                             r.get("instruction", "")])
    print(f"[saved] {out_path}")


def main():
    import math  # heatmap에서 사용
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, default="chess_eval_out/eval_summary.json")
    ap.add_argument("--out_dir", type=str, default="chess_eval_out")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = load_summary(args.summary)
    cat_stat, scen_stat = print_table(summary)

    plot_cat_bar(cat_stat, out / "cat_success_rate.png")
    plot_scenario_heatmap(scen_stat, out / "scenario_success_rate.png")
    plot_place_err_boxplot(summary["rows"], out / "place_err_boxplot.png")
    save_csv(scen_stat, summary["rows"], out / "success_summary.csv")

    print(f"\n완료. 파일 저장: {out}/")


if __name__ == "__main__":
    import math
    main()
