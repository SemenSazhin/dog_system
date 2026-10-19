#!/usr/bin/env python3
"""Графики сравнения по results/runs.json -> results/charts/*.png и results/summary.csv."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "charts"
OUT.mkdir(exist_ok=True)

LABELS = {
    "kp_none_bbox": "Без точек\n(только bbox)",
    "kp_all": "Все 20 точек",
    "kp_spine_hips": "Позвоночник\n+ таз",
    "kp_hips_tail": "Таз + хвост",
    "kp_tail_only": "Только хвост",
    "kp_legs": "Только лапы",
    "kp_spine_tail": "Позвоночник\n+ хвост",
}
C_BASE, C_ALL, C_GRP = "#9AA5B1", "#1F4E79", "#E07B39"
METRICS = [("f1", "F1"), ("accuracy", "Accuracy"), ("precision", "Precision"),
           ("recall", "Recall"), ("roc_auc", "ROC-AUC")]

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})


def color(name):
    return C_BASE if name == "kp_none_bbox" else C_ALL if name == "kp_all" else C_GRP


def main():
    df = pd.DataFrame(json.load(open(RES / "runs.json", encoding="utf-8")))
    order = [k for k in LABELS if k in set(df.experiment_base)]
    g = df.groupby("experiment_base")
    mean = g[[m for m, _ in METRICS]].mean().loc[order]
    std = g[[m for m, _ in METRICS]].std().loc[order]
    info = g[["feature_dim", "n_points"]].first().loc[order]
    summ = mean.add_suffix("_mean").join(std.add_suffix("_std")).join(info)
    summ.insert(0, "n_seeds", g.size().loc[order])
    summ.round(4).to_csv(RES / "summary.csv", encoding="utf-8-sig")
    names = [LABELS[k] for k in order]
    x = np.arange(len(order))

    # 1. F1 mean ± std
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x, mean.f1, yerr=std.f1, capsize=5, color=[color(k) for k in order], width=0.65)
    for i, k in enumerate(order):
        ax.text(i, mean.f1[k] + std.f1[k] + 0.012, f"{mean.f1[k]:.3f}", ha="center", fontsize=10)
    if "kp_none_bbox" in order:
        ax.axhline(mean.f1["kp_none_bbox"], color=C_BASE, ls="--", lw=1)
    if "kp_all" in order:
        ax.axhline(mean.f1["kp_all"], color=C_ALL, ls="--", lw=1)
    ax.set_xticks(x, names)
    ax.set_ylabel("F1 на тесте (среднее ± std по сидам)")
    ax.set_ylim(0, min(1.08, mean.f1.max() + std.f1.max() + 0.15))
    ax.set_title("F1 по конфигурациям ключевых точек")
    fig.tight_layout(); fig.savefig(OUT / "f1_by_config.png", dpi=170); plt.close(fig)

    # 2. все метрики
    fig, ax = plt.subplots(figsize=(12, 5.4))
    w = 0.16
    cols = ["#1F4E79", "#E07B39", "#4C9F70", "#B5483A", "#7D6BAA"]
    for j, (m, lab) in enumerate(METRICS):
        ax.bar(x + (j - 2) * w, mean[m], w, yerr=std[m], capsize=2, label=lab, color=cols[j])
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05); ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.set_title("Все метрики по конфигурациям")
    fig.tight_layout(); fig.savefig(OUT / "all_metrics.png", dpi=170); plt.close(fig)

    # 3. ROC-AUC
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x, mean.roc_auc, yerr=std.roc_auc, capsize=5, color=[color(k) for k in order], width=0.65)
    for i, k in enumerate(order):
        ax.text(i, mean.roc_auc[k] + std.roc_auc[k] + 0.01, f"{mean.roc_auc[k]:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x, names); ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("ROC-AUC (среднее ± std)"); ax.set_title("ROC-AUC по конфигурациям")
    fig.tight_layout(); fig.savefig(OUT / "roc_auc_by_config.png", dpi=170); plt.close(fig)

    # 4. разброс по сидам
    fig, ax = plt.subplots(figsize=(11, 5.2))
    data = [df[df.experiment_base == k].f1.values for k in order]
    bp = ax.boxplot(data, positions=x, widths=0.5, patch_artist=True, showfliers=False)
    for patch, k in zip(bp["boxes"], order):
        patch.set_facecolor(color(k)); patch.set_alpha(0.55)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i) + np.random.RandomState(0).uniform(-0.12, 0.12, len(d)), d,
                   color="#222", s=14, zorder=3)
    ax.set_xticks(x, names); ax.set_ylabel("F1 на тесте")
    ax.set_title("Разброс F1 по разным разбиениям (сидам)")
    fig.tight_layout(); fig.savefig(OUT / "f1_spread.png", dpi=170); plt.close(fig)

    # 5. число точек vs F1
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for k in order:
        ax.errorbar(info.n_points[k], mean.f1[k], yerr=std.f1[k], fmt="o", ms=9, capsize=4,
                    color=color(k))
        ax.annotate(LABELS[k].replace("\n", " "), (info.n_points[k], mean.f1[k]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Число ключевых точек"); ax.set_ylabel("F1 (среднее ± std)")
    ax.set_title("Больше точек - лучше качество?")
    fig.tight_layout(); fig.savefig(OUT / "points_vs_f1.png", dpi=170); plt.close(fig)

    # 6. выигрыш относительно «все точки»
    if "kp_all" in order:
        d = (mean.f1 - mean.f1["kp_all"]).drop("kp_all")
        e = std.f1.drop("kp_all")
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar(range(len(d)), d.values, yerr=e.values, capsize=4,
               color=[C_GRP if v >= 0 else "#B5483A" for v in d.values])
        ax.axhline(0, color="#222", lw=1)
        ax.set_xticks(range(len(d)), [LABELS[k] for k in d.index])
        ax.set_ylabel("ΔF1 относительно «все 20 точек»")
        ax.set_title("Что даёт группировка точек по сравнению с полным набором")
        fig.tight_layout(); fig.savefig(OUT / "delta_vs_all.png", dpi=170); plt.close(fig)

    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
