#!/usr/bin/env python3
"""Очередь экспериментов skeleton_cv.py: параллельно N процессов, возобновляемо (маркеры .done)."""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PY = r"D:\dog-ml\venv\Scripts\python.exe"
CV = str(Path(__file__).with_name("skeleton_cv.py"))
OUT = Path(r"D:\dog-ml\cv")
OUT.mkdir(parents=True, exist_ok=True)

NETS3 = ["lstm", "tcn", "stgcn"]
S5 = ["--seeds", "5"]
S3 = ["--seeds", "3"]
JOBS = []


def job(tag, models, *extra, heavy=False):
    JOBS.append(dict(tag=tag, models=models, extra=list(extra), heavy=heavy))


# 1. зоопарк моделей на базовом протоколе
job("zoo_nets", ["lstm", "tcn", "stgcn", "transformer", "mamba"], *S5)
job("zoo_nets_noasp", ["lstm", "tcn", "stgcn", "transformer", "mamba"], *S5, "--no-asp")
job("zoo_gbm", ["meta", "aspect", "gbm"], *S3, heavy=True)
job("zoo_gbm_noasp", ["gbm"], *S3, "--no-asp", heavy=True)

# 2. параметры окна и порог доли кадров с позой
for T in (30, 60, 90):
    job(f"win_T{T}", NETS3, *S3, "--T", str(T))
for T, mv in ((30, 0.7), (45, 0.7), (45, 0.5), (30, 0.5), (90, 0.7)):
    job(f"win_T{T}_mv{int(mv * 100)}", NETS3, *S3, "--T", str(T), "--min-valid", str(mv))
job("win_stride2", NETS3, *S3, "--stride", "2")

# 3. группы точек (без признаков рамки, чтобы измерять именно точки)
GRP = {
    "only_spine": ["--keep", "spine"], "only_hips": ["--keep", "hips"], "only_tail": ["--keep", "tail"],
    "only_front": ["--keep", "front"], "only_rear": ["--keep", "rear"],
    "spine_hips": ["--keep", "spine", "hips"], "hips_tail": ["--keep", "hips", "tail"],
    "spine_tail": ["--keep", "spine", "tail"], "legs": ["--keep", "front", "rear"],
    "tail_rear": ["--keep", "tail", "rear"], "hips_tail_rear": ["--keep", "hips", "tail", "rear"],
    "drop_spine": ["--drop", "spine"], "drop_hips": ["--drop", "hips"], "drop_tail": ["--drop", "tail"],
    "drop_front": ["--drop", "front"], "drop_rear": ["--drop", "rear"],
}
job("grp_all", ["tcn", "stgcn"], *S5, "--no-asp")
for name, a in GRP.items():
    job(f"grp_{name}", ["tcn", "stgcn"], *S5, "--no-asp", *a)

# 4. абляции признаков и обучения
job("abl_no_conf", NETS3, *S5, "--no-asp", "--no-conf")
job("abl_no_aug", NETS3, *S5, "--no-asp", "--no-aug")
job("abl_no_vel", NETS3, *S5, "--no-asp", "--no-vel")
job("abl_minimal", NETS3, *S5, "--no-asp", "--no-conf", "--no-vel", "--no-aug")
job("abl_ep20", ["tcn", "stgcn"], *S5, "--no-asp", "--epochs", "20")
job("abl_ep80", ["tcn", "stgcn"], *S5, "--no-asp", "--epochs", "80")

# 5. сдвиг домена: обучение без камерных клипов
job("dom_excl_cam", NETS3, *S5, "--exclude-cam-train")
job("dom_excl_cam_noasp", NETS3, *S5, "--no-asp", "--exclude-cam-train")

# 6. лучшая конфигурация: окно 30 шагов, порог 0.7, без confidence/скоростей/аугментаций/рамки
BEST = ["--T", "30", "--min-valid", "0.7", "--no-asp", "--no-conf", "--no-vel", "--no-aug"]
job("best_nets", ["lstm", "tcn", "stgcn", "transformer", "mamba"], *S5, *BEST)
job("best_gbm", ["gbm"], *S3, "--T", "30", "--min-valid", "0.7", "--no-asp", heavy=True)
job("best_ctrl", ["meta", "aspect"], *S3, "--T", "30", "--min-valid", "0.7")
job("best_rep", ["lstm", "tcn", "transformer", "mamba"], *S5, *BEST, "--seed-offset", "1000")
BG = {
    "hips_tail": ["--keep", "hips", "tail"], "drop_hips": ["--drop", "hips"], "drop_tail": ["--drop", "tail"],
    "drop_front": ["--drop", "front"], "drop_rear": ["--drop", "rear"], "drop_spine": ["--drop", "spine"],
    "only_front": ["--keep", "front"], "only_hips": ["--keep", "hips"], "only_tail": ["--keep", "tail"],
    "only_rear": ["--keep", "rear"], "only_spine": ["--keep", "spine"], "spine_hips": ["--keep", "spine", "hips"],
}
for name, a in BG.items():
    job(f"bgrp_{name}", ["lstm", "tcn"], *S5, *BEST, *a)
job("best_excl_cam", ["lstm", "tcn"], *S5, *BEST, "--exclude-cam-train")


def run(j):
    tag = j["tag"]
    done = OUT / f"{tag}.done"
    if done.exists():
        return tag, "skip", 0
    cmd = [PY, "-u", CV, "--tag", tag, "--out", str(OUT), "--models", *j["models"], *j["extra"]]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4" if j["heavy"] else "2",
               MKL_NUM_THREADS="4" if j["heavy"] else "2")
    t0 = time.time()
    with open(OUT / f"log_{tag}.txt", "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    if rc == 0:
        done.write_text("ok")
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: rc={rc} {time.time() - t0:.0f}s", flush=True)
    return tag, rc, time.time() - t0


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"{len(JOBS)} jobs, {workers} workers", flush=True)
    order = sorted(JOBS, key=lambda j: not j["heavy"])  # тяжёлые первыми
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, order))
    print("QUEUE_DONE", flush=True)
