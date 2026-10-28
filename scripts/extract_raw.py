#!/usr/bin/env python3
"""Один проход YOLO (детектор собаки + pose) по всем видео датасета, сохраняет сырой скелет.

Для каждого видео пишет <out>/<id>.npz:
  bbox[N,5]   x1,y1,x2,y2,conf лучшей собаки (NaN, если собаки нет)
  kps[N,20,2] координаты точек в координатах кадра (NaN, если позы нет)
  kconf[N,20] confidence точек
  fps, w, h, cls, name
Логика отбора детекции и позы как в DataPreparator ноутбука (conf>0.3, bbox>=30px, ROI без полей).

Запуск: python extract_raw.py <data_dir> <detect.pt> <pose.pt> <out_dir> [shard n_shards]
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

data_dir, det_path, pose_path, out_dir = sys.argv[1:5]
shard, n_shards = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (0, 1)
os.makedirs(out_dir, exist_ok=True)

det = YOLO(det_path)
pose = YOLO(pose_path)
NUM_KP, CONF, MIN_BOX = 20, 0.3, 30

items = []
for cls in ("0", "1"):
    for f in sorted(os.listdir(os.path.join(data_dir, cls))):
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            items.append((int(cls), os.path.join(data_dir, cls, f)))


def vid_id(cls, path):
    return f"{cls}_" + hashlib.md5(os.path.basename(path).encode("utf-8")).hexdigest()[:10]


index = {}
for k, (cls, path) in enumerate(items):
    vid = vid_id(cls, path)
    index[vid] = dict(cls=cls, name=os.path.basename(path))
    if k % n_shards != shard:
        continue
    out = os.path.join(out_dir, vid + ".npz")
    if os.path.exists(out):
        continue
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(3)), int(cap.get(4))
    boxes, kps, kconf = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        b = np.full(5, np.nan, np.float32)
        kp = np.full((NUM_KP, 2), np.nan, np.float32)
        kc = np.zeros(NUM_KP, np.float32)
        r = det(frame, imgsz=int(os.environ.get("DET_IMGSZ", "640")), verbose=False)[0]
        best = None
        if r.boxes is not None:
            for xyxy, c, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                                   r.boxes.conf.cpu().numpy()):
                if r.names[int(c)] == "dog" and cf > CONF:
                    x1, y1, x2, y2 = map(int, xyxy)
                    if x2 - x1 >= MIN_BOX and y2 - y1 >= MIN_BOX and (best is None or cf > best[4]):
                        best = (x1, y1, x2, y2, float(cf))
        if best is not None:
            b[:] = best
            x1, y1, x2, y2 = best[:4]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                pr = pose(roi, verbose=False)[0]
                if pr.keypoints is not None and len(pr.keypoints) > 0:
                    xy = pr.keypoints[0].xy[0].cpu().numpy()
                    cc = pr.keypoints[0].conf
                    cc = cc[0].cpu().numpy() if cc is not None else np.ones(len(xy), np.float32)
                    n = min(len(xy), NUM_KP)
                    kp[:n] = xy[:n] + np.array([x1, y1], np.float32)
                    kc[:n] = cc[:n]
        boxes.append(b)
        kps.append(kp)
        kconf.append(kc)
    cap.release()
    np.savez_compressed(out, bbox=np.array(boxes), kps=np.array(kps), kconf=np.array(kconf),
                        fps=fps, w=w, h=h, cls=cls, name=os.path.basename(path))
    print(f"[shard {shard}] {k + 1}/{len(items)} {vid} frames={len(boxes)} "
          f"dog={int(np.isfinite(np.array(boxes)[:, 0]).sum())}", flush=True)

if shard == 0:
    json.dump(index, open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("DONE shard", shard, flush=True)
