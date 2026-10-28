#!/usr/bin/env python3
"""Готовит клипы к просмотру: скачивает их с Яндекс Диска (ничего не удаляет и не загружает),
считает техническую пригодность и собирает ленты кадров вокруг собаки для проверки глазами.

Результат в D:\\dog-ml\\yd_sync: clips\\ (копии), sheets\\ (ленты по клипам), montage\\ (по 3 клипа на картинку),
manifest.json (клип -> пригодность и причина).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import yandex_to_gdrive as Y  # noqa: E402

CLIPS = Y.WORK / "clips"
SHEETS = Y.WORK / "sheets"
MONT = Y.WORK / "montage"
N_FRAMES, TILE_W = 8, 300


def sheet(path, dog):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(7))
    tiles, box = [], None
    for i in np.linspace(0, n - 1, N_FRAMES).astype(int):
        cap.set(1, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        r = dog.predict(fr, imgsz=1280, conf=0.25, classes=[16], verbose=False)[0]
        if len(r.boxes):
            b = r.boxes.xyxy[r.boxes.conf.argmax()].cpu().numpy()
            box = b if box is None else 0.5 * box + 0.5 * b   # сглаживаем рамку между кадрами
        h, w = fr.shape[:2]
        if box is not None:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            s = max(box[2] - box[0], box[3] - box[1]) * 2.2 + 80
            x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
            crop = fr[y0:int(min(h, cy + s / 2)), x0:int(min(w, cx + s / 2))]
        else:
            crop = fr
        tiles.append(cv2.resize(crop, (TILE_W, TILE_W * 3 // 4)))
    cap.release()
    if not tiles:
        return None
    while len(tiles) < N_FRAMES:
        tiles.append(np.zeros_like(tiles[0]))
    row = lambda t: np.hstack(t)  # noqa: E731
    return np.vstack([row(tiles[:4]), row(tiles[4:8])])


def main():
    from ultralytics import YOLO
    for d in (CLIPS, SHEETS, MONT):
        d.mkdir(parents=True, exist_ok=True)
    token = Y.yandex_token()
    dog = YOLO(Y.DOG_WEIGHTS)
    items = []
    for folder in ("no_defecation", "defecation"):
        for cam in Y.ylist(token, f"{Y.BASE}/{folder}"):
            if cam["type"] == "dir":
                for f in Y.ylist(token, cam["path"]):
                    if f["type"] == "file":
                        items.append((folder, cam["name"], f))
    mpath = Y.WORK / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    for folder, cam, f in items:
        key = f["path"]
        if key in manifest and (SHEETS / manifest[key]["sheet"]).exists():
            continue
        name = f"{folder}__{cam}__{f['name']}"
        local = CLIPS / name
        try:
            if not local.exists():
                Y.ydownload(token, key, local)
            ok, why = Y.check_clip(local, dog)
            img = sheet(local, dog) if ok else None
            sname = name.replace(".mp4", ".jpg")
            if img is not None:
                cv2.imwrite(str(SHEETS / sname), img)
            manifest[key] = {"clip": name, "folder": folder, "cam": cam, "tech_ok": ok, "why": why,
                             "sheet": sname if img is not None else None, "md5": f.get("md5")}
        except Exception as e:  # noqa: BLE001
            manifest[key] = {"clip": name, "folder": folder, "cam": cam, "tech_ok": False, "why": f"ошибка: {e}"[:100],
                             "sheet": None}
        mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = [v for v in manifest.values() if v.get("sheet")]
    print(f"клипов {len(manifest)}, лент для просмотра {len(ok)}")
    # montage: по 3 клипа на картинку, подписи в кадре
    for i in range(0, len(ok), 3):
        parts = []
        for v in ok[i:i + 3]:
            im = cv2.imread(str(SHEETS / v["sheet"]))
            cv2.putText(im, f"#{ok.index(v)} {v['cam'][:28]} {v['clip'][-24:-4]}", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            parts.append(im)
        cv2.imwrite(str(MONT / f"m{i // 3:03d}.jpg"), np.vstack(parts))
    (Y.WORK / "order.json").write_text(json.dumps([v["clip"] for v in ok], ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
