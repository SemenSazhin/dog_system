#!/usr/bin/env python3
"""Ежедневная проверка клипов video-scan на Яндекс Диске.

Для каждого клипа из no_defecation: скачать, проверить, залить годные на Google Диск, после
успешной проверки загрузки перенести файл в корзину Яндекс Диска. Негодные тоже уходят в корзину
(только с --apply; корзина обратима, безвозвратно файлы не удаляются никогда).
Папки health_check, *_with_boxes и defecation не переносятся: defecation только попадает в отчёт.

Без --apply ничего не загружается и не удаляется, только пишется отчёт.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ENV = Path(r"C:\Users\Zhenya\Desktop\Детекция Собак\video-scan\.env")
BASE = "disk:/Датасет Собак"
WORK = Path(r"D:\dog-ml\yd_sync")
GTOKEN = WORK / "gtoken.json"          # создаёт get_refresh_token.py, в git не попадает
STATE = WORK / "state.json"
API = "https://cloud-api.yandex.net/v1/disk/resources"
MIN_SEC, MIN_SIDE, MIN_READ, MIN_MOTION = 3.0, 360, 0.9, 0.15
DOG_WEIGHTS = r"D:\dog-ml\bench\yolo26s.pt"


def yandex_token():
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith("YANDEX_DISK_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("YANDEX_DISK_TOKEN не найден в .env")


def ycall(token, method, url, **kw):
    req = urllib.request.Request(url, method=method, headers={"Authorization": f"OAuth {token}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
        return json.loads(body) if body else {}


def ylist(token, path):
    out, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"path": path, "limit": 500, "offset": offset,
                                    "fields": "_embedded.items.name,_embedded.items.type,_embedded.items.size,_embedded.items.md5,_embedded.items.path,_embedded.total"})
        emb = ycall(token, "GET", f"{API}?{q}")["_embedded"]
        out += emb["items"]
        offset += 500
        if offset >= emb["total"]:
            return out


def ydownload(token, path, dest, tries=3):
    for k in range(tries):
        try:
            href = ycall(token, "GET", "https://cloud-api.yandex.net/v1/disk/resources/download?"
                         + urllib.parse.urlencode({"path": path}))["href"]
            urllib.request.urlretrieve(href, dest)
            return
        except (urllib.error.URLError, OSError):
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def ytrash(token, path):
    # permanently=false: файл идёт в корзину и восстанавливается
    ycall(token, "DELETE", f"{API}?" + urllib.parse.urlencode({"path": path, "permanently": "false"}))


def check_clip(path, dog_model):
    cap = cv2.VideoCapture(str(path))
    try:
        return _check_clip(cap, dog_model)
    finally:
        cap.release()


def _check_clip(cap, dog_model):
    if not cap.isOpened():
        return False, "не открывается"
    n, fps = int(cap.get(7)), cap.get(5) or 0
    w, h = int(cap.get(3)), int(cap.get(4))
    if n <= 0 or fps <= 0:
        return False, "нет кадров или fps"
    if n / fps < MIN_SEC:
        return False, f"короткий ({n / fps:.1f} с)"
    if min(w, h) < MIN_SIDE:
        return False, f"малое разрешение {w}x{h}"
    idx = np.linspace(0, n - 1, 6).astype(int)
    frames, bad = [], 0
    for i in idx:
        cap.set(1, int(i))
        ok, fr = cap.read()
        if ok and fr is not None:
            frames.append(fr)
        else:
            bad += 1
    if len(frames) / len(idx) < MIN_READ:
        return False, "кадры не читаются"
    g = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    motion = float(np.mean([np.abs(a - b).mean() for a, b in zip(g, g[1:])]))
    if motion < MIN_MOTION:
        return False, f"статичный кадр (движение {motion:.2f})"
    if dog_model is not None:
        found = sum(len(dog_model.predict(f, imgsz=1280, conf=0.25, classes=[16], verbose=False)[0].boxes) > 0
                    for f in frames)
        if found == 0:
            return False, "собака не найдена"
    return True, "ok"


def gdrive_service():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    info = json.loads(GTOKEN.read_text(encoding="utf-8"))
    creds = Credentials(None, refresh_token=info["refresh_token"], token_uri="https://oauth2.googleapis.com/token",
                        client_id=info["client_id"], client_secret=info["client_secret"],
                        scopes=["https://www.googleapis.com/auth/drive.file"])
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def gdrive_upload(svc, local, name, parent):
    from googleapiclient.http import MediaFileUpload
    meta = {"name": name, "parents": [parent]}
    media = MediaFileUpload(str(local), mimetype="video/mp4", resumable=True)
    f = svc.files().create(body=meta, media_body=media, fields="id,md5Checksum,size").execute()
    md5 = hashlib.md5(Path(local).read_bytes()).hexdigest()
    return f["id"], f.get("md5Checksum") == md5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="загружать и переносить в корзину")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gdrive-parent", default="", help="id папки Google Диска для клипов")
    ap.add_argument("--no-dog-check", action="store_true")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    token = yandex_token()
    dog = None
    if not a.no_dog_check:
        from ultralytics import YOLO
        dog = YOLO(DOG_WEIGHTS)
    svc = gdrive_service() if (a.apply and GTOKEN.exists() and a.gdrive_parent) else None
    if a.apply and svc is None:
        print("Google Диск не настроен: годные клипы НЕ переносятся и не удаляются, негодные уйдут в корзину")

    items = []
    for cam in ylist(token, f"{BASE}/no_defecation"):
        if cam["type"] == "dir":
            items += [f for f in ylist(token, cam["path"]) if f["type"] == "file"]
    if a.limit:
        items = items[:a.limit]
    report = {"ok": [], "bad": [], "moved": [], "trashed": [], "errors": []}
    tmp = Path(tempfile.mkdtemp(dir=WORK))
    for it in items:
        p = it["path"]
        if state.get(p, {}).get("done"):
            continue
        local = tmp / it["name"]
        try:
            ydownload(token, p, local)
            ok, why = check_clip(local, dog)
            if ok:
                report["ok"].append(p)
                if svc:
                    gid, good = gdrive_upload(svc, local, it["name"], a.gdrive_parent)
                    if good:
                        ytrash(token, p)
                        state[p] = {"done": True, "gdrive": gid}
                        report["moved"].append(p)
                    else:
                        report["errors"].append((p, "md5 не совпал после загрузки"))
            else:
                report["bad"].append((p, why))
                if a.apply:
                    ytrash(token, p)
                    state[p] = {"done": True, "trashed": why}
                    report["trashed"].append(p)
        except (urllib.error.URLError, OSError, Exception) as e:  # noqa: BLE001
            report["errors"].append((p, str(e)[:120]))
        finally:
            local.unlink(missing_ok=True)
    try:
        tmp.rmdir()
    except OSError:
        pass
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    stamp = time.strftime("%Y%m%d_%H%M")
    (WORK / f"report_{stamp}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"всего {len(items)}: годных {len(report['ok'])}, негодных {len(report['bad'])}, "
          f"перенесено {len(report['moved'])}, в корзину {len(report['trashed'])}, ошибок {len(report['errors'])}")
    for p, why in report["bad"][:15]:
        print("  негодный:", p.split("/")[-1], "-", why)
    for p, why in report["errors"][:5]:
        print("  ошибка:", p.split("/")[-1], "-", why)


if __name__ == "__main__":
    sys.exit(main())
