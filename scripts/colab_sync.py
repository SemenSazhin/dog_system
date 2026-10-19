#!/usr/bin/env python3
"""Двусторонняя синхронизация *.ipynb репозитория с Google Drive (Colab) + генерация README.

Репозиторий -> Drive: новые ноутбуки создаются, изменённые обновляются (ID файла, а значит
и ссылка на Colab, не меняется).
Drive -> репозиторий: если ноутбук изменили в Colab, новая версия скачивается в файл.
Состояние (ID файлов и хэши последней синхронизации) хранится в .colab-sync.json.

Переменные окружения: GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN,
опционально COLAB_FOLDER_NAME (папка на Drive, по умолчанию имя репозитория).
"""
import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / ".colab-sync.json"
README = ROOT / "README.md"
NB_MIME = "application/vnd.google.colaboratory"
FOLDER_MIME = "application/vnd.google-apps.folder"


def warn(msg):
    print(f"::warning::{msg}", flush=True)


def drive_service():
    creds = Credentials(
        None,
        refresh_token=os.environ["GOOGLE_REFRESH_TOKEN"],
        client_id=os.environ["GOOGLE_CLIENT_ID"],
        client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive.file"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def find_notebooks():
    skip = {".git", ".ipynb_checkpoints", "node_modules"}
    out = []
    for p in ROOT.rglob("*.ipynb"):
        rel = p.relative_to(ROOT)
        if not skip.intersection(rel.parts):
            out.append(rel.as_posix())
    return sorted(out)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def get_meta(svc, file_id):
    try:
        m = svc.files().get(fileId=file_id, fields="id,md5Checksum,trashed").execute()
    except HttpError as e:
        if e.resp.status == 404:
            return None
        raise
    return None if m.get("trashed") else m


def ensure_folder(svc, state):
    fid = state.get("folder_id")
    if fid and get_meta(svc, fid):
        return
    name = os.environ.get("COLAB_FOLDER_NAME") or os.environ.get(
        "GITHUB_REPOSITORY", ROOT.name).split("/")[-1]
    f = svc.files().create(
        body={"name": name, "mimeType": FOLDER_MIME}, fields="id").execute()
    state["folder_id"] = f["id"]
    print(f"created Drive folder {name!r}")


def upload(svc, state, rel, data, file_id=None):
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype="application/json", resumable=False)
    if file_id:
        f = svc.files().update(fileId=file_id, media_body=media,
                               fields="id,md5Checksum").execute()
    else:
        body = {"name": rel.replace("/", "__"), "mimeType": NB_MIME,
                "parents": [state["folder_id"]]}
        f = svc.files().create(body=body, media_body=media,
                               fields="id,md5Checksum").execute()
    return f["id"], f["md5Checksum"]


def download(svc, file_id):
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, svc.files().get_media(fileId=file_id))
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()


def sync(svc, state, on_conflict):
    files = state.setdefault("files", {})
    notebooks = find_notebooks()
    for rel in notebooks:
        path = ROOT / rel
        data = path.read_bytes()
        entry = files.get(rel)
        meta = get_meta(svc, entry["id"]) if entry else None

        if not entry or not meta:
            if entry:
                warn(f"{rel}: файл удалён с Drive, создаю заново")
            fid, md5 = upload(svc, state, rel, data)
            files[rel] = {"id": fid, "repo_sha": sha(data), "drive_md5": md5}
            print(f"[new]    {rel}")
            continue

        repo_changed = sha(data) != entry["repo_sha"]
        drive_changed = meta["md5Checksum"] != entry["drive_md5"]

        if repo_changed and drive_changed:
            if on_conflict == "repo":
                warn(f"{rel}: изменён и в репо, и в Colab - приоритет у репо "
                     "(версия из Colab осталась в истории версий файла на Drive)")
                drive_changed = False
            else:
                warn(f"{rel}: конфликт (изменён и в репо, и в Colab), пропускаю")
                continue

        if repo_changed:
            _, md5 = upload(svc, state, rel, data, entry["id"])
            entry.update(repo_sha=sha(data), drive_md5=md5)
            print(f"[push]   {rel}")
        elif drive_changed:
            new = download(svc, entry["id"])
            path.write_bytes(new)
            entry.update(repo_sha=sha(new), drive_md5=meta["md5Checksum"])
            print(f"[pull]   {rel}")
        else:
            print(f"[same]   {rel}")

    # ноутбуки, которых больше нет в репо, убираем из состояния (на Drive не трогаем)
    for rel in [r for r in files if r not in notebooks]:
        del files[rel]


def write_readme(state):
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    branch = os.environ.get("GITHUB_REF_NAME", "master")
    lines = [
        "# Ноутбуки в Google Colab", "",
        "Файл генерируется автоматически (`scripts/colab_sync.py`), вручную не править.",
        "Ноутбуки из репозитория синхронизируются с Google Drive в обе стороны: "
        "пуш обновляет Colab, правки из Colab подтягиваются обратно в репозиторий.", "",
        "| Ноутбук | Colab | GitHub |", "|---|---|---|",
    ]
    for rel, e in sorted(state.get("files", {}).items()):
        colab = f"https://colab.research.google.com/drive/{e['id']}"
        badge = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})"
        gh = f"[файл](https://github.com/{repo}/blob/{branch}/{quote(rel)})" if repo else "-"
        lines.append(f"| `{rel}` | {badge} | {gh} |")
    lines.append("")
    README.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on-conflict", choices=["repo", "skip"], default="skip")
    args = ap.parse_args()

    state = json.loads(STATE_FILE.read_text("utf-8")) if STATE_FILE.exists() else {}
    svc = drive_service()
    ensure_folder(svc, state)
    sync(svc, state, args.on_conflict)
    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(state)


if __name__ == "__main__":
    sys.exit(main())
