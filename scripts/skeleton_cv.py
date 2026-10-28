#!/usr/bin/env python3
"""Эксперименты по сырому скелету (см. extract_raw.py) на честном протоколе.

* окна по T шагов при 15 fps (не зависят от fps ролика), шаг stride;
* признаки не зависят от разрешения: центр и масштаб тела, разворот собаки по оси позвоночника;
* кросс-валидация по группам источников (нарезки одного ролика и одноимённые ролики
  из разных классов - одна группа), метрики на уровне видео и по доменам;
* модели: gbm (статистики окна), lstm, tcn, stgcn (граф по скелету + внимание по суставам);
* контроли: meta (только разрешение/fps/длительность) и aspect (только форма рамки).

python skeleton_cv.py --raw D:/dog-ml/raw --out D:/dog-ml/cv --models gbm lstm tcn stgcn meta --seeds 3
"""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FPS_OUT = 15
CFG = dict(T=45, stride=5, min_valid=0.9, keep=None, conf=True, aug=True, vel=True, asp=True,
           exclude_cam_train=False)
GROUPS = {"spine": [0, 1, 2, 3, 4, 5], "hips": [6, 7, 8, 9], "tail": [10, 11, 12],
          "front": [13, 14, 15], "rear": [16, 17, 18, 19]}
NKP = 20
CONNS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),
         (11, 12), (2, 13), (13, 14), (14, 15), (6, 16), (16, 17), (17, 18), (8, 19), (5, 9), (2, 6), (13, 16)]
ANGLES = [(5, 6, 7), (9, 10, 11), (2, 5, 8), (6, 8, 16), (1, 2, 5)]


# ----------------------------------------------------------------------------- данные
def group_key(name):
    n = os.path.splitext(name)[0].lower()
    m = re.match(r"^(?:no)?df_(.+?)_(\d{2}_\d{2}_\d{4})", n)
    if m:
        return f"cam:{m.group(1)}:{m.group(2)}"
    n = re.sub(r"\(online-video-cutter\.com\)", "", n)
    n = re.sub(r"\s*\(\d+\)", "", n)
    n = re.sub(r"[\s_-]*trim\d*\s*$", "", n)
    n = re.sub(r"^download.*$", "download", n)
    return re.sub(r"\s+", " ", n).strip()


def domain_of(name):
    return "cam" if re.match(r"^(no)?df_", name.lower()) else "other"


def resample_idx(n_frames, fps):
    n_out = int(n_frames / fps * FPS_OUT)
    return np.minimum(np.round(np.arange(n_out) / FPS_OUT * fps).astype(int), n_frames - 1)


def fill_gaps(x, valid, max_gap=8):
    """Линейная интерполяция коротких пропусков по времени; x: (T,...), valid: (T,)."""
    x = x.copy()
    valid = valid.copy()
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return x, valid
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            for t in range(a + 1, b):
                w = (t - a) / (b - a)
                x[t] = (1 - w) * x[a] + w * x[b]
                valid[t] = True
    return x, valid


def build_windows(raw_dir, T_WIN=None, STRIDE=None, min_valid=None, min_pts=8):
    T_WIN, STRIDE, min_valid = T_WIN or CFG["T"], STRIDE or CFG["stride"], min_valid or CFG["min_valid"]
    raw_dir = Path(raw_dir)
    kps_all, kc_all, asp_all, meta = [], [], [], []
    vids = []
    for f in sorted(raw_dir.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        name, cls = str(z["name"]), int(z["cls"])
        fps = float(z["fps"])
        idx = resample_idx(len(z["bbox"]), fps)
        vids.append(dict(vid=f.stem, name=name, cls=cls, fps=fps, w=int(z["w"]), h=int(z["h"]),
                         dur=len(z["bbox"]) / fps, group=group_key(name), domain=domain_of(name)))
        if len(idx) < T_WIN:
            continue
        kp = z["kps"][idx]                      # (N,20,2)
        kc = z["kconf"][idx]                    # (N,20)
        bb = z["bbox"][idx]                     # (N,5)
        valid = np.isfinite(kp[..., 0]).sum(1) >= min_pts
        kc = np.where(np.isfinite(kp[..., 0]), kc, 0.0)
        kpf = np.nan_to_num(kp, nan=0.0)
        flat = np.concatenate([kpf.reshape(len(idx), -1), kc, np.nan_to_num(bb[:, :4])], 1)
        flat, valid2 = fill_gaps(flat, valid)
        kpf = flat[:, :40].reshape(-1, NKP, 2)
        kc = flat[:, 40:60]
        bbx = flat[:, 60:64]
        asp = np.clip((bbx[:, 3] - bbx[:, 1]) / np.maximum(bbx[:, 2] - bbx[:, 0], 1.0), 0, 3) / 3.0
        for s in range(0, len(idx) - T_WIN + 1, STRIDE):
            sl = slice(s, s + T_WIN)
            if valid2[sl].mean() >= min_valid:
                kps_all.append(kpf[sl].astype(np.float32))
                kc_all.append(kc[sl].astype(np.float32))
                asp_all.append(asp[sl].astype(np.float32))
                meta.append(len(vids) - 1)
    return (np.array(kps_all), np.array(kc_all), np.array(asp_all), np.array(meta), vids)


# ----------------------------------------------------------------------------- признаки (torch)
def featurize(kp, kc, asp):
    """kp (B,T,20,2) в пикселях, kc (B,T,20), asp (B,T) -> vec (B,T,127), nodes (B,T,20,5)."""
    m = (kc > 0).float()
    cnt = m.sum(2, keepdim=True).clamp(min=1)
    center = (kp * m[..., None]).sum(2, keepdim=True) / cnt[..., None]
    both = m[:, :, 0] * m[:, :, 5]
    spine = (kp[:, :, 5] - kp[:, :, 0])
    spine_len = spine.norm(dim=-1)
    sl = torch.where(both > 0, spine_len, torch.full_like(spine_len, float("nan")))
    scale = torch.nanmedian(sl, dim=1).values
    fallback = ((kp - center).norm(dim=-1) * m).sum(2).mean(1) / cnt.squeeze(-1).mean(1) * 3.0
    scale = torch.where(torch.isnan(scale), fallback, scale).clamp(min=8.0)[:, None, None, None]
    sx = (spine[..., 0] * both).sum(1)
    sign = torch.where(sx < 0, -torch.ones_like(sx), torch.ones_like(sx))[:, None, None]
    if CFG["keep"] is not None:  # центр, масштаб и разворот считаются по полному скелету, маска - после
        km = torch.zeros(NKP, device=kp.device)
        km[list(CFG["keep"])] = 1.0
        m = m * km
    c = (kp - center) / scale
    c = torch.stack([c[..., 0] * sign, c[..., 1]], -1) * m[..., None]
    vel = torch.zeros_like(c)
    vel[:, 1:] = (c[:, 1:] - c[:, :-1]) * FPS_OUT * (m[:, 1:, :, None] * m[:, :-1, :, None])
    vel = vel.clamp(-5, 5)
    if not CFG["vel"]:
        vel = torch.zeros_like(vel)
    if not CFG["conf"]:
        kc = torch.zeros_like(kc)
    else:
        kc = kc * m
    if not CFG["asp"]:
        asp = torch.zeros_like(asp)
    lens = []
    for i, j in CONNS:
        ok = (m[:, :, i] * m[:, :, j])
        lens.append((c[:, :, i] - c[:, :, j]).norm(dim=-1).clamp(max=3) * ok)
    lens = torch.stack(lens, -1)
    angs = []
    for i, j, k in ANGLES:
        v1, v2 = c[:, :, i] - c[:, :, j], c[:, :, k] - c[:, :, j]
        cos = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1)).clamp(min=1e-6)
        ok = m[:, :, i] * m[:, :, j] * m[:, :, k]
        angs.append(torch.acos(cos.clamp(-1, 1)) / math.pi * ok)
    angs = torch.stack(angs, -1)
    vec = torch.cat([c.flatten(2), kc, lens, angs, vel.flatten(2), asp[..., None]], -1)
    nodes = torch.cat([c, kc[..., None], vel], -1)
    return vec, nodes


N_FEAT = 40 + 20 + len(CONNS) + len(ANGLES) + 40 + 1


def augment(kp, kc, train=True):
    if not train or not CFG["aug"]:
        return kp, kc
    B = kp.shape[0]
    ang = (torch.rand(B, 1, 1, device=kp.device) - 0.5) * 2 * math.radians(12)
    cos, sin = torch.cos(ang), torch.sin(ang)
    ctr = kp.mean((1, 2), keepdim=True)
    d = kp - ctr
    kp = ctr + torch.stack([d[..., 0] * cos - d[..., 1] * sin, d[..., 0] * sin + d[..., 1] * cos], -1)
    kp = kp * (1 + (torch.rand(B, 1, 1, 1, device=kp.device) - 0.5) * 0.3)
    kp = kp + torch.randn_like(kp) * 0.01 * kp.std((1, 2), keepdim=True)
    drop = (torch.rand(B, 1, NKP, device=kp.device) < 0.1).float()
    kc = kc * (1 - drop)
    return kp, kc


# ----------------------------------------------------------------------------- модели
class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(d, 32), nn.Tanh(), nn.Linear(32, 1, bias=False))

    def forward(self, h):  # (B,T,D)
        w = torch.softmax(self.a(h), 1)
        return (w * h).sum(1)


class LSTMNet(nn.Module):
    def __init__(self, d_in=N_FEAT, hid=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.3))
        self.lstm = nn.LSTM(64, hid, 1, batch_first=True, bidirectional=True)
        self.pool = AttnPool(hid * 2)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(hid * 2, 1))

    def forward(self, vec, nodes):
        h, _ = self.lstm(self.enc(vec))
        return self.head(self.pool(h)).squeeze(-1)


class TCNNet(nn.Module):
    def __init__(self, d_in=N_FEAT, ch=48):
        super().__init__()
        self.inp = nn.Conv1d(d_in, ch, 1)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(0.3))
            for d in (1, 2, 4)])
        self.pool = AttnPool(ch)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(ch, 1))

    def forward(self, vec, nodes):
        h = self.inp(vec.transpose(1, 2))
        for b in self.blocks:
            h = h + b(h)
        return self.head(self.pool(h.transpose(1, 2))).squeeze(-1)


class STGCN(nn.Module):
    """Граф по скелету + внимание по суставам. self.joint_attn - вес каждого сустава (для анализа)."""

    def __init__(self, c_in=5, ch=32):
        super().__init__()
        A = torch.eye(NKP)
        for i, j in CONNS:
            A[i, j] = A[j, i] = 1
        self.register_buffer("A0", A / A.sum(1, keepdim=True))
        self.A1 = nn.Parameter(torch.zeros(NKP, NKP))
        self.g1, self.g2 = nn.Linear(c_in, ch), nn.Linear(ch, ch)
        self.t1 = nn.Conv2d(ch, ch, (5, 1), padding=(2, 0))
        self.t2 = nn.Conv2d(ch, ch, (5, 1), padding=(4, 0), dilation=(2, 1))
        self.bn1, self.bn2 = nn.BatchNorm2d(ch), nn.BatchNorm2d(ch)
        self.jscore = nn.Linear(ch, 1)
        self.tpool = AttnPool(ch)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(ch, 1))
        self.last_joint_w = None

    def forward(self, vec, x):  # x (B,T,V,C)
        A = self.A0 + torch.tanh(self.A1) * 0.2
        h = F.relu(torch.einsum("vw,btwc->btvc", A, self.g1(x)))
        h = self.bn1(self.t1(h.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        h = h + F.relu(torch.einsum("vw,btwc->btvc", A, self.g2(h)))
        h = F.relu(self.bn2(self.t2(h.permute(0, 3, 1, 2)))).permute(0, 2, 3, 1)
        jw = torch.softmax(self.jscore(h).squeeze(-1), dim=2)          # (B,T,V)
        self.last_joint_w = jw.detach()
        z = (jw[..., None] * h).sum(2)                                  # (B,T,C)
        return self.head(self.tpool(z)).squeeze(-1)



class TransformerNet(nn.Module):
    def __init__(self, d_in=N_FEAT, d=48, T=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.randn(1, T, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, d * 2, dropout=0.3, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.pool = AttnPool(d)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(d, 1))

    def forward(self, vec, nodes):
        h = self.inp(vec) + self.pos[:, :vec.shape[1]]
        return self.head(self.pool(self.enc(h))).squeeze(-1)


class MambaBlock(nn.Module):
    """Упрощённый selective SSM (Mamba) на чистом PyTorch: последовательный скан по времени."""

    def __init__(self, d, n_state=8, expand=2, dt_rank=4):
        super().__init__()
        e = d * expand
        self.e, self.n, self.r = e, n_state, dt_rank
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 2 * e)
        self.conv = nn.Conv1d(e, e, 4, groups=e, padding=3)
        self.x_proj = nn.Linear(e, dt_rank + 2 * n_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, e)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n_state + 1).float()).repeat(e, 1))
        self.D = nn.Parameter(torch.ones(e))
        self.out = nn.Linear(e, d)

    def forward(self, x):
        B, T, _ = x.shape
        xz = self.in_proj(self.norm(x))
        u, z = xz.chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :T].transpose(1, 2))
        dt, Bm, Cm = torch.split(self.x_proj(u), [self.r, self.n, self.n], -1)
        delta = F.softplus(self.dt_proj(dt))                       # (B,T,e)
        A = -torch.exp(self.A_log)                                  # (e,n)
        dA = torch.exp(delta[..., None] * A)                        # (B,T,e,n)
        dBu = delta[..., None] * Bm[:, :, None, :] * u[..., None]   # (B,T,e,n)
        h = torch.zeros(B, self.e, self.n, device=x.device)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dBu[:, t]
            ys.append((h * Cm[:, t, None, :]).sum(-1))
        y = torch.stack(ys, 1) + u * self.D
        return x + self.out(y * F.silu(z))


class MambaNet(nn.Module):
    def __init__(self, d_in=N_FEAT, d=32):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d), nn.Dropout(0.3))
        self.blocks = nn.Sequential(MambaBlock(d), MambaBlock(d))
        self.pool = AttnPool(d)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(d, 1))

    def forward(self, vec, nodes):
        return self.head(self.pool(self.blocks(self.inp(vec)))).squeeze(-1)


MODELS = {"lstm": LSTMNet, "tcn": TCNNet, "stgcn": STGCN, "transformer": TransformerNet, "mamba": MambaNet}


# ----------------------------------------------------------------------------- обучение
def train_torch(name, tr, va, te, seed, epochs=40, bs=64, lr=2e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MODELS[name]().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * math.ceil(len(tr[0]) / bs))
    pos = max(float(tr[3].mean()), 1e-3)
    pw = torch.tensor((1 - pos) / pos, device=DEV)
    T = lambda a: torch.as_tensor(a, device=DEV)
    Xtr = [T(a) for a in tr[:3]]
    ytr = T(tr[3]).float()
    best, best_state = -1, None

    def predict(d):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(d[0]), 256):
                kp, kc, asp = (T(a[i:i + 256]) for a in d[:3])
                vec, nodes = featurize(kp, kc, asp)
                out.append(torch.sigmoid(model(vec, nodes)).cpu().numpy())
        return np.concatenate(out)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr), device=DEV)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 4:
                continue
            kp, kc = augment(Xtr[0][b], Xtr[1][b])
            vec, nodes = featurize(kp, kc, Xtr[2][b])
            loss = F.binary_cross_entropy_with_logits(model(vec, nodes), ytr[b], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        if ep >= epochs // 3 and len(set(va[3])) > 1:
            auc = roc_auc_score(va[3], predict(va))
            if auc >= best:
                best, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    p = predict(te)
    jw = None
    if name == "stgcn":  # усреднённое внимание по суставам на тесте
        model.eval()
        acc = []
        with torch.no_grad():
            for i in range(0, len(te[0]), 256):
                kp, kc, asp = (T(a[i:i + 256]) for a in te[:3])
                vec, nodes = featurize(kp, kc, asp)
                model(vec, nodes)
                acc.append(model.last_joint_w.mean(1).cpu().numpy())
        jw = np.concatenate(acc)
    return p, jw


def window_stats(d):
    with torch.no_grad():
        vec, _ = featurize(torch.as_tensor(d[0]), torch.as_tensor(d[1]), torch.as_tensor(d[2]))
    v = vec.numpy()
    return np.concatenate([v.mean(1), v.std(1), v.min(1), v.max(1), v[:, -1] - v[:, 0]], 1)


def train_gbm(tr, te, seed, feat="all"):
    A, B = window_stats(tr), window_stats(te)
    if feat == "aspect":
        sel = slice(N_FEAT - 1, N_FEAT)
        A, B = A[:, [N_FEAT * k + N_FEAT - 1 for k in range(5)]], B[:, [N_FEAT * k + N_FEAT - 1 for k in range(5)]]
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
                                         class_weight="balanced", random_state=seed)
    clf.fit(A, tr[3])
    return clf.predict_proba(B)[:, 1]


# ----------------------------------------------------------------------------- метрики
def video_scores(p, vid_idx, k=3):
    out = {}
    for v in np.unique(vid_idx):
        s = np.sort(p[vid_idx == v])[::-1]
        out[v] = float(s[:k].mean())
    return out


def best_f1(y, s):
    ths = np.unique(s)
    return max((f1_score(y, s >= t), t) for t in ths) if len(ths) else (0.0, 0.5)


def agg_scores(p, vid_idx, mode):
    out = {}
    for v in np.unique(vid_idx):
        a = np.sort(p[vid_idx == v])[::-1]
        out[v] = float(a.mean() if mode == "mean" else a[:mode].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=r"D:\dog-ml\raw")
    ap.add_argument("--out", default=r"D:\dog-ml\cv")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--models", nargs="+", default=["gbm", "lstm", "tcn", "stgcn", "meta", "aspect"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--T", type=int, default=45)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--min-valid", type=float, default=0.9)
    ap.add_argument("--keep", nargs="*", default=None, help="группы точек: spine hips tail front rear")
    ap.add_argument("--drop", nargs="*", default=None, help="убрать группы точек")
    ap.add_argument("--no-conf", action="store_true")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--no-vel", action="store_true")
    ap.add_argument("--no-asp", action="store_true")
    ap.add_argument("--exclude-cam-train", action="store_true")
    ap.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    keep = None
    if args.keep:
        keep = sorted({j for g in args.keep for j in GROUPS[g]})
    if args.drop:
        base = set(range(NKP)) if keep is None else set(keep)
        keep = sorted(base - {j for g in args.drop for j in GROUPS[g]})
    CFG.update(T=args.T, stride=args.stride, min_valid=args.min_valid, keep=keep, conf=not args.no_conf,
               aug=not args.no_aug, vel=not args.no_vel, asp=not args.no_asp,
               exclude_cam_train=args.exclude_cam_train)
    print("CFG", CFG, flush=True)

    kp, kc, asp, midx, vids = build_windows(args.raw)
    y = np.array([vids[i]["cls"] for i in midx])
    print(f"windows={len(y)} (pos {int(y.sum())} / neg {int((1 - y).sum())}), videos with windows="
          f"{len(set(midx.tolist()))} of {len(vids)}", flush=True)
    data = (kp, kc, asp, y)
    vid_cls = np.array([v["cls"] for v in vids])
    vid_grp = np.array([v["group"] for v in vids])
    vid_dom = np.array([v["domain"] for v in vids])
    have = np.array(sorted(set(midx.tolist())))
    results, store = {}, {}
    joint_imp = np.zeros(NKP)
    joint_n = 0

    for model_name in args.models:
        oof_runs = []
        for seed in range(args.seed_offset, args.seed_offset + args.seeds):
            skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
            t0 = time.time()
            vscore = {m: {} for m in (1, 3, 5, "mean")}
            win_p = np.full(len(y), np.nan)
            for fold, (trv, tev) in enumerate(skf.split(have, vid_cls[have], vid_grp[have])):
                trv, tev = have[trv], have[tev]
                if CFG["exclude_cam_train"]:
                    trv = np.array([v for v in trv if vid_dom[v] != "cam"])
                rs = np.random.RandomState(seed * 100 + fold)
                gs = np.unique(vid_grp[trv])
                vgs = set(rs.choice(gs, max(1, int(0.2 * len(gs))), replace=False))
                va_v = np.array([v for v in trv if vid_grp[v] in vgs])
                tr_v = np.array([v for v in trv if vid_grp[v] not in vgs])
                sel = lambda vs: np.flatnonzero(np.isin(midx, vs))
                tr, va, te = [tuple(a[sel(vs)] for a in data) for vs in (tr_v, va_v, tev)]
                if model_name in MODELS:
                    p, jw = train_torch(model_name, tr, va, te, seed * 100 + fold, epochs=args.epochs)
                    if jw is not None:
                        joint_imp += jw.sum(0)
                        joint_n += len(jw)
                elif model_name == "gbm":
                    p = train_gbm(tr, te, seed)
                elif model_name == "aspect":
                    p = train_gbm(tr, te, seed, feat="aspect")
                elif model_name == "meta":
                    mf = lambda vs: np.array([[vids[v]["w"], vids[v]["h"], vids[v]["fps"], vids[v]["dur"]] for v in vs])
                    clf = HistGradientBoostingClassifier(max_iter=100, max_depth=3, class_weight="balanced",
                                                         random_state=seed)
                    clf.fit(mf(np.concatenate([tr_v, va_v])), vid_cls[np.concatenate([tr_v, va_v])])
                    for v, s_ in zip(tev, clf.predict_proba(mf(tev))[:, 1]):
                        for m in vscore:
                            vscore[m][v] = float(s_)
                    continue
                win_p[sel(tev)] = p
                for m in vscore:
                    vscore[m].update(agg_scores(p, midx[sel(tev)], m))
            ids = np.array(sorted(vscore[3]))
            yv = vid_cls[ids]
            sc = np.array([vscore[3][i] for i in ids])
            dom = vid_dom[ids]
            r = dict(seed=seed, auc=roc_auc_score(yv, sc), ap=average_precision_score(yv, sc),
                     f1_05=f1_score(yv, sc >= 0.5), f1_best=best_f1(yv, sc)[0],
                     n_videos=int(len(ids)), sec=round(time.time() - t0, 1))
            for m in (1, 5, "mean"):
                r[f"auc_k{m}"] = roc_auc_score(yv, [vscore[m][i] for i in ids])
            other = dom == "other"
            if len(set(yv[other])) > 1:
                r["auc_other"] = roc_auc_score(yv[other], sc[other])
            cam = dom == "cam"
            r["cam_fpr_05"] = float((sc[cam] >= 0.5).mean()) if cam.any() else None
            if np.isfinite(win_p).any():
                mm = np.isfinite(win_p)
                r["win_auc"] = roc_auc_score(y[mm], win_p[mm])
            oof_runs.append(r)
            store[f"{model_name}_s{seed}"] = sc
            store[f"{model_name}_ids"] = ids
            print(f"  [{model_name}] seed {seed}: " + " ".join(f"{k}={v:.3f}" for k, v in r.items()
                  if isinstance(v, float)), flush=True)
        results[model_name] = oof_runs
        keys = ("auc", "ap", "f1_best", "auc_other", "cam_fpr_05", "win_auc", "auc_k1", "auc_k5", "auc_kmean")
        agg = {k: (float(np.mean([r[k] for r in oof_runs if r.get(k) is not None])),
                   float(np.std([r[k] for r in oof_runs if r.get(k) is not None])))
               for k in keys if any(r.get(k) is not None for r in oof_runs)}
        print(f"== {model_name}: " + " | ".join(f"{k} {m:.3f}±{s:.3f}" for k, (m, s) in agg.items()), flush=True)
        json.dump(dict(cfg={k: v for k, v in CFG.items()}, args=vars(args), results=results,
                       n_windows=int(len(y)), n_videos=int(len(have))),
                  open(os.path.join(args.out, f"results_{args.tag}.json"), "w"), indent=1)
        np.savez(os.path.join(args.out, f"scores_{args.tag}.npz"), vid_names=np.array([v["name"] for v in vids]),
                 vid_cls=vid_cls, vid_dom=vid_dom, **store)
    if joint_n:
        np.save(os.path.join(args.out, f"joint_{args.tag}.npy"), joint_imp / joint_n)
        print("joint attention:", np.round(joint_imp / joint_n, 3).tolist())


if __name__ == "__main__":
    sys.exit(main())
