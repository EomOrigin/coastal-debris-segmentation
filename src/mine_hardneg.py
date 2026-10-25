#!/usr/bin/env python
"""mine_hardneg.py — 실배경 헛경보 채굴 (v4 D4)

학습 장면(train.py 와 같은 분할로 검증 장면은 제외)을 4x4 격자의 256 크롭으로 자르고, 원본(raw)과 장면 단위 위성 룩(sat,
synth.sat_scene)을 입힌 판을 모두 모델에 넣는다. 제출 노트북과 같은 결정 규칙으로 '양성'이 나온 크롭 = 모델의 헛경보
(실배경에는 쓰레기 집적대가 사실상 없다) → v4 학습의 음성 풀(synth.MinedBank, cfg.mined_p).

    python mine_hardneg.py --run runs/v2_r34 --bg ../data/backgrounds/splits/v2_train.txt \\
                           --out ../data/backgrounds/mined_negatives_v2
v5 2차 채굴: --run runs/v4_r34 --variants raw,sat,cstretch --out ../data/backgrounds/mined_negatives_v4
산출: <out>/crops/*.png, <out>/mined.csv(장면·위치·존재 확률·면적), <out>/sheet.jpg(눈으로 거르는 용도)
"""
import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from postproc import decide
from synth import background_paths, sat_scene
from train import MEAN, STD, Net, split_scenes


@torch.no_grad()
def predict_flip(net, imgs):
    x = torch.from_numpy(((imgs.astype(np.float32) / 255.0 - MEAN) / STD).transpose(0, 3, 1, 2)).cuda()
    ap, ac = 0.0, 0.0
    for flip in (False, True):
        xt = x.flip(-1) if flip else x
        with torch.autocast("cuda", dtype=torch.float16):
            logit, cls = net(xt)
        p = torch.sigmoid(logit.float())
        p = F.avg_pool2d(p, net.up) if net.up > 1 else p
        ap = ap + (p.flip(-1) if flip else p)
        ac = ac + torch.sigmoid(cls.float()).flatten()
    return (ap / 2)[:, 0].cpu().numpy(), (ac / 2).cpu().numpy()


VARIANTS = ("raw", "sat")


def _crop_stretch(c):
    f = c.astype(np.float32)
    lo = np.percentile(f, 2, axis=(0, 1)); hi = float(np.percentile(f, 98, axis=(0, 1)).max())
    return np.clip((f - lo) / max(1.0, hi - float(lo.min())) * 255, 0, 255).astype(np.uint8)


def scene_crops(job):
    """(장면 번호, 경로) → [(변형, 행, 열, 크롭)]. 변형: raw 원본 / sat 장면 단위 위성 룩 / cstretch 크롭 단위 2~98% 스트레치"""
    k, path = job
    s = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if s is None:
        return k, path, []
    s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
    views = [("raw", s)] if "raw" in VARIANTS or "cstretch" in VARIANTS else []
    if "sat" in VARIANTS:
        views.append(("sat", sat_scene(np.random.default_rng([2026, k]), s)))
    out = []
    for name, v in views:
        for r in range(v.shape[0] // 256):
            for c in range(v.shape[1] // 256):
                crop = np.ascontiguousarray(v[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256])
                if name == "raw" and "raw" in VARIANTS:
                    out.append(("raw", r, c, crop))
                if name == "raw" and "cstretch" in VARIANTS:
                    out.append(("cstretch", r, c, _crop_stretch(crop)))
                if name == "sat":
                    out.append(("sat", r, c, crop))
    return k, path, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--bg", required=True, nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="train.py 의 --seed 와 같게(같은 검증 분할을 빼기 위해)")
    ap.add_argument("--batch", type=int, default=128); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variants", default="raw,sat", help="raw,sat,cstretch 중 선택(쉼표)")
    a = ap.parse_args()
    global VARIANTS
    VARIANTS = tuple(a.variants.split(","))
    run, out = Path(a.run), Path(a.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run / "config.json").read_text())
    net = Net(cfg["arch"], cfg["encoder"], cfg["upscale"], encoder_weights=None).cuda().eval()
    sd = torch.load(run / cfg["weights"][0], map_location="cpu", weights_only=True)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    tr_paths, val_paths = split_scenes(background_paths(a.bg), a.seed)
    if a.limit:
        tr_paths = tr_paths[:a.limit]
    print(f"mining with {run.name} (mask_thr {cfg['mask_thr']}, presence_thr {cfg['presence_thr']}) on {len(tr_paths)} train scenes "
          f"(val {len(val_paths)} excluded)", flush=True)
    rows, buf, t0, n_seen = [], [], time.time(), 0

    def flush():
        nonlocal buf, n_seen
        if not buf:
            return
        imgs = np.stack([b[-1] for b in buf])
        probs, press = predict_flip(net, imgs)
        for (k, path, name, r, c, img), p, q in zip(buf, probs, press):
            m = decide(p, q, cfg)
            if m.sum() >= 50:
                fn = f"{Path(path).stem}_{name}_{r}{c}.png"
                cv2.imwrite(str(out / "crops" / fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                rows.append(dict(file=fn, scene=Path(path).name, source=Path(path).parent.name, variant=name, row=r, col=c,
                                 presence=round(float(q), 4), area=int(m.sum())))
        n_seen += len(buf); buf = []

    with ThreadPoolExecutor(16) as ex:
        for k, path, crops in ex.map(scene_crops, enumerate(tr_paths)):
            for name, r, c, img in crops:
                buf.append((k, path, name, r, c, img))
                if len(buf) >= a.batch:
                    flush()
            if (k + 1) % 250 == 0:
                print(f"  {k + 1}/{len(tr_paths)} scenes, crops {n_seen}, mined {len(rows)}, {time.time() - t0:.0f}s", flush=True)
    flush()
    with open(out / "mined.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "scene", "source", "variant", "row", "col", "presence", "area"])
        w.writeheader(); w.writerows(rows)
    by = {}
    for r in rows:
        by[(r["source"], r["variant"])] = by.get((r["source"], r["variant"]), 0) + 1
    print(f"DONE crops {n_seen}, mined {len(rows)} ({len(rows) / max(1, n_seen):.2%}) | " +
          " ".join(f"{s}/{v}:{n}" for (s, v), n in sorted(by.items())), flush=True)
    # 눈으로 거르는 시트: 무작위 96장 + 예측 경계
    rng = np.random.default_rng(0)
    pick = [rows[i] for i in rng.permutation(len(rows))[:96]]
    tiles = []
    for r in pick:
        im = cv2.imread(str(out / "crops" / r["file"]))
        cv2.putText(im, f"{r['source'][:4]} {r['variant']} {r['presence']:.2f}", (4, 16), 0, 0.5, (0, 255, 255), 1)
        tiles.append(cv2.resize(im, (192, 192)))
    while len(tiles) % 12:
        tiles.append(np.zeros((192, 192, 3), np.uint8))
    if tiles:
        cv2.imwrite(str(out / "sheet.jpg"), np.concatenate([np.concatenate(tiles[i:i + 12], 1) for i in range(0, len(tiles), 12)], 0),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
