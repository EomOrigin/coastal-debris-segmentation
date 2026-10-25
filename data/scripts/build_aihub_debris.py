#!/usr/bin/env python
"""build_aihub_debris.py — AI-Hub 71363 드론(0.1 m) 해안쓰레기 → 0.3 m 실측 자산

주최 공지(참고 데이터 안내)에 따라 학습에 사용. 원본은 재배포하지 않는다(제출물·코드에 포함 금지).
해안쓰레기 폴리곤은 라벨 래스터(코드 70/80, 80 은 대부분 산림)가 아니라 JSON 의 ANN_NM('해안쓰레기_*')으로 래스터화한다.
AI-Hub 분할이 위치 단위로 겹치지 않으므로(학습 40곳 / 검증 25곳) 그대로 쓴다.

    python build_aihub_debris.py --split val   --src <VS_01.Drone.zip>
    python build_aihub_debris.py --split train --src <TS_01.Drone.zip>

산출 (data/aihub_71363/derived/):
  tiles/<split>/<name>.png, <name>_debris.png        0.1 m 원본 RGB + 쓰레기 마스크
  sprites/<split>/<name>_<k>.png (RGBA) + sprites.csv 쓰레기 조각(연결 성분) — 합성기 조각 이식(v6)
  real03/<split>/<name>_img.png, <name>_gt.png       0.3 m 타일(면적 평균) + GT(피복률 ≥0.5 = '한 화소 미만 조각 미표시')
  eval_<split>_{raw,sat}/images + patches.csv, eval_<split>_{raw,sat}_gt.json   e2e_check·run_local 형식 실측 평가셋
  stats_<split>.json                                  0.3 m 쓰레기 외관 통계(합성기 보정용)
  kr_neg/<split>/<name>_<r><c>.png                    한국 해안 음성: 쓰레기 타일의 네 모서리 256 창 중 쓰레기에서 10 px 이상 떨어진 창
"""
import argparse
import csv
import io
import json
import zipfile
import zlib
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.features import rasterize

ROOT = Path(__file__).resolve().parents[1] / "aihub_71363"
DL = ROOT / "download" / "310.AI기반_국립공원_변화탐지_모니터링_플랫폼_구축" / "01-1.정식개방데이터"
JSON_ZIP = {"train": DL / "Training/02.라벨링데이터/TL_02.JSON_01.Drone.zip",
            "val": DL / "Validation/02.라벨링데이터/VL_02.JSON_01._Drone.zip"}
OUT = ROOT / "derived"


def debris_tiles(split):
    """→ {tile name: [geometry, ...]} (해안쓰레기 폴리곤만)"""
    zf = zipfile.ZipFile(JSON_ZIP[split]); out = {}
    for n in zf.namelist():
        if not n.endswith(".json"):
            continue
        d = json.loads(zf.read(n))
        g = [f["geometry"] for f in d.get("features", []) if "해안쓰레기" in str(f["properties"].get("ANN_NM"))]
        if g:
            out.setdefault(d["name"], []).extend(g)
    return out


def rle(m):
    f = m.reshape(-1).astype(np.uint8)
    d = np.diff(np.concatenate([[0], f, [0]]))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return " ".join(f"{a} {b - a}" for a, b in zip(s, e))


def sat_look(img, rng):
    """평가 영상 쪽 외관(팬샤픈 색 번짐 σ1.5 + 휘도 블러 + 타일 단위 2~98% 스트레치 + 잡음)."""
    f = img.astype(np.float32)
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y[..., 0] = cv2.GaussianBlur(y[..., 0], (0, 0), 0.5)
    y[..., 1:] = cv2.GaussianBlur(y[..., 1:], (0, 0), 1.5)
    f = cv2.cvtColor(np.clip(y + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB).astype(np.float32)
    lo = np.percentile(f, 2, axis=(0, 1)); hi = float(np.percentile(f, 98, axis=(0, 1)).max())
    f = (f - lo) / max(1.0, hi - float(lo.min())) * 255 + rng.normal(0, 3.0, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val"]); ap.add_argument("--src", required=True)
    a = ap.parse_args()
    tiles = debris_tiles(a.split)
    zf = zipfile.ZipFile(a.src); members = {Path(n).stem: n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))}
    print(f"{a.split}: 쓰레기 타일 {len(tiles)} / 원천 zip 타일 {len(members)} / 매칭 {len(set(tiles) & set(members))}", flush=True)
    for sub in [f"tiles/{a.split}", f"sprites/{a.split}", f"real03/{a.split}", f"kr_neg/{a.split}"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    ev = {v: OUT / f"eval_{a.split}_{v}" for v in ("raw", "sat")}
    for p in ev.values():
        (p / "images").mkdir(parents=True, exist_ok=True)
    sprite_rows, gts = [], {"raw": {}, "sat": {}}
    stat = dict(n_px=0, lum_diff=[], lum_std_in=[], chroma_in=[], chroma_ring=[], colored_frac=[], area03=[])
    for name, geoms in sorted(tiles.items()):
        if name not in members:
            continue
        with rasterio.MemoryFile(zf.read(members[name])) as mf, mf.open() as ds:
            arr = ds.read(); tr = ds.transform; h, w = ds.height, ds.width
        rgb = np.ascontiguousarray(np.moveaxis(arr[:3], 0, -1)).astype(np.uint8)
        mask = rasterize([(g, 1) for g in geoms], out_shape=(h, w), transform=tr, fill=0, dtype="uint8").astype(bool)
        if not mask.any():
            continue
        cv2.imwrite(str(OUT / f"tiles/{a.split}/{name}.png"), rgb[..., ::-1])
        cv2.imwrite(str(OUT / f"tiles/{a.split}/{name}_debris.png"), mask.astype(np.uint8) * 255)
        # ---- 조각 스프라이트(0.1 m): 3x3 closing 후 연결 성분, 0.3 m² 이상
        mc = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(mc, connectivity=8)
        for k in range(1, n):
            x, y, bw, bh, ar = st[k]
            if ar < 30:
                continue
            x0, y0, x1, y1 = max(0, x - 3), max(0, y - 3), min(w, x + bw + 3), min(h, y + bh + 3)
            al = ((lab[y0:y1, x0:x1] == k) & mask[y0:y1, x0:x1]).astype(np.float32)
            al = cv2.GaussianBlur(al, (0, 0), 0.7) if min(al.shape) > 4 else al
            rgba = np.dstack([rgb[y0:y1, x0:x1][..., ::-1], (al * 255).astype(np.uint8)])
            fn = f"{name}_{k}.png"
            cv2.imwrite(str(OUT / f"sprites/{a.split}/{fn}"), rgba)
            sprite_rows.append(dict(file=fn, tile=name, area_m2=round(ar * 0.01, 2), w=int(x1 - x0), h=int(y1 - y0)))
        # ---- 0.3 m 타일 + GT(피복률 ≥ 0.5)
        W3, H3 = w // 3, h // 3
        img3 = cv2.resize(rgb[:H3 * 3, :W3 * 3], (W3, H3), interpolation=cv2.INTER_AREA)
        cov = cv2.resize(mask[:H3 * 3, :W3 * 3].astype(np.float32), (W3, H3), interpolation=cv2.INTER_AREA)
        gt3 = cov >= 0.5
        cv2.imwrite(str(OUT / f"real03/{a.split}/{name}_img.png"), img3[..., ::-1])
        cv2.imwrite(str(OUT / f"real03/{a.split}/{name}_gt.png"), gt3.astype(np.uint8) * 255)
        if not gt3.any():
            continue
        far = cv2.dilate(gt3.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)   # 쓰레기 ±10 px
        if H3 >= 256 and W3 >= 256:
            for r, oy in enumerate((0, H3 - 256)):
                for c, ox in enumerate((0, W3 - 256)):
                    if not far[oy:oy + 256, ox:ox + 256].any():
                        cv2.imwrite(str(OUT / f"kr_neg/{a.split}/{name}_{r}{c}.png"), img3[oy:oy + 256, ox:ox + 256][..., ::-1])
        # ---- 평가 패치: 쓰레기 무게중심 중심 256 창
        ys, xs = np.nonzero(gt3)
        cy = int(np.clip(np.median(ys), 128, H3 - 128)); cx = int(np.clip(np.median(xs), 128, W3 - 128))
        if H3 < 256 or W3 < 256:
            continue
        win = (slice(cy - 128, cy + 128), slice(cx - 128, cx + 128))
        pim, pgt = img3[win], gt3[win]
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        for v, im in (("raw", pim), ("sat", sat_look(pim, rng))):
            cv2.imwrite(str(ev[v] / "images" / f"{name}.png"), im[..., ::-1])
            gts[v][name] = rle(pgt)
        # ---- 외관 통계(0.3 m, raw)
        f = pim.astype(np.float32); lum = f @ np.array([0.299, 0.587, 0.114], np.float32)
        cb, cr = (f[..., 2] - lum) * 0.564, (f[..., 0] - lum) * 0.713; ch = np.sqrt(cb ** 2 + cr ** 2)
        ring = cv2.dilate(pgt.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~pgt
        stat["n_px"] += int(pgt.sum()); stat["area03"].append(int(pgt.sum()))
        stat["lum_diff"].append(float(lum[pgt].mean() - lum[ring].mean()))
        stat["lum_std_in"].append(float(lum[pgt].std())); stat["chroma_in"].append(float(ch[pgt].mean()))
        stat["chroma_ring"].append(float(ch[ring].mean())); stat["colored_frac"].append(float((ch[pgt] > 15).mean()))
    for v, p in ev.items():
        with open(p / "patches.csv", "w", newline="") as fcsv:
            wr = csv.writer(fcsv); wr.writerow(["id"]); wr.writerows([[k] for k in gts[v]])
        (p.parent / f"{p.name}_gt.json").write_text(json.dumps(gts[v]))
    with open(OUT / f"sprites/{a.split}/sprites.csv", "w", newline="") as fcsv:
        wr = csv.DictWriter(fcsv, fieldnames=["file", "tile", "area_m2", "w", "h"]); wr.writeheader(); wr.writerows(sprite_rows)
    summ = {k: (float(np.median(v)) if isinstance(v, list) and v else v) for k, v in stat.items()}
    summ["n_patches"] = len(gts["raw"]); summ["n_sprites"] = len(sprite_rows)
    (OUT / f"stats_{a.split}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
