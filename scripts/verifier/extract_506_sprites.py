"""extract_506_sprites.py — AI-Hub 「해안 오염물질」(506) 샘플(부유쓰레기, 스마트폰 근접 촬영 3024², 박스 라벨)에서
물체 RGBA 스프라이트를 0.1 m/px 축척으로 만든다(synth.render_sprites 가 쓰는 v6 스프라이트 규격과 같음).
  알파: 박스 테두리(물) 중앙값 색과의 Lab 거리 → Otsu → 열기/닫기 → 1 px 페더
  축척: 라벨별 물리 크기(가정, 긴 변 m) / 0.1 m → 긴 변 화소 수
    python extract_506_sprites.py → data/aihub_506_sample/derived/sprites/*.png + sprites.csv"""
import csv, glob, json
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/aihub_506_sample"
OUT = SRC / "derived/sprites"; OUT.mkdir(parents=True, exist_ok=True)
SIZE_M = {"Plastic_Buoy": (0.3, 0.5), "Plastic_Buoy_China": (0.4, 0.6), "Styrofoam_Buoy": (0.5, 0.8), "Styrofoam_Box": (0.4, 0.6),
          "Styrofoam_Piece": (0.1, 0.4), "PET_Bottle": (0.2, 0.3), "Glass": (0.25, 0.35), "Metal": (0.15, 0.4),
          "Plastic_ETC": (0.2, 0.6), "Net": (0.6, 2.0), "Rope": (0.5, 2.0)}
rng = np.random.default_rng(506)
rows = []
for jp in sorted(glob.glob(str(SRC / "라벨링데이터/**/*.json"), recursive=True)):
    d = json.load(open(jp, encoding="utf-8"))
    ip = next(iter(glob.glob(str(SRC / "원천데이터/**/" / d["imagePath"]), recursive=True)), None)
    if ip is None:
        continue
    img = cv2.imread(ip, cv2.IMREAD_COLOR)
    if img is None:
        continue
    H, W = img.shape[:2]
    for k, s in enumerate(d["shapes"]):
        lab = s["label"]
        if lab not in SIZE_M:
            continue
        (x0, y0), (x1, y1) = s["points"]
        x0, x1 = sorted((int(x0), int(x1))); y0, y1 = sorted((int(y0), int(y1)))
        if x1 - x0 < 12 or y1 - y0 < 12 or x0 <= 1 or y0 <= 1 or x1 >= W - 2 or y1 >= H - 2:
            continue                                                   # 너무 작거나 잘린 물체 제외
        pad = int(0.15 * max(x1 - x0, y1 - y0))
        X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
        crop = img[Y0:Y1, X0:X1]
        lab_img = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        ring = np.ones(crop.shape[:2], bool); ring[pad:-pad or None, pad:-pad or None] = False
        if ring.sum() < 20:
            continue
        bgc = np.median(lab_img[ring], 0)
        dist = np.linalg.norm(lab_img - bgc, axis=2)
        dn = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, m = cv2.threshold(dn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        inner = np.zeros_like(m); inner[pad:-pad or None, pad:-pad or None] = 255
        m = cv2.bitwise_and(m, inner)
        if m.mean() / 255 < 0.08:
            continue
        ys, xs = np.nonzero(m)
        crop, m = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1], m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        a = cv2.GaussianBlur(m, (0, 0), 1.0)
        rgba = np.dstack([cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), a])
        size_m = rng.uniform(*SIZE_M[lab])
        L = max(2, int(round(size_m / 0.1)))
        sc = L / max(rgba.shape[:2])
        rgba = cv2.resize(rgba, (max(1, int(round(rgba.shape[1] * sc))), max(1, int(round(rgba.shape[0] * sc)))), interpolation=cv2.INTER_AREA)
        fn = f"{Path(jp).stem}_{k:02d}_{lab}.png"
        cv2.imwrite(str(OUT / fn), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        rows.append(dict(file=fn, label=lab, origin=s.get("origin"), size_m=round(size_m, 2), px=f"{rgba.shape[1]}x{rgba.shape[0]}"))
with open(OUT.parent / "sprites.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
import collections
print("sprites", len(rows), collections.Counter(r["label"] for r in rows).most_common())
