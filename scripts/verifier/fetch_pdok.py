"""fetch_pdok.py — 네덜란드 PDOK 항공사진 2024(25 cm, CC BY 4.0 "Beeldmateriaal.nl / PDOK")에서 와덴해·젤란트 갯벌 장면 수집.
EPSG:3857 z19 타일(위도 53°에서 0.178 m/px) 7x7 모자이크 → 0.3 m 로 면적 축소 → 1024² JPEG. 빈 타일(촬영 범위 밖)은 버린다.
    python fetch_pdok.py [장면 수=220]   → data/backgrounds/pdok_tidal/*.jpg + ledger.csv"""
import csv, io, math, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/backgrounds/pdok_tidal"; OUT.mkdir(parents=True, exist_ok=True)
TPL = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/2024_ortho25/EPSG:3857/{z}/{x}/{y}.jpeg"
BOXES = {   # (lon0, lat0, lon1, lat1): 갯벌·염습지·사주가 있는 해안 띠
    "balgzand": (4.78, 52.88, 4.92, 52.95), "texel_wad": (4.85, 53.00, 4.95, 53.10),
    "holwerd_paesens": (5.75, 53.38, 6.05, 53.43), "ameland_south": (5.65, 53.42, 5.90, 53.45),
    "schiermonnikoog": (6.10, 53.45, 6.35, 53.49), "groningen_wad": (6.50, 53.42, 6.85, 53.47),
    "dollard": (7.05, 53.25, 7.20, 53.32), "saeftinghe": (4.10, 51.33, 4.22, 51.40),
    "oosterschelde": (3.80, 51.55, 4.10, 51.65), "terschelling_wad": (5.20, 53.36, 5.45, 53.40)}
Z, N = 19, 7


def tile(x, y):
    u = TPL.format(z=Z, x=x, y=y)
    for _ in range(3):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "coastal-research/1.0"})
            return Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=30).read())).convert("RGB")
        except Exception:
            pass
    return None


def scene(args):
    name, lon, lat, k = args
    n2 = 2 ** Z
    xt = (lon + 180) / 360 * n2
    yt = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n2
    x0, y0 = int(xt) - N // 2, int(yt) - N // 2
    mos = Image.new("RGB", (256 * N, 256 * N))
    miss = 0
    for j in range(N):
        for i in range(N):
            t = tile(x0 + i, y0 + j)
            if t is None:
                miss += 1
                continue
            mos.paste(t, (256 * i, 256 * j))
    if miss > 2:
        return None
    gsd = 156543.03392804097 / n2 * math.cos(math.radians(lat))
    side = int(round(256 * N * gsd / 0.3))
    im = mos.resize((side, side), Image.BOX)
    c = (side - 1024) // 2
    im = im.crop((c, c, c + 1024, c + 1024))
    a = np.asarray(im).astype(np.float32)
    if a.std() < 6 or (a.mean(2) > 245).mean() > 0.2:        # 빈 영역·흰 무자료
        return None
    fn = OUT / f"pdok_{name}_{k:03d}.jpg"
    im.save(fn, quality=95)
    return dict(file=fn.name, region=name, lon=round(lon, 6), lat=round(lat, 6), gsd_src=round(gsd, 4),
                source="PDOK Luchtfoto 2024 ortho25", license="CC BY 4.0 (Beeldmateriaal.nl / PDOK)")


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 220
    rng = np.random.default_rng(16)
    per = total // len(BOXES)
    jobs = [(name, rng.uniform(a, c), rng.uniform(b, d), k) for name, (a, b, c, d) in BOXES.items() for k in range(per)]
    rows = []
    with ThreadPoolExecutor(12) as ex:
        for r in ex.map(scene, jobs):
            if r:
                rows.append(r)
    with open(OUT / "ledger.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)}/{len(jobs)} scenes → {OUT}")


if __name__ == "__main__":
    main()
