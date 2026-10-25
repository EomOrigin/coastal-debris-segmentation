#!/usr/bin/env python
"""extract_kr_bg.py — AI-Hub 71363 학습 분할의 태안해안국립공원(DR20) 드론 타일 → 0.3 m 한국 해안 배경
쓰레기가 라벨된 타일은 제외(라벨 안 된 쓰레기가 섞였을 수 있음). 검증 분할은 실측 평가용이라 쓰지 않는다.
    python extract_kr_bg.py <TS_01.Drone.zip> <out_dir>
"""
import sys, json, zipfile, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, rasterio
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_aihub_debris import debris_tiles

src, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
skip = set(debris_tiles("train"))
zf = zipfile.ZipFile(src)
names = [n for n in zf.namelist() if Path(n).stem.startswith("DR20") and n.endswith(".tif") and Path(n).stem not in skip]
print("DR20 학습 타일", len(names), "(쓰레기 타일 제외)", flush=True)
datas = ((n, zf.read(n)) for n in names)
def conv(item):
    n, b = item
    with rasterio.MemoryFile(b) as mf, mf.open() as ds:
        a = ds.read()
    rgb = np.ascontiguousarray(np.moveaxis(a[:3], 0, -1)).astype(np.uint8)
    if a.shape[0] >= 4 and (a[3] == 0).mean() > 0.01:          # 촬영 밖(알파 0) 1% 초과면 버림
        return 0
    if (rgb.max(-1) <= 2).mean() > 0.01:
        return 0
    h, w = rgb.shape[:2]; H3, W3 = h // 3, w // 3
    img = cv2.resize(rgb[:H3 * 3, :W3 * 3], (W3, H3), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / f"{Path(n).stem}.jpg"), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    return 1
with ThreadPoolExecutor(8) as ex:
    ok = sum(ex.map(conv, datas))
print("저장", ok, "→", out, flush=True)
