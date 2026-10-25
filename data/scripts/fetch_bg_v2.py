#!/usr/bin/env python
"""fetch_bg_v2.py — v2 배경 장면 수집 (1024x1024 @ 0.3 m, 해안선 중심, 북쪽 위)

    python fetch_bg_v2.py points --shp <lines.shp>   # OSM 해안선에서 장면 중심 후보 좌표 생성 → _meta/points_<source>.csv
                                                     #   (lines.shp = osmdata.openstreetmap.de/download/coastlines-split-4326.zip)
    python fetch_bg_v2.py fetch --source gsi     # 소스별 수집 → backgrounds/<source>/*.jpg + _meta/ledger_<source>.csv

소스(모두 의무가 출처표시뿐인 개방 라이선스, data/README.md 참조)
    gsi   일본 국토지리원 ort 타일 z18 (~0.49 m → 0.3 m 업샘플)              PDL1.0
    naip  USGS NAIP ImageServer, 0.3 m 원본만(where resolution_value<=0.3)   Public Domain  (+NIR 밴드 별도 저장)
    linz  LINZ nz-imagery 공개 S3 COG (0.2~0.3 m)                            CC BY 4.0
    ign   IGN BD ORTHO HR (0.2 m → 0.3 m 면적 평균 축소)                     Licence Ouverte 2.0
    eri   NOAA NGS ERI 평상시(Pre_Event) GeoTIFF (~0.3 m)                   NOAA NODD open data (CC0 취지)

좌표 추출에만 OSM(Overpass)을 쓰고 OSM 데이터 자체는 저장·학습하지 않는다.
의존: requests numpy opencv-python-headless rasterio<1.4 (학습 환경과 무관한 수집 전용)
"""
import argparse
import csv
import io
import json
import math
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1] / "backgrounds"
META = ROOT / "_meta"
UA = {"User-Agent": "knps-coastal-bg-sampler/0.2 (research; contact via aifactory)"}
OUT_PX, GSD = 1024, 0.3
HALF_M = OUT_PX * GSD / 2                                        # 153.6 m
R_EARTH = 6378137.0

# (지역명, 남, 서, 북, 동, 목표 장면 수). 지역 = 하드 네거티브 유형을 고려해 고른 해안 구간.
REGIONS = {
    "gsi": [("tsushima", 34.08, 129.15, 34.72, 129.50, 150), ("iki", 33.72, 129.63, 33.87, 129.80, 50),
            ("goto", 32.55, 128.55, 33.30, 129.20, 100), ("nagasaki_sasebo", 32.60, 129.55, 33.40, 130.00, 100),
            ("ariake_tidalflat", 32.75, 130.10, 33.25, 130.60, 120), ("genkai_fukuoka", 33.45, 129.80, 33.95, 130.60, 100),
            ("hiroshima_bay_oyster", 34.10, 132.20, 34.40, 132.60, 120), ("okayama_kagawa", 34.25, 133.60, 34.60, 134.30, 60),
            ("yamaguchi_north", 34.20, 130.85, 34.70, 131.60, 60), ("shimane", 35.25, 132.60, 35.60, 133.30, 50),
            ("saiki_oita", 32.80, 131.80, 33.20, 132.10, 40)],
    "naip": [("oregon", 42.00, -124.60, 46.25, -123.80, 200), ("ca_north", 38.00, -124.40, 41.80, -122.90, 140),
             ("ca_south", 33.40, -120.70, 35.70, -117.50, 100), ("massachusetts", 41.20, -71.00, 42.90, -69.90, 170),
             ("ri_ct", 41.20, -73.60, 41.70, -71.10, 90), ("new_jersey", 38.90, -74.50, 40.50, -73.90, 70),
             ("delmarva", 38.00, -76.50, 39.30, -75.00, 80), ("maine", 43.00, -70.80, 44.90, -66.90, 150),
             ("florida", 25.00, -82.90, 30.50, -80.00, 100)],
    "linz": [("northland_2023-2024_0.3m", 250), ("waikato_2023-2024_0.3m", 100), ("bay-of-plenty_2023-2024_0.2m", 80),
             ("marlborough_2023_0.25m", 150), ("tasman_2024-2025_0.25m", 60), ("canterbury_2023_0.3m", 60),
             ("otago_2023-2024_0.2m", 50)],
    "ign": [("bretagne_nord", 48.40, -4.80, 48.90, -2.50, 200), ("finistere_morbihan", 47.30, -4.80, 48.10, -2.40, 150),
            ("cancale_msm_tidalflat", 48.55, -2.00, 48.75, -1.35, 80), ("cotentin_normandie", 48.80, -1.90, 49.75, -0.20, 100),
            ("vendee_charente_oyster", 45.60, -2.40, 47.10, -1.00, 130), ("arcachon_oyster", 44.50, -1.35, 44.80, -1.00, 60),
            ("thau_camargue", 43.20, 3.30, 43.60, 4.90, 80)],
    "eri": [("texas", 26.00, -97.80, 29.80, -93.50, 60), ("louisiana_ms_al", 28.80, -93.50, 30.50, -87.50, 70),
            ("fl_panhandle", 29.50, -87.50, 30.50, -84.00, 50), ("fl_west", 26.00, -83.00, 29.50, -81.70, 50),
            ("carolinas", 32.00, -81.50, 36.60, -75.40, 70), ("georgia", 30.70, -81.60, 32.10, -80.80, 30),
            ("washington", 46.20, -124.80, 48.50, -122.30, 20)],
}
LINZ_REGION = {"northland": "northland", "waikato": "waikato", "bay-of-plenty": "bay-of-plenty", "marlborough": "marlborough",
               "tasman": "tasman", "canterbury": "canterbury", "otago": "otago"}
LICENSE = {"gsi": ("PDL1.0 (GSI 地理院タイル ort)", "https://www.gsi.go.jp/kikakuchousei/kikakuchousei40182.html"),
           "naip": ("Public Domain (USGS/USDA NAIP)", "https://www.usgs.gov/faqs/what-are-terms-uselicensing-map-services-and-data-national-map"),
           "linz": ("CC-BY-4.0 (LINZ / 각 licensor)", "https://nz-imagery.s3.ap-southeast-2.amazonaws.com/LICENSE"),
           "ign": ("Licence Ouverte / Etalab 2.0 (IGN BD ORTHO)", "https://cartes.gouv.fr/cgu"),
           "eri": ("NOAA NODD open data, 'can be used as desired' (CC0 취지)", "https://registry.opendata.aws/noaa-eri/")}
LEDGER_COLS = ["file", "source", "region", "layer", "lon", "lat", "native_gsd_m", "license", "license_url", "request", "fetched_at"]


# --------------------------------------------------------------------------- 좌표
def merc(lon, lat):
    return R_EARTH * math.radians(lon), R_EARTH * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def bbox3857(lon, lat):
    x, y = merc(lon, lat)
    h = HALF_M / math.cos(math.radians(lat))
    return x - h, y - h, x + h, y + h


def overpass_coast(s, w, n, e):
    q = f'[out:json][timeout:240];way["natural"="coastline"]({s},{w},{n},{e});out geom;'
    for url in ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"):
        for attempt in range(3):
            try:
                r = requests.post(url, data={"data": q}, headers=UA, timeout=300)
                if r.status_code == 200:
                    return [[(p["lon"], p["lat"]) for p in el["geometry"]] for el in r.json()["elements"] if el.get("geometry")]
                print("  overpass", url, r.status_code, flush=True)
            except Exception as ex:                                  # noqa: BLE001
                print("  overpass err", url, repr(ex)[:80], flush=True)
            time.sleep(15 * (attempt + 1))
    raise RuntimeError("overpass failed")


def sample_line_points(lines, s, w, n, e, step_m=300.0, cell_m=350.0):
    """해안선 폴리라인을 step_m 간격으로 찍고, cell_m 격자로 중복 제거(장면이 겹치지 않게)."""
    lat0 = (s + n) / 2
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0
    seen, pts = set(), []
    for line in lines:
        acc = step_m / 2
        for (x0, y0), (x1, y1) in zip(line[:-1], line[1:]):
            dx, dy = (x1 - x0) * kx, (y1 - y0) * ky
            seg = math.hypot(dx, dy)
            while acc <= seg and seg > 0:
                t = acc / seg
                lon, lat = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                if s <= lat <= n and w <= lon <= e:
                    key = (round(lon * kx / cell_m), round(lat * ky / cell_m))
                    if key not in seen:
                        seen.add(key); pts.append((lon, lat))
                acc += step_m
            acc -= seg
    return pts


def linz_collection(name):
    region = LINZ_REGION[name.split("_")[0]]
    base = f"https://nz-imagery.s3.ap-southeast-2.amazonaws.com/{region}/{name}/rgb/2193/"
    return base, requests.get(base + "collection.json", headers=UA, timeout=60).json()


def shp_coast(shp, boxes):
    """OSM 해안선 shapefile(coastlines-split-4326, osmdata.openstreetmap.de)에서 각 bbox 에 걸치는 선을 한 번에 뽑는다.
    Overpass 는 지역당 수 분 + 504 가 잦아 로컬 파일로 대체했다."""
    import shapefile
    out = {k: [] for k in boxes}
    sf = shapefile.Reader(shp)
    for i in range(len(sf)):
        sh = sf.shape(i)
        x0, y0, x1, y1 = sh.bbox
        for k, (s, w, n, e) in boxes.items():
            if x1 >= w and x0 <= e and y1 >= s and y0 <= n:
                out[k].append([tuple(pt) for pt in sh.points])
    return out


def region_boxes(src):
    for reg in REGIONS[src]:
        if src == "linz":
            name, target = reg
            _, col = linz_collection(name)
            assert col.get("license") == "CC-BY-4.0", (name, col.get("license"))
            w, s, e, n = col["extent"]["spatial"]["bbox"][0]
            yield name, (s, w, n, e), target
        else:
            region, s, w, n, e, target = reg
            yield region, (s, w, n, e), target


def cmd_points(args):
    META.mkdir(parents=True, exist_ok=True)
    regs = {(src, region): (box, target) for src in args.sources for region, box, target in region_boxes(src)}
    lines_by = shp_coast(args.shp, {k: v[0] for k, v in regs.items()}) if args.shp else None
    rng = random.Random(0)
    for src in args.sources:
        rows = []
        for (sr, region), ((s, w, n, e), target) in regs.items():
            if sr != src:
                continue
            lines = lines_by[(sr, region)] if lines_by is not None else overpass_coast(s, w, n, e)
            pts = sample_line_points(lines, s, w, n, e)
            rng.shuffle(pts)
            print(f"{src}/{region}: coastline ways {len(lines)}, candidate points {len(pts)} (target {target})", flush=True)
            rows += [dict(source=src, region=region, lon=f"{lo:.6f}", lat=f"{la:.6f}", target=target, rank=i)
                     for i, (lo, la) in enumerate(pts)]
        with open(META / f"points_{src}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["source", "region", "lon", "lat", "target", "rank"]); wr.writeheader(); wr.writerows(rows)


# --------------------------------------------------------------------------- 소스별 fetch → (RGB uint8 1024², extra dict) 또는 None
SESSION = threading.local()


def sess():
    if not hasattr(SESSION, "s"):
        SESSION.s = requests.Session(); SESSION.s.headers.update(UA)
    return SESSION.s


def get(url, **kw):
    for attempt in range(4):
        try:
            r = sess().get(url, timeout=kw.pop("timeout", 120), **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(5 * (attempt + 1)); continue
            return r
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    return None


def fetch_gsi(lon, lat, reg):
    z = 18
    res = 2 * math.pi * R_EARTH / 256 / 2 ** z
    x0, y0, x1, y1 = bbox3857(lon, lat)
    px0, px1 = (x0 + math.pi * R_EARTH) / res, (x1 + math.pi * R_EARTH) / res
    py0, py1 = (math.pi * R_EARTH - y1) / res, (math.pi * R_EARTH - y0) / res
    tx0, tx1, ty0, ty1 = int(px0 // 256), int(px1 // 256), int(py0 // 256), int(py1 // 256)
    mos = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), np.uint8)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            r = get(f"https://cyberjapandata.gsi.go.jp/xyz/ort/{z}/{tx}/{ty}.jpg")
            if r is None or r.status_code != 200 or not r.headers.get("content-type", "").startswith("image"):
                return None
            t = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            if t is None:
                return None
            mos[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = t[..., ::-1]
            time.sleep(0.05)
    c = mos[int(py0 - ty0 * 256):int(py1 - ty0 * 256), int(px0 - tx0 * 256):int(px1 - tx0 * 256)]
    img = cv2.resize(c, (OUT_PX, OUT_PX), interpolation=cv2.INTER_CUBIC)
    return img, dict(layer="ort z18", native_gsd_m=round(res * math.cos(math.radians(lat)), 3), request=f"xyz ort z18 tiles x{tx0}-{tx1} y{ty0}-{ty1}")


NAIP_URL = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer"


def fetch_naip(lon, lat, reg):
    import rasterio
    q = get(f"{NAIP_URL}/query", params=dict(geometry=f"{lon},{lat}", geometryType="esriGeometryPoint", inSR=4326,
                                               spatialRel="esriSpatialRelIntersects", where="Category=1 AND resolution_value<=0.3",
                                               outFields="Name,State,Year,resolution_value", returnGeometry="false", f="json"))
    feats = (q.json().get("features") if q is not None and q.status_code == 200 else None) or []
    if not feats:
        return None
    best = max(feats, key=lambda f: f["attributes"]["Year"])["attributes"]
    x0, y0, x1, y1 = bbox3857(lon, lat)
    mr = json.dumps({"mosaicMethod": "esriMosaicAttribute", "sortField": "Year", "sortValue": "2100", "ascending": True,
                     "where": "Category=1 AND resolution_value<=0.3"})
    r = get(f"{NAIP_URL}/exportImage", params=dict(bbox=f"{x0},{y0},{x1},{y1}", bboxSR=3857, imageSR=3857, size=f"{OUT_PX},{OUT_PX}",
                                                    format="tiff", bandIds="0,1,2,3", interpolation="RSP_BilinearInterpolation",
                                                    mosaicRule=mr, f="image"), timeout=180)
    if r is None or r.status_code != 200 or "tiff" not in r.headers.get("content-type", ""):
        return None
    with rasterio.MemoryFile(r.content) as m, m.open() as d:
        a = d.read()
    if a.shape[0] < 4:
        return None
    return np.ascontiguousarray(a[:3].transpose(1, 2, 0)), dict(layer=f"NAIP {best['State']} {best['Year']} ({best['Name']})",
                                                               native_gsd_m=best["resolution_value"], nir=a[3],
                                                               request="exportImage bbox3857 " + ",".join(f"{v:.1f}" for v in (x0, y0, x1, y1)))


_LINZ_IDX = {}
_LINZ_LOCK = threading.Lock()


def linz_index(name):
    with _LINZ_LOCK:
        if name in _LINZ_IDX:
            return _LINZ_IDX[name]
        base, col = linz_collection(name)
        hrefs = [l["href"][2:] for l in col["links"] if l["rel"] == "item"]
        def one(h):
            for _ in range(3):
                try:
                    it = requests.get(base + h, headers=UA, timeout=60).json()
                    return it["bbox"], base + it["assets"]["visual"]["href"][2:]
                except Exception:                                        # noqa: BLE001
                    time.sleep(2)
            return None
        with ThreadPoolExecutor(32) as ex:
            items = [x for x in ex.map(one, hrefs) if x]
        print(f"  linz index {name}: {len(items)} items", flush=True)
        _LINZ_IDX[name] = items
        return items


def fetch_linz(lon, lat, reg):
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import transform
    items = linz_index(reg)
    dlon, dlat = 200 / (111320 * math.cos(math.radians(lat))), 200 / 110540
    hit = [u for (w, s, e, n), u in items if w - dlon <= lon <= e + dlon and s - dlat <= lat <= n + dlat]
    if not hit:
        return None
    xs, ys = transform("EPSG:4326", "EPSG:2193", [lon], [lat])
    cx, cy = xs[0], ys[0]
    b = (cx - HALF_M, cy - HALF_M, cx + HALF_M, cy + HALF_M)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff,.tif"):
        ds = [rasterio.open("/vsicurl/" + u) for u in hit]
        try:
            ds = [d for d in ds if not (d.bounds.right < b[0] or d.bounds.left > b[2] or d.bounds.top < b[1] or d.bounds.bottom > b[3])]
            if not ds:
                return None
            gsd = ds[0].res[0]
            arr, _ = merge(ds, bounds=b, res=GSD, resampling=rasterio.enums.Resampling.average, nodata=0, indexes=[1, 2, 3])
        finally:
            for d in ds:
                d.close()
    img = np.ascontiguousarray(arr.transpose(1, 2, 0))[:OUT_PX, :OUT_PX]
    if img.shape[:2] != (OUT_PX, OUT_PX):
        img = cv2.resize(img, (OUT_PX, OUT_PX), interpolation=cv2.INTER_AREA)
    return img, dict(layer=f"LINZ {reg}", native_gsd_m=round(gsd, 3), request="COG merge " + ";".join(h.split("/")[-1] for h in hit))


def fetch_ign(lon, lat, reg):
    x0, y0, x1, y1 = bbox3857(lon, lat)
    n = int(OUT_PX * 1.5)                                                # 0.2 m 원본 해상도로 받아 면적 평균 축소
    r = get("https://data.geopf.fr/wms-r", params=dict(SERVICE="WMS", VERSION="1.3.0", REQUEST="GetMap", LAYERS="HR.ORTHOIMAGERY.ORTHOPHOTOS",
                                                       STYLES="", CRS="EPSG:3857", BBOX=f"{x0},{y0},{x1},{y1}", WIDTH=n, HEIGHT=n, FORMAT="image/jpeg"))
    if r is None or r.status_code != 200 or not r.headers.get("content-type", "").startswith("image"):
        return None
    a = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    if a is None:
        return None
    img = cv2.resize(a[..., ::-1], (OUT_PX, OUT_PX), interpolation=cv2.INTER_AREA)
    return img, dict(layer="IGN HR.ORTHOIMAGERY.ORTHOPHOTOS", native_gsd_m=0.2, request="WMS bbox3857 " + ",".join(f"{v:.1f}" for v in (x0, y0, x1, y1)))


_ERI_KEYS = None
_ERI_LOCK = threading.Lock()
ERI_FLIGHTS = ["2023_Pre_Event/GC2301a_OB_N_RGB", "2023_Pre_Event/GC2301b_OB_N_RGB", "2023_Pre_Event/EC2301a_OB_N_RGB",
               "2023_Pre_Event/WC2301a_OB_N_RGB", "2022_Pre_Event/GC2201a_OB_N_RGB", "2022_Pre_Event/GC2201b_OB_N_RGB",
               "2022_Pre_Event/EC2201a_OB_N_RGB", "2022_Pre_Event/CB2201a_OB_N_RGB", "2022_Pre_Event/WC2201a_OB_N_RGB"]


def eri_keys():
    """비행별 타일 키 전체 목록 → {(서경 초, 북위 초): [key...]} (파일명 = 타일 북서 모서리)."""
    global _ERI_KEYS
    with _ERI_LOCK:
        if _ERI_KEYS is not None:
            return _ERI_KEYS
        idx = {}
        pat = re.compile(r"C(\d{3})(\d{2})(\d{2})w(\d{2})(\d{2})(\d{2})n\.tif$")
        for fl in ERI_FLIGHTS:
            token, n = None, 0
            while True:
                p = dict(**{"list-type": "2", "prefix": fl + "/", "max-keys": "1000"})
                if token:
                    p["continuation-token"] = token
                t = requests.get("https://noaa-eri-pds.s3.amazonaws.com/", params=p, timeout=60).text
                for k in re.findall(r"<Key>([^<]+)</Key>", t):
                    m = pat.search(k)
                    if m:
                        a = list(map(int, m.groups()))
                        idx.setdefault((a[0] * 3600 + a[1] * 60 + a[2], a[3] * 3600 + a[4] * 60 + a[5]), []).append(k); n += 1
                m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", t)
                if not m:
                    break
                token = m.group(1)
            print(f"  eri {fl}: {n} tiles", flush=True)
        _ERI_KEYS = idx
        return idx


def fetch_eri(lon, lat, reg):
    import rasterio
    from rasterio.merge import merge
    idx = eri_keys()
    dlon, dlat = HALF_M / (111320 * math.cos(math.radians(lat))), HALF_M / 110540
    b = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    need, keys = set(), []
    for qlon in (b[0], b[2]):
        for qlat in (b[1], b[3]):
            wsec = int(math.ceil(-qlon * 3600 / 45.0 - 1e-9) * 45)       # 서경: 북서 모서리 = 더 서쪽(큰 값)
            nsec = int(math.ceil(qlat * 3600 / 45.0 - 1e-9) * 45)
            need.add((wsec, nsec))
    for k in need:
        if k not in idx:
            return None
        keys.append(sorted(idx[k])[-1])                                   # 가장 최근 비행
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        ds = [rasterio.open("/vsicurl/https://noaa-eri-pds.s3.amazonaws.com/" + k) for k in keys]
        try:
            res = (2 * dlon / OUT_PX, 2 * dlat / OUT_PX)
            arr, _ = merge(ds, bounds=b, res=res, resampling=rasterio.enums.Resampling.average, indexes=[1, 2, 3, 4])
        finally:
            for d in ds:
                d.close()
    img = np.ascontiguousarray(arr[:3].transpose(1, 2, 0))[:OUT_PX, :OUT_PX]
    if img.shape[:2] != (OUT_PX, OUT_PX) or (arr[3][:OUT_PX, :OUT_PX] == 0).mean() > 0.005:   # 알파 0 = 촬영 밖
        return None
    return img, dict(layer="ERI " + ",".join(sorted({k.split("/")[1] for k in keys})), native_gsd_m=0.3, request="COG merge " + ";".join(k.split("/")[-1] for k in keys))


FETCH = dict(gsi=fetch_gsi, naip=fetch_naip, linz=fetch_linz, ign=fetch_ign, eri=fetch_eri)
WORKERS = dict(gsi=4, naip=8, linz=8, ign=4, eri=8)


def qa(img):
    """촬영 밖(순흑·순백)·균일 장면을 걸러낸다. 합격이면 None, 아니면 사유."""
    black = (img.max(-1) <= 2).mean()
    white = (img.min(-1) >= 253).mean()
    if black > 0.005 or white > 0.02:
        return f"nodata black={black:.3f} white={white:.3f}"
    if img.std() < 6:
        return "uniform"
    return None


def cmd_fetch(args):
    src = args.source
    out = ROOT / src; out.mkdir(parents=True, exist_ok=True)
    pts = list(csv.DictReader(open(META / f"points_{src}.csv")))
    ledger_p = META / f"ledger_{src}.csv"
    done = {r["file"] for r in csv.DictReader(open(ledger_p))} if ledger_p.exists() else set()
    new_ledger = not ledger_p.exists()
    lf = open(ledger_p, "a", newline=""); lw = csv.DictWriter(lf, fieldnames=LEDGER_COLS)
    if new_ledger:
        lw.writeheader()
    lock = threading.Lock()
    by_region = {}
    for p in pts:
        by_region.setdefault(p["region"], []).append(p)
    got = {reg: sum(1 for f in done if f.startswith(f"{src}_{reg}_")) for reg in by_region}
    stats = dict(ok=0, fail=0, qa=0)

    def job(p):
        reg = p["region"]; target = int(p["target"])
        with lock:
            if got[reg] >= target:
                return
        fn = f"{src}_{reg}_{int(p['rank']):05d}.jpg"
        if fn in done:
            return
        lon, lat = float(p["lon"]), float(p["lat"])
        try:
            res = FETCH[src](lon, lat, reg)
        except Exception as ex:                                             # noqa: BLE001
            res = None
            if args.verbose:
                print("  err", fn, repr(ex)[:120], flush=True)
        if res is None:
            with lock:
                stats["fail"] += 1
            return
        img, extra = res
        why = qa(img)
        if why:
            with lock:
                stats["qa"] += 1
            return
        with lock:
            if got[reg] >= target:
                return
            got[reg] += 1
        cv2.imwrite(str(out / fn), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        if "nir" in extra:
            (ROOT / f"{src}_nir").mkdir(exist_ok=True)
            cv2.imwrite(str(ROOT / f"{src}_nir" / fn.replace(".jpg", ".png")), extra.pop("nir"))
        lic, lic_url = LICENSE[src]
        with lock:
            lw.writerow(dict(file=fn, source=src, region=reg, layer=extra["layer"], lon=f"{lon:.6f}", lat=f"{lat:.6f}",
                             native_gsd_m=extra["native_gsd_m"], license=lic, license_url=lic_url, request=extra["request"],
                             fetched_at=time.strftime("%Y-%m-%d %H:%M:%S")))
            lf.flush()
            stats["ok"] += 1
            if stats["ok"] % 25 == 0:
                print(f"[{src}] ok {stats['ok']} fail {stats['fail']} qa {stats['qa']} | " +
                      " ".join(f"{k}:{v}" for k, v in got.items()), flush=True)

    # 지역별로 번갈아 가며 제출(한 지역이 모자라도 다른 지역 진행이 막히지 않게)
    order, i = [], 0
    lists = [sorted(v, key=lambda r: int(r["rank"])) for v in by_region.values()]
    while any(i < len(l) for l in lists):
        order += [l[i] for l in lists if i < len(l)]; i += 1
    with ThreadPoolExecutor(WORKERS[src]) as ex:
        for f in as_completed([ex.submit(job, p) for p in order]):
            f.result()
    lf.close()
    print(f"[{src}] DONE ok {stats['ok']} fail {stats['fail']} qa {stats['qa']} | " + " ".join(f"{k}:{v}" for k, v in got.items()), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("points"); p1.add_argument("--sources", nargs="+", default=list(REGIONS)); p1.add_argument("--shp", help="OSM coastlines-split-4326 lines.shp (없으면 Overpass)")
    p2 = sub.add_parser("fetch"); p2.add_argument("--source", required=True, choices=list(FETCH)); p2.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    cmd_points(a) if a.cmd == "points" else cmd_fetch(a)
