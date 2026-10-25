"""build_verifier_data.py — 2차 분류기(v16) 학습 특징 만들기. 컨테이너 coastal, GPU0.

1차(v4, D4 — 제출 노트북과 같은 계산)가 '속는' 표본만 모은다.
  음성(실제): v4 채굴 11,722 · v2 채굴 4,667 · PDOK 와덴해·젤란트 갯벌(신규, 원본/위성 룩/크롭 스트레치) · 태안 드론 0.3 m 타일
  음성(합성): 합성 혼동물(shellband·foam·gravel·wrack·puddle·buoys)만 얹은 배경
  양성: 합성 쓰레기(v4 레시피 외관 그대로, v4 가 잡은 것) · 태안 실측 쓰레기 146 타일(GT 중심 크롭)
음성은 v4 존재 ≥ 0.25, 합성 양성은 v4 존재 ≥ 0.4 만 남긴다(2차는 1차 양성만 보므로).
특징 = DINOv2-B reg4 @448(원본+좌우반전 평균): [CLS ‖ 패치 평균 ‖ v4 마스크 가중 패치 평균] (2,304) + 1차 스칼라 4개.
    python build_verifier_data.py      → scripts/verifier/feats.npz (+ 시트 sheet_*.jpg)
"""
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NB = ROOT / "src"
sys.path.insert(0, str(HERE))
from diag_common import predict_ours                                          # noqa: E402
from train import Net, split_scenes                                           # noqa: E402
import synth as S                                                             # noqa: E402

BG = ROOT / "data/backgrounds"
IMN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
NEG_MIN, POS_MIN = 0.25, 0.4


def load_v4():
    cfg = json.loads((NB / "runs/v4_r34/config.json").read_text())
    net = Net(cfg["arch"], cfg["encoder"], cfg["upscale"], encoder_weights=None).cuda().eval()
    sd = torch.load(NB / "runs/v4_r34/weights_fp16.pt", map_location="cpu", weights_only=True)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    return net


def load_dino():
    m = timm.create_model("vit_base_patch14_reg4_dinov2", pretrained=False, num_classes=0, img_size=448)
    sd = torch.load(HERE / "dinov2_vitb14_reg4_448_fp16.pt", map_location="cpu", weights_only=True)
    m.load_state_dict({k: v.float() for k, v in sd.items()})
    return m.cuda().eval()


@torch.no_grad()
def dino_feats(model, imgs, qmask, bs=64):
    """imgs (N,256,256,3) uint8, qmask (N,256,256) v4 마스크 확률 → (N, 2304) float16"""
    out = []
    for s in range(0, len(imgs), bs):
        x = torch.from_numpy(np.ascontiguousarray(imgs[s:s + bs])).cuda().permute(0, 3, 1, 2).float() / 255.0
        x = (x - IMN_MEAN.cuda()) / IMN_STD.cuda()
        x = F.interpolate(x, size=448, mode="bilinear", align_corners=False)
        q = torch.from_numpy(np.ascontiguousarray(qmask[s:s + bs])).cuda().float()[:, None]
        acc = 0.0
        for flip in (False, True):
            xx, qq = (x.flip(-1), q.flip(-1)) if flip else (x, q)
            with torch.autocast("cuda", dtype=torch.float16):
                t = model.forward_features(xx)
            t = t.float()
            cls, pt = t[:, 0], t[:, model.num_prefix_tokens:]
            w = F.adaptive_avg_pool2d(qq, 32).flatten(1)
            w = w / (w.sum(1, keepdim=True) + 1e-6)
            acc = acc + torch.cat([cls, pt.mean(1), (pt * w[..., None]).sum(1)], 1)
        out.append((acc / 2).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def scalars(prob, pres):
    area = (prob >= 0.4).reshape(len(prob), -1).sum(1)
    pm = np.where(prob >= 0.4, prob, 0).reshape(len(prob), -1).sum(1) / np.maximum(area, 1)
    lg = np.log(np.clip(pres, 1e-4, 1 - 1e-4) / (1 - np.clip(pres, 1e-4, 1 - 1e-4)))
    return np.stack([lg, np.log1p(area), prob.reshape(len(prob), -1).max(1), pm], 1).astype(np.float32)


def stretch(c):
    f = c.astype(np.float32)
    lo = np.percentile(f, 2, axis=(0, 1)); hi = float(np.percentile(f, 98, axis=(0, 1)).max())
    return np.clip((f - lo) / max(1.0, hi - float(lo.min())) * 255, 0, 255).astype(np.uint8)


def read_rgb(p):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ 표본 공급원 (이미지, 종류, 그룹) 생성기
def src_mined(folder, tag):
    rows = list(csv.DictReader(open(BG / folder / "mined.csv")))
    for r in rows:
        im = read_rgb(BG / folder / "crops" / r["file"])
        if im is not None and im.shape[:2] == (256, 256):
            yield im, f"{tag}_{r['source']}", r["scene"]


def src_scenes(paths, tag, seed):
    """1024² 장면 → 4x4 크롭 × (원본 · 장면 위성 룩 · 크롭 스트레치)"""
    for k, p in enumerate(paths):
        s = read_rgb(p)
        if s is None or min(s.shape[:2]) < 1024:
            continue
        sat = S.sat_scene(np.random.default_rng([seed, k]), s)
        for r in range(4):
            for c in range(4):
                crop = np.ascontiguousarray(s[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256])
                yield crop, f"{tag}_raw", Path(p).name
                yield stretch(crop), f"{tag}_cstretch", Path(p).name
                yield np.ascontiguousarray(sat[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256]), f"{tag}_sat", Path(p).name


def src_kr(paths, seed):
    rng = np.random.default_rng(seed)
    for p in paths:
        t = read_rgb(p)
        if t is None or min(t.shape[:2]) < 256:
            continue
        y, x = rng.integers(0, t.shape[0] - 255), rng.integers(0, t.shape[1] - 255)
        crop = np.ascontiguousarray(t[y:y + 256, x:x + 256])
        yield crop, "kr_raw", Path(p).name
        yield stretch(crop), "kr_cstretch", Path(p).name
        yield S.sat_scene(rng, crop), "kr_sat", Path(p).name


def src_synth(bank, n, positive, seed):
    rng = np.random.default_rng(seed)
    cfg = S.cfg_v4()
    if not positive:
        from dataclasses import replace
        cfg = replace(cfg, p_hardneg=1.0, hardneg_kinds={"shellband": 0.3, "foam": 0.2, "gravel": 0.2, "wrack": 0.1, "puddle": 0.1, "buoys": 0.1})
    for i in range(n):
        img, m, _ = S.make_sample(rng, bank.crop(rng), cfg, force_pos=positive)
        if bool(m.any()) != positive:
            continue
        yield img, ("synth_pos" if positive else "synth_confuser"), f"synth{seed}_{i // 50}"


def src_real03(seed):
    rng = np.random.default_rng(seed)
    for p in sorted((ROOT / "data/aihub_71363/derived/real03/train").glob("*_img.png")):
        im, gt = read_rgb(p), cv2.imread(str(p).replace("_img", "_gt"), cv2.IMREAD_GRAYSCALE)
        if im is None or gt is None or not gt.any():
            continue
        ys, xs = np.nonzero(gt)
        cy = int(np.clip(ys.mean() + rng.integers(-40, 41), 128, im.shape[0] - 128))
        cx = int(np.clip(xs.mean() + rng.integers(-40, 41), 128, im.shape[1] - 128))
        yield np.ascontiguousarray(im[cy - 128:cy + 128, cx - 128:cx + 128]), "real_debris", p.name[:12]


# ------------------------------------------------------------------ 처리
def run_source(name, gen, label, keep_min, v4, dino, store, batch=256):
    t0, buf, n_in, n_keep = time.time(), [], 0, 0

    def flush():
        nonlocal buf, n_in, n_keep
        if not buf:
            return
        imgs = np.stack([b[0] for b in buf])
        prob, pres = predict_ours(v4, imgs, tta="d4", bs=64)
        keep = pres >= keep_min if keep_min is not None else np.ones(len(buf), bool)
        if keep.any():
            ki = np.flatnonzero(keep)
            store["feat"].append(dino_feats(dino, imgs[ki], prob[ki]))
            store["scal"].append(scalars(prob[ki], pres[ki]))
            store["label"].extend([label] * len(ki))
            store["kind"].extend([buf[i][1] for i in ki])
            store["group"].extend([buf[i][2] for i in ki])
            store["pres"].append(pres[ki])
            th = store["thumbs"].setdefault(name, [])
            for i in ki[:max(0, 40 - len(th))]:
                th.append((imgs[i], float(pres[i])))
        n_in += len(buf); n_keep += int(keep.sum()); buf = []

    for item in gen:
        buf.append(item)
        if len(buf) >= batch:
            flush()
    flush()
    print(f"[{name}] 입력 {n_in} → 유지 {n_keep} ({n_keep / max(n_in, 1):.1%}), {time.time() - t0:.0f}s", flush=True)


def sheet(store):
    for name, items in store["thumbs"].items():
        tiles = []
        for im, q in items[:40]:
            t = cv2.cvtColor(cv2.resize(im, (160, 160)), cv2.COLOR_RGB2BGR)
            cv2.putText(t, f"{q:.2f}", (4, 14), 0, 0.45, (0, 255, 255), 1)
            tiles.append(t)
        while len(tiles) % 8:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.concatenate(tiles[i:i + 8], 1) for i in range(0, len(tiles), 8)]
        cv2.imwrite(str(HERE / f"sheet_{name}.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 80])


def main():
    v4, dino = load_v4(), load_dino()
    store = dict(feat=[], scal=[], label=[], kind=[], group=[], pres=[], thumbs={})
    tr, _ = split_scenes(S.background_paths([str(BG / "splits/v2_train.txt")]), 0)
    bank = S.BackgroundBank(tr, sat_p=0.6, center_p=0.5)
    pdok = sorted((BG / "pdok_tidal").glob("*.jpg"))
    kr = sorted((ROOT / "data/aihub_71363/derived/kr_bg/train").glob("*.jpg"))
    run_source("mined_v4", src_mined("mined_negatives_v4", "minedv4"), 0, NEG_MIN, v4, dino, store)
    run_source("mined_v2", src_mined("mined_negatives_v2", "minedv2"), 0, NEG_MIN, v4, dino, store)
    run_source("pdok_tidal", src_scenes(pdok, "pdok", 16), 0, NEG_MIN, v4, dino, store)
    run_source("kr_drone", src_kr(kr, 17), 0, NEG_MIN, v4, dino, store)
    run_source("synth_confuser", src_synth(bank, 9000, False, 18), 0, NEG_MIN, v4, dino, store)
    run_source("synth_pos", src_synth(bank, 9000, True, 19), 1, POS_MIN, v4, dino, store)
    run_source("real_debris", src_real03(20), 1, None, v4, dino, store)
    np.savez(HERE / "feats.npz", feat=np.concatenate(store["feat"]), scal=np.concatenate(store["scal"]),
             label=np.array(store["label"], np.int8), kind=np.array(store["kind"]), group=np.array(store["group"]),
             pres=np.concatenate(store["pres"]))
    sheet(store)
    lab = np.array(store["label"])
    print("저장 feats.npz: 음성", int((lab == 0).sum()), "양성", int((lab == 1).sum()))


if __name__ == "__main__":
    main()
