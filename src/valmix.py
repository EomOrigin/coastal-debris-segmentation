"""valmix.py — v4/v5 모델 선택용 '강건성 검증셋' (M1, docs/v4_병목분석_개선계획.md §3.3·§4)

v2 는 기본 합성 1종으로 체크포인트·임계를 골랐고, 그 선택이 '색 점' 규칙을 가장 굳힌 지점이었다.
평가 영상에서 예상되는 외관 변화를 섞은 검증셋으로 고른다. 구성과 변환은 research/v2_diagnosis/stress_panel.py 와 같다.

  구성(비율)  base 0.30 · chroma_1.5 0.125 · chroma_2.0 0.125 · sat_look 0.20 · achromatic 0.15 · desat 0.10
  생성기      v1 합성기(SynthCfg) + 집적대 크기만 LB 역산 규모(띠 폭 3~18, 덩어리 반경 5~26)
  배경        train.py 가 나눈 검증 장면(학습 장면과 겹치지 않음). 각 표본의 난수는 (seed, 성분, 번호)로 고정.
"""
from dataclasses import replace

import cv2
import numpy as np

import synth as S

COMPONENTS = [("base", 0.30), ("chroma_1.5", 0.125), ("chroma_2.0", 0.125), ("sat_look", 0.20),
              ("achromatic", 0.15), ("desat", 0.10)]
# v5: v4 진단(research/v4_diagnosis)의 약점을 선택 기준에 넣는다 — 크롭 스트레치(헛경보), 해안선 쌍둥이 음성이 섞인 패치,
#     밝기가 섞인 무채색 집적대(대회 정의의 '얼룩덜룩'). 색 번짐 σ1.5 는 v1 의 LB 성분을 재현한 대리 지표라 비중을 올린다.
COMPONENTS_V5 = [("base", 0.15), ("chroma_1.5", 0.20), ("chroma_2.0", 0.10), ("sat_look", 0.15), ("stretch", 0.10),
                 ("shellband", 0.10), ("achro_hetero", 0.10), ("achromatic", 0.05), ("desat", 0.05)]


def _gen_cfg():
    return replace(S.SynthCfg(), band_width=(3.0, 18.0), clump_r=(5.0, 26.0))


def sat_look(img, rng):
    """위성 룩 근사(크롭 단위): 물을 어둡고 푸르게 + 휘도 블러 + 색 번짐 σ1.5 + 영상 단위 2~98% 스트레치."""
    f = img.astype(np.float32)
    water = S.water_mask(img)
    f[water] = f[water] * rng.uniform(0.25, 0.5) * np.array([0.8, 0.9, 1.15], np.float32)
    f = cv2.GaussianBlur(f, (0, 0), 0.6)
    out = S.chroma_blur(np.clip(f, 0, 255).astype(np.uint8), 1.5).astype(np.float32)
    lo = np.percentile(out, 2, axis=(0, 1))
    hi = float(np.percentile(out, 98, axis=(0, 1)).max())
    return np.clip((out - lo) / max(1.0, hi - float(lo.min())) * 255, 0, 255).astype(np.uint8)


def _stretch(img, rng):
    f = img.astype(np.float32)
    lo = np.percentile(f, 2, axis=(0, 1))
    hi = float(np.percentile(f, 98, axis=(0, 1)).max())
    return np.clip((f - lo) / max(1.0, hi - float(lo.min())) * 255, 0, 255).astype(np.uint8)


def _component(name):
    """→ (SynthCfg, 영상 후처리 fn(img, rng) 또는 None)"""
    base = _gen_cfg()
    if name == "stretch":
        return base, _stretch
    if name == "shellband":                                     # 모든 패치에 해안선 쌍둥이 음성(균일한 흰 알갱이 띠)을 얹음
        return replace(base, p_hardneg=1.0, hardneg_kinds={"shellband": 1.0}), None
    if name == "achro_hetero":                                  # 무채색이지만 밝기 5단계가 섞인 집적대
        return replace(base, palette={k: v for k, v in S.V5_PALETTE.items() if k in S.ACHROMATIC}), None
    if name == "base":
        return base, None
    if name == "chroma_1.5":
        return base, lambda im, r: S.chroma_blur(im, 1.5)
    if name == "chroma_2.0":
        return base, lambda im, r: S.chroma_blur(im, 2.0)
    if name == "sat_look":
        return base, sat_look
    if name == "achromatic":
        return replace(base, palette={k: v for k, v in base.palette.items() if k in ("white", "gray")}), None
    if name == "desat":
        return replace(base, palette={k: (tuple(0.35 * c + 0.65 * 0.62 for c in rgb), w) for k, (rgb, w) in base.palette.items()}), None
    raise KeyError(name)


def make_robust_val(paths, n, seed=12345, recipe="v4"):
    """→ imgs (n,256,256,3) uint8, masks (n,256,256) bool, groups (n,) 성분 이름"""
    comps = COMPONENTS_V5 if recipe == "v5" else COMPONENTS
    bank = S.BackgroundBank(paths)
    counts = [int(round(n * w)) for _, w in comps]
    counts[0] += n - sum(counts)
    xs, ms, gs = [], [], []
    for ci, ((name, _), c) in enumerate(zip(comps, counts)):
        cfg, post = _component(name)
        for j in range(c):
            rng = np.random.default_rng([seed, ci, j])
            img, m, _ = S.draw(rng, bank, cfg)
            if post is not None:
                img = post(img, rng)
            xs.append(img); ms.append(m); gs.append(name)
    return np.stack(xs), np.stack(ms), np.array(gs)
