#!/usr/bin/env python
"""synth.py — 해안쓰레기 집적대 절차적 합성기 (v1 기본 + v4 확장: cfg_v4())

주최 측 데이터 설명을 '렌더링 규칙'으로 옮긴 것이다.
  * 대상은 조각이 아니라 집적대: 해안선을 따라가는 띠(band) 또는 한자리에 뭉친 덩어리(clump)
  * 조각 하나는 0.1~0.3 m(=0.3~1 화소)가 대부분 → 초해상(ss배) 캔버스에 그린 뒤 면적 평균으로 줄인다
  * 정답 = '조각이 덮은 화소'. 화소 피복률 cov >= cov_thr 인 화소만 1 → 한 화소에 못 미치는 조각은 표시되지 않고,
    조각 사이 좁은 틈은 0 으로 남으며, 오목·다조각 형태가 자연히 생긴다
  * 팔레트: 흰색·회색·청색·녹색 위주의 밝은 점/얼룩 + 소수의 기타색
  * 정답 제외 대상(하드 네거티브): 포말 선, 밝은 자갈/조개 무더기, 해조류·유목의 어두운 띠, 물고임
  * 장면별 톤 조정: 합성 '후'에 감마/채널이득/채도/연무를 장면 단위로 흔든다(평가 영상이 장면마다 따로 스트레치됨)

의존: numpy, scipy, opencv(학습 환경 전용. 추론 노트북은 이 파일을 쓰지 않는다)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

H = W = 256


@dataclass
class SynthCfg:
    ss: int = 4                                   # 초해상 배율(1 화소 = ss x ss 서브화소)
    p_pos: float = 0.64                           # 양성 패치 비율. 공개 LB 역산값 62/97=0.639 에 맞춘다(설계서 §3.4)
    p_hardneg: float = 0.6                        # 패치마다 합성 하드 네거티브를 얹을 확률(양성·음성 공통)
    cov_thr: float = 0.5                          # GT 규칙: 화소 피복률 임계
    n_accum: Tuple[int, int] = (1, 3)             # 양성 패치당 집적대 수
    p_band: float = 0.65                          # 띠 vs 덩어리
    band_len: Tuple[float, float] = (25, 240)     # 띠 길이(화소)
    band_width: Tuple[float, float] = (1.5, 9.0)  # 띠 폭(화소)
    clump_r: Tuple[float, float] = (2.5, 14.0)    # 덩어리 반경(화소)
    shore_off: Tuple[float, float] = (2.0, 28.0)  # 수변에서 육지 쪽 거리(화소) — 만조선/랙 라인
    density: Tuple[float, float] = (0.25, 0.95)   # 영역 내 조각 밀도(서브화소 피복 목표)
    piece_med_px: Tuple[float, float] = (0.45, 1.3)   # 조각 크기 중앙값(화소) 범위, 로그정규
    piece_sigma: float = 0.55
    big_item_p: float = 0.04                      # 부표·어망 뭉치 같은 큰 물체(1.5~5 화소) 비율
    palette: dict = field(default_factory=lambda: {   # (R,G,B 0~1, 가중치)
        "white": ((0.97, 0.97, 0.96), 0.40), "gray": ((0.72, 0.73, 0.74), 0.18),
        "blue": ((0.20, 0.42, 0.85), 0.13), "green": ((0.18, 0.62, 0.42), 0.10),
        "cyan": ((0.35, 0.75, 0.85), 0.05), "red": ((0.85, 0.22, 0.18), 0.04),
        "orange": ((0.95, 0.55, 0.15), 0.04), "yellow": ((0.93, 0.85, 0.30), 0.03),
        "black": ((0.10, 0.10, 0.11), 0.03)})
    psf_sigma: Tuple[float, float] = (0.45, 0.95) # 센서 MTF 흉내(화소)
    noise_std: Tuple[float, float] = (0.0, 2.5)   # 8bit 잡음
    tiny_p: float = 0.12                          # 50 화소 미만의 아주 작은 집적대를 일부러 만든다(GT>=1 화소면 양성)
    scatter_n: Tuple[int, int] = (6, 40)         # v26(H2) 하드 네거티브 "scatter": 흩어진 조각 자리 수
    scatter_gap: float = 8.0                      # v26: 흩어진 조각 자리 사이 최소 거리(화소) — 집적대를 이루지 않게
    scatter_gap_hi: float = 0.0                   # v27: >scatter_gap 이면 최소 거리를 U(scatter_gap, scatter_gap_hi) 로 뽑는다
    scatter_cov: Tuple[float, float] = (1.0, 6.0)  # v27: 자리당 피복(화소) 범위 (v26 = 1~6)
    scatter_shore_p: float = 0.0                  # v27: 자리를 해안선 띠(육지, 물에서 2~30 px)에 놓을 확률(밀려온 쓰레기 선 흉내)
    pair_p: float = 0.0                           # v27: 반사실 짝(모은 작은 집적대 = 양성 / 같은 양을 흩은 것 = 음성)을 시작할 확률(학습 스트림이 사용)
    # ---- v4 확장 (기본값 = v1 동작 그대로. v4 값은 cfg_v4() — 근거는 docs/v4_병목분석_개선계획.md) ----
    palette_modes: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # D1 집적대별 색 구성 확률: (무채색만, 유채색 소수, 기본)
    colored_frac: Tuple[float, float] = (0.05, 0.45)  # D1 '유채색 소수' 모드의 유채색 조각 비율
    sat_scale: Tuple[float, float] = (1.0, 1.0)   # D1 집적대별 조각 채도 배율
    chroma_p: float = 0.0                         # D1 합성 직후 색만 번지게(팬샤픈·JPEG 4:2:0 흉내) 할 확률
    chroma_sigma: Tuple[float, float] = (0.3, 2.5)
    jpeg_p: float = 0.0                           # D1 마지막에 JPEG(4:2:0) 재인코딩할 확률
    jpeg_q: Tuple[int, int] = (70, 95)
    hardneg_kinds: dict = field(default_factory=lambda: {"foam": 0.35, "gravel": 0.3, "wrack": 0.2, "puddle": 0.15})
    neg_tint: float = 0.0                         # D1 자갈·조개에 허용하는 약한 색조 폭(색만으로 쓰레기/자갈을 못 가르게)
    sat_p: float = 0.0                            # D2 장면 단위 위성 룩(어두운 물 + 2~98% 스트레치) 확률 — BackgroundBank 가 적용
    center_p: float = 0.0                         # D2 크롭을 장면 중앙(=해안선 좌표) 쪽에서 뽑을 확률 — BackgroundBank 가 적용
    mined_p: float = 0.0                          # D4 음성 표본을 실배경 헛경보 채굴 크롭에서 뽑을 확률(draw 에 mined 뱅크 필요)
    # ---- v5 확장 (기본값 = 이전 동작. 근거: research/v4_diagnosis — 스트레치하면 배경만 있어도 헛경보 2배,
    #      흰·회색 놓침은 '속이 균일한' 집적대에 몰림 = 대회가 음성으로 정의한 '밝은 흰색 자갈·조개 무더기'와 같은 외관) ----
    crop_stretch_p: float = 0.0                   # 크롭 단위 퍼센타일 스트레치(양성·음성 모두) — 대비 불변 증강
    contrast_p: float = 0.0                       # 대비 증폭 ×U(1.2, 2.0) — 대비 불변 증강
    # ---- v6 확장: AI-Hub 71363 드론 실측 쓰레기 (기본 0 = 이전 동작) ----
    sprite_p: float = 0.0                         # 집적대를 실제 쓰레기 조각(스프라이트, 0.1 m)으로 채울 확률 — draw 에 sprites 뱅크 필요
    real_p: float = 0.0                           # 양성 표본을 실측 0.3 m 크롭(AI-Hub 학습 분할)에서 뽑을 확률 — draw 에 real 뱅크 필요
    piece_shade: Tuple[float, float] = (0.85, 1.05)  # 절차적 조각 밝기 = 장면 흰색(99%) x U(범위). v6: 실측 '주변보다 +15 밝음'에 맞춰 상향
    # ---- v15 확장: 팬샤픈형 색-밝기 어긋남 (기본 0 = 이전 동작). 근거: LB S2 프로브(평가셋 마스크 P 0.71/R 0.82 = 위치 밀림),
    #      기존 chroma_blur 는 색을 '중심을 지킨 채' 번지게만 해서 모델이 색 얼룩 위치를 쓰레기 위치로 배웠다 ----
    chroma_shift_p: float = 0.0                   # 색(Cb/Cr)만 무작위 방향으로 서브픽셀 이동할 확률 — 휘도와 GT 는 그대로(MS-pan 정합 오차)
    chroma_shift_px: Tuple[float, float] = (0.5, 3.5)  # 이동 크기(화소) 범위
    # ---- v17 확장: SDXL 생성 혼동물(대회 '정답 제외 대상' 5종) 스프라이트 폴더 — hardneg_kinds 의 gen_confuser 가 쓴다(기본 없음) ----
    confuser_dirs: Tuple[str, ...] = ()


ACHROMATIC = ("white", "lightgray", "gray", "midgray", "black")
V3_PALETTE = {                                    # v1 팔레트 + 어두운 조각(로프·그물) 10% (D3)
    "white": ((0.97, 0.97, 0.96), 0.38), "gray": ((0.72, 0.73, 0.74), 0.18),
    "blue": ((0.20, 0.42, 0.85), 0.11), "green": ((0.18, 0.62, 0.42), 0.09),
    "cyan": ((0.35, 0.75, 0.85), 0.04), "red": ((0.85, 0.22, 0.18), 0.04),
    "orange": ((0.95, 0.55, 0.15), 0.03), "yellow": ((0.93, 0.85, 0.30), 0.03),
    "black": ((0.10, 0.10, 0.11), 0.06), "navy": ((0.12, 0.16, 0.30), 0.04)}


V5_PALETTE = {                                    # v4 + 밝기 단계(밝은 회색·짙은 회색) → 무채색 집적대도 '얼룩덜룩'하게
    "white": ((0.97, 0.97, 0.96), 0.30), "lightgray": ((0.86, 0.86, 0.85), 0.08), "gray": ((0.72, 0.73, 0.74), 0.14),
    "midgray": ((0.55, 0.56, 0.57), 0.06), "blue": ((0.20, 0.42, 0.85), 0.11), "green": ((0.18, 0.62, 0.42), 0.09),
    "cyan": ((0.35, 0.75, 0.85), 0.04), "red": ((0.85, 0.22, 0.18), 0.04), "orange": ((0.95, 0.55, 0.15), 0.03),
    "yellow": ((0.93, 0.85, 0.30), 0.03), "black": ((0.10, 0.10, 0.11), 0.05), "navy": ((0.12, 0.16, 0.30), 0.03)}


V6_PALETTE = dict(V5_PALETTE, black=((0.10, 0.10, 0.11), 0.02), navy=((0.12, 0.16, 0.30), 0.01))   # 어두운 조각 8% → 3%


def cfg_v17() -> "SynthCfg":
    """v17 = v4 + 렌더러 혼합(집적대의 60% 를 SDXL 생성·AI-Hub 506·71363 스프라이트로) + SDXL 생성 혼동물(gen_confuser, 쓰레기와 같은
    배치 경로·GT 없음) + 센서·처리 증강 강화(색 번짐·색 밀림·PSF·잡음·JPEG·스트레치). 근거: docs/v17_1차재학습_후처리_분류기_설계.md"""
    from pathlib import Path
    c = cfg_v4()
    c.sprite_p = 0.6
    c.p_hardneg = 0.75
    c.hardneg_kinds = {"gen_confuser": 0.5, "shellband": 0.1, "foam": 0.1, "gravel": 0.1, "wrack": 0.07, "puddle": 0.05, "buoys": 0.08}
    c.chroma_p, c.chroma_sigma = 0.7, (0.3, 3.0)
    c.chroma_shift_p, c.chroma_shift_px = 0.5, (0.5, 3.5)
    c.psf_sigma = (0.4, 1.4)
    c.noise_std = (0.0, 10.0)
    c.jpeg_p, c.jpeg_q = 0.4, (60, 95)
    c.crop_stretch_p, c.contrast_p = 0.3, 0.1
    c.mined_p = 0.3
    c.confuser_dirs = (str(Path(__file__).resolve().parents[1] / "data/gen_confusers/train"),)
    return c


def cfg_v6() -> "SynthCfg":
    """v6 = v5 + AI-Hub 실측 쓰레기: 집적대의 35% 를 실제 조각 스프라이트로, 양성의 10% 를 실측 크롭으로.
    실측 보정(AI-Hub 검증 25곳, 0.3 m): 유채색 화소 0.3%, 쓰레기가 주변보다 +15 밝음, 내부 밝기 표준편차 30
      → 무채색만 집적대 0.15 → 0.45, 조각 채도 배율 0.3~1.0 → 0.2~0.8, 어두운 조각 비중 축소."""
    c = cfg_v5()
    c.sprite_p, c.real_p = 0.35, 0.10
    c.palette = dict(V6_PALETTE)
    c.palette_modes = (0.45, 0.45, 0.10)
    c.sat_scale = (0.2, 0.8)
    c.piece_shade = (1.0, 1.25)
    # v5 교훈: 해안선 쌍둥이 음성 35% 는 흰 쪽 쓰레기 재현율을 깎았다(진단 패널 밝기 섞인 무채색 0.781→0.741, 실측 검출률 0.284→0.034)
    c.hardneg_kinds = {"foam": 0.22, "gravel": 0.18, "wrack": 0.15, "puddle": 0.12, "buoys": 0.18, "shellband": 0.15}
    c.p_hardneg = 0.6
    return c


def cfg_v5() -> "SynthCfg":
    """v5 = v4 + 대비 불변 증강 + 해안선 쌍둥이 음성(shellband) + 양성 이질성 보강. docs/v4_병목분석_개선계획.md §4."""
    c = cfg_v4()
    c.palette = dict(V5_PALETTE)
    c.palette_modes = (0.15, 0.55, 0.30)                                          # 무채색만 0.25 → 0.15 (대신 밝기 5단계로 이질적)
    c.hardneg_kinds = {"foam": 0.18, "gravel": 0.15, "wrack": 0.12, "puddle": 0.08, "buoys": 0.12, "shellband": 0.35}
    c.p_hardneg = 0.7
    c.crop_stretch_p, c.contrast_p = 0.3, 0.2
    c.mined_p = 0.3
    return c


def cfg_v4() -> "SynthCfg":
    """v4 합성 설정. D1 색 지름길 끊기 · D2 위성 룩 · D3 집적대 형태·면적 · D4 채굴 음성."""
    return SynthCfg(
        band_len=(30, 260), band_width=(3.0, 18.0), clump_r=(5.0, 26.0),          # D3 면적 중앙 ~365 → LB 역산 600~1,100 px 규모
        big_item_p=0.15, palette=dict(V3_PALETTE),                                # D3 여러 화소짜리 얼룩 조각, 어두운 조각
        noise_std=(0.0, 8.0),                                                     # D2 (주최 측 영상이 화소 고주파가 더 많음)
        palette_modes=(0.25, 0.50, 0.25), colored_frac=(0.05, 0.45), sat_scale=(0.3, 1.0),
        chroma_p=0.7, chroma_sigma=(0.3, 2.5), jpeg_p=0.3, jpeg_q=(70, 95),
        hardneg_kinds={"foam": 0.28, "gravel": 0.25, "wrack": 0.17, "puddle": 0.12, "buoys": 0.18},
        neg_tint=0.1, sat_p=0.6, center_p=0.5, mined_p=0.2)


def cfg_v26() -> "SynthCfg":
    """v26(H2): v4 + '집적대를 이루지 않은 흩어진 조각' 음성(hardneg scatter). 평가셋 음성 22개가 우리 모델 전부에 쓰레기로 보이는 현상에 대한
    가설 = 흩어진 조각은 정답이 아니다. 양성·음성 공통으로 흩어진 조각을 얹고(양성이면 집적대만 GT), 50 px 미만 tiny 양성은 모순이라 끈다."""
    c = cfg_v4()
    c.p_hardneg = 0.7
    c.hardneg_kinds = {"scatter": 0.5, "foam": 0.14, "gravel": 0.12, "wrack": 0.09, "puddle": 0.06, "buoys": 0.09}
    c.tiny_p = 0.0
    return c


def cfg_v27() -> "SynthCfg":
    """v27: v26 이 '쓰레기 양이 적으면 음성'이라는 지름길을 배워 작은 양성을 잃은 것(docs/v27_판정_설계.md §1)을 고친다.
    양성은 v4 그대로(tiny 복원), 흩어짐 음성은 총량·위치를 다양화(간격 10~30 px, 자리당 1~8 px, 해안선 띠 30%),
    반사실 짝(같은 조각 묶음을 모으면 양성/흩으면 음성)으로 배치만 다르게. 간격 4~9 px 의 모호 구간은 만들지 않는다."""
    c = cfg_v26()
    c.tiny_p = 0.12
    c.scatter_n = (3, 60)
    c.scatter_gap, c.scatter_gap_hi = 10.0, 30.0
    c.scatter_cov = (1.0, 8.0)
    c.scatter_shore_p = 0.3
    c.pair_p = 0.15
    return c


def cfg_v28() -> "SynthCfg":
    """v28 일반화 판정(docs/v28_일반화_판정_설계.md): 음성 분류 체계 전부 + 양성 전 형태 + 실측 + 지름길 차단을 한 레시피로.
    음성: 흩어진 조각·SDXL 생성 혼동물(쓰레기 자리에)·포말·자갈·조개 띠·해조/유목·물고임·양식장 부표 줄 + 채굴/한국 실측 음성(mined_p)
    양성: v4 집적대 전 형태(작은 것 포함) + 실제 조각 스프라이트(sprite_p) + 한국 실측 타일(real_p) + 반사실 짝(v27).
    조개 띠(shellband)는 v5 에서 35% 일 때 실측 재현율을 무너뜨렸으므로 8% 로 제한."""
    from pathlib import Path
    c = cfg_v27()
    c.p_hardneg = 0.7
    c.hardneg_kinds = {"scatter": 0.22, "gen_confuser": 0.22, "foam": 0.10, "gravel": 0.10, "shellband": 0.08,
                       "wrack": 0.10, "puddle": 0.08, "buoys": 0.10}
    c.confuser_dirs = (str(Path(__file__).resolve().parents[1] / "data/gen_confusers/train"),)
    c.sprite_p, c.real_p = 0.35, 0.10
    c.mined_p = 0.3
    return c


def make_pair(rng, bg: np.ndarray, cfg: SynthCfg):
    """v27 반사실 짝: 같은 배경·같은 팔레트·같은 밝기·같은 PSF·같은 총 피복으로
    (가) 작은 덩어리/짧은 띠로 모은 것 = 양성(GT = 조각), (나) 패치 전체에 간격 ≥ 10 px 로 흩은 것 = 음성.
    → ((img_pos, mask_pos), (img_neg, empty)) 또는 만들 수 없으면 None. 센서 꼬리(finish)도 같은 난수로 둘에 똑같이."""
    img = bg.copy()
    water = water_mask(img)
    small = replace(cfg, clump_r=(3.0, 12.0), band_len=(20.0, 70.0), band_width=(2.0, 6.0))
    rho = accumulation_field(rng, small, water, rng.random() < 0.3)
    if rho is None or rho.sum() < 4:
        return None
    total = float(rho.sum())
    n = int(np.clip(round(total / rng.uniform(2.0, 6.0)), 3, 60))
    land = ~water
    cand = np.argwhere(land) if land.mean() > 0.05 else np.argwhere(np.ones((H, W), bool))
    gap2, pts = rng.uniform(10.0, 30.0) ** 2, []
    for _ in range(n * 25):
        if len(pts) >= n:
            break
        y, x = cand[rng.integers(len(cand))]
        if 2 <= y < H - 2 and 2 <= x < W - 2 and all((y - py) ** 2 + (x - px) ** 2 >= gap2 for py, px in pts):
            pts.append((int(y), int(x)))
    if len(pts) < 3:
        return None
    rs = np.zeros((H, W), np.float32)
    for y, x in pts:
        rs[y, x] = total / len(pts)
    rs = cv2.GaussianBlur(rs, (0, 0), 0.9)
    rs *= total / max(1e-6, float(rs.sum()))
    pal = zone_palette(rng, cfg)
    f0 = img.astype(np.float32) / 255.0
    white = float(np.clip(np.percentile(f0, 99.0), 0.55, 1.0))
    shade = min(1.0, white * rng.uniform(*cfg.piece_shade))
    sig = rng.uniform(*cfg.psf_sigma)
    seed_f = int(rng.integers(1 << 31))
    out = []
    for r in (rho, rs):
        cov, pm = render_pieces(rng, cfg, r, *pal)
        cov_b = cv2.GaussianBlur(cov, (0, 0), sig)[..., None]
        pm_b = cv2.GaussianBlur(pm, (0, 0), sig)
        f = f0 * (1 - cov_b) + pm_b * shade
        out.append((finish(np.random.default_rng(seed_f), np.clip(f * 255.0, 0, 255).astype(np.uint8), cfg), cov >= cfg.cov_thr))
    (ip, mp), (ineg, _) = out
    if not mp.any():
        return None
    return (ip, mp), (ineg, np.zeros((H, W), bool))


# --------------------------------------------------------------------------- 유틸
def smooth_noise(rng, shape, scale, octaves=3):
    """저주파 잡음 0~1. scale = 가장 굵은 옥타브의 격자 크기(화소)."""
    out = np.zeros(shape, np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = max(2, int(round(scale / (2 ** o))))
        g = rng.random((shape[0] // s + 3, shape[1] // s + 3)).astype(np.float32)
        out += amp * cv2.resize(g, (shape[1] + 2 * s, shape[0] + 2 * s), interpolation=cv2.INTER_CUBIC)[s:s + shape[0], s:s + shape[1]]
        tot += amp
        amp *= 0.5
    out /= tot
    return (out - out.min()) / max(1e-6, out.max() - out.min())


def water_mask(img: np.ndarray) -> np.ndarray:
    """거친 수역 마스크(색 + 저질감). 배치 위치를 고르는 용도다.
    v1 의 가장 약한 고리 → 실전에서는 배경을 받을 때 NAIP NIR(4번째 밴드)로 NDWI 수역 마스크를 같이 저장해 대체한다."""
    f = img.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    v = f.max(-1)
    gray = f.mean(-1)
    tex = np.sqrt(np.maximum(cv2.blur(gray * gray, (7, 7)) - cv2.blur(gray, (7, 7)) ** 2, 0))
    m = ((b - r) > 6) | (((g - r) > 10) & (tex < 9)) | (v < 60)
    m &= ~((r > g) & (g > b) & (v > 110))                     # 모래·암반(R>G>B, 밝음)은 물이 아니다
    pad = 8                                                     # 가장자리에서 closing 이 깎아 먹지 않게 복제 패딩
    mp = np.pad(m, pad, mode="edge")
    mp = ndi.binary_closing(ndi.binary_opening(mp, iterations=2), iterations=3)
    m = mp[pad:-pad, pad:-pad]
    lab, n = ndi.label(m)
    if n:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.flatnonzero(sizes >= 400))
    return m


# --------------------------------------------------------------------------- 집적대 영역(밀도장)
def accumulation_field(rng, cfg: SynthCfg, water: np.ndarray, tiny: bool):
    """(H,W) 밀도장 rho∈[0,1] 을 돌려준다. 수변이 있으면 수변 등거리선을 따라가는 띠, 없으면 임의 방향 띠/덩어리."""
    land = ~water
    if land.mean() < 0.03:                                      # 전부 바다인 패치에는 '해안에 쌓인' 쓰레기를 만들지 않는다
        return None
    has_shore = water.mean() > 0.02 and land.mean() > 0.05
    dist = ndi.distance_transform_edt(land) if has_shore else None
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    is_band = rng.random() < cfg.p_band and not tiny
    if is_band:
        length = rng.uniform(*cfg.band_len)
        width = rng.uniform(*cfg.band_width)
        if has_shore:
            d0 = rng.uniform(*cfg.shore_off)
            wmap = width * (0.55 + 0.9 * smooth_noise(rng, (H, W), 40))
            core = np.abs(dist - d0) < wmap / 2
            cand = np.argwhere(core & land)
            if len(cand) == 0:
                return None
            cy, cx = cand[rng.integers(len(cand))]
            window = ((yy - cy) ** 2 + (xx - cx) ** 2) < (length / 2) ** 2
            region = core & window & land
        else:
            th = rng.uniform(0, np.pi)
            cy, cx = rng.uniform(30, H - 30), rng.uniform(30, W - 30)
            u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
            v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
            wob = 10 * (smooth_noise(rng, (H, W), 60) - 0.5)
            wmap = width * (0.55 + 0.9 * smooth_noise(rng, (H, W), 40))
            region = (np.abs(v + wob) < wmap / 2) & (np.abs(u) < length / 2) & land
    else:
        r = rng.uniform(1.2, 3.2) if tiny else rng.uniform(*cfg.clump_r)
        if has_shore:
            cand = np.argwhere(land & (dist > 1.5) & (dist < cfg.shore_off[1]))
            if len(cand) == 0:
                cand = np.argwhere(land)
        else:
            cand = np.argwhere(land)
        cy, cx = cand[rng.integers(len(cand))]
        a, b = r * rng.uniform(0.7, 1.6), r * rng.uniform(0.6, 1.2)
        th = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        e = (u / a) ** 2 + (v / b) ** 2
        region = (e + 0.7 * (smooth_noise(rng, (H, W), 12) - 0.5)) < 1.0
        region &= land
    if region.sum() < 2:
        return None
    rho = region.astype(np.float32) * (0.35 + 0.65 * smooth_noise(rng, (H, W), 10))
    rho = cv2.GaussianBlur(rho, (0, 0), 0.8) * region           # 가장자리로 갈수록 성기게
    return rho * rng.uniform(*cfg.density) / max(1e-6, rho.max())


# --------------------------------------------------------------------------- 조각 렌더링
def zone_palette(rng, cfg: SynthCfg):
    """집적대 하나의 색 구성과 채도 배율(v4 D1). v1 설정이면 rng 를 쓰지 않고 cfg.palette 를 그대로 돌려준다."""
    if tuple(cfg.palette_modes) == (0.0, 0.0, 1.0) and tuple(cfg.sat_scale) == (1.0, 1.0):
        return cfg.palette, 1.0
    pm = np.asarray(cfg.palette_modes, np.float64)
    mode = rng.choice(3, p=pm / pm.sum())
    pal = dict(cfg.palette)
    if mode == 0:                                               # 무채색만(흰·회·검정)
        pal = {k: v for k, v in pal.items() if k in ACHROMATIC}
    elif mode == 1:                                             # 유채색은 소수
        cf = rng.uniform(*cfg.colored_frac)
        a = sum(v[1] for k, v in pal.items() if k in ACHROMATIC)
        c = sum(v[1] for k, v in pal.items() if k not in ACHROMATIC)
        pal = {k: (v[0], v[1] / a * (1 - cf) if k in ACHROMATIC else v[1] / c * cf) for k, v in pal.items()}
    return pal, float(rng.uniform(*cfg.sat_scale))


def render_pieces(rng, cfg: SynthCfg, rho: np.ndarray, palette: Optional[dict] = None, sat: float = 1.0):
    """초해상 캔버스에 조각을 그려 (cov(H,W) 0~1, premultiplied color(H,W,3)) 을 돌려준다.
    palette/sat: 집적대별 색 구성과 채도 배율(v4). 생략하면 cfg.palette, 채도 그대로(v1)."""
    palette = cfg.palette if palette is None else palette
    ss = cfg.ss
    hs, ws = H * ss, W * ss
    alpha = np.zeros((hs, ws), np.uint8)
    color = np.zeros((hs, ws, 3), np.uint8)
    ys, xs = np.nonzero(rho > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    med = rng.uniform(*cfg.piece_med_px)
    target = float(rho.sum())                                   # 목표 피복 면적(화소)
    names = list(palette)
    wts = np.array([palette[k][1] for k in names], np.float64)
    wts *= rng.uniform(0.4, 1.6, len(wts))                      # 집적대마다 색 구성이 다르다
    wts /= wts.sum()
    p = rho[y0:y1, x0:x1].astype(np.float64).ravel()
    p /= p.sum()
    covered, n_max = 0.0, 6000
    n_draw = int(min(n_max, max(3, target / (med ** 2) * 1.6)))
    idx = rng.choice(p.size, size=n_draw, p=p)
    py = (y0 + idx // (x1 - x0) + rng.random(n_draw)) * ss
    px = (x0 + idx % (x1 - x0) + rng.random(n_draw)) * ss
    size = np.exp(rng.normal(np.log(med), cfg.piece_sigma, n_draw))
    big = rng.random(n_draw) < cfg.big_item_p
    size[big] = rng.uniform(1.5, 5.0, big.sum())
    ck = rng.choice(len(names), size=n_draw, p=wts)
    for i in range(n_draw):
        if covered >= target:
            break
        s = size[i]
        a = max(1.0, s * ss * rng.uniform(0.6, 1.5) / 2)
        b = max(1.0, s * ss * rng.uniform(0.4, 1.0) / 2)
        base = np.array(palette[names[ck[i]]][0])
        col = np.clip(base * rng.uniform(0.75, 1.08) + rng.normal(0, 0.03, 3), 0, 1)
        if sat != 1.0:                                          # 채도 배율(휘도 유지)
            lum = float(col @ np.array([0.299, 0.587, 0.114]))
            col = np.clip(lum + (col - lum) * sat, 0, 1)
        c255 = tuple(int(v) for v in (col * 255))
        ctr = (int(px[i]), int(py[i]))
        ang = float(rng.uniform(0, 180))
        if rng.random() < 0.5:
            cv2.ellipse(alpha, ctr, (int(a), int(b)), ang, 0, 360, 255, -1)
            cv2.ellipse(color, ctr, (int(a), int(b)), ang, 0, 360, c255, -1)
        else:
            box = cv2.boxPoints((ctr, (2 * a, 2 * b), ang)).astype(np.int32)
            cv2.fillConvexPoly(alpha, box, 255)
            cv2.fillConvexPoly(color, box, c255)
        covered += np.pi * a * b / (ss * ss) * 0.8
    cov = cv2.resize(alpha.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_AREA)
    pm = cv2.resize(color.astype(np.float32) / 255.0 * (alpha[..., None] > 0), (W, H), interpolation=cv2.INTER_AREA)
    return cov, pm


def render_sprites(rng, cfg: SynthCfg, rho: np.ndarray, bank):
    """v6: 초해상 캔버스에 실제 쓰레기 조각(AI-Hub 드론 0.1 m RGBA)을 밀도장 rho 를 따라 붙인다. 물리 크기 유지(0.1 m → 캔버스 0.3/ss m).
    반환 형식은 render_pieces 와 같다(cov 0~1, premultiplied color)."""
    ss = cfg.ss
    hs, ws = H * ss, W * ss
    alpha = np.zeros((hs, ws), np.float32)
    color = np.zeros((hs, ws, 3), np.float32)
    ys, xs = np.nonzero(rho > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    p = rho[y0:y1, x0:x1].astype(np.float64).ravel(); p /= p.sum()
    target, covered, base = float(rho.sum()), 0.0, 0.1 / (0.3 / ss)
    for _ in range(400):
        if covered >= target:
            break
        i = rng.choice(p.size, p=p)
        cy = int((y0 + i // (x1 - x0) + rng.random()) * ss); cx = int((x0 + i % (x1 - x0) + rng.random()) * ss)
        spr = bank.get(rng)
        s = base * rng.uniform(0.7, 1.3)
        spr = cv2.resize(spr, (max(1, int(spr.shape[1] * s)), max(1, int(spr.shape[0] * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        spr = np.rot90(spr, rng.integers(4))
        if rng.random() < 0.5:
            spr = spr[:, ::-1]
        h, w = spr.shape[:2]
        ty0, tx0 = cy - h // 2, cx - w // 2
        sy0, sx0 = max(0, -ty0), max(0, -tx0)
        ty0, tx0 = max(0, ty0), max(0, tx0)
        ty1, tx1 = min(hs, cy - h // 2 + h), min(ws, cx - w // 2 + w)
        if ty1 <= ty0 or tx1 <= tx0:
            continue
        sp = spr[sy0:sy0 + (ty1 - ty0), sx0:sx0 + (tx1 - tx0)].astype(np.float32) / 255.0
        a = sp[..., 3:4]
        color[ty0:ty1, tx0:tx1] = color[ty0:ty1, tx0:tx1] * (1 - a) + sp[..., :3] * a
        alpha[ty0:ty1, tx0:tx1] = np.maximum(alpha[ty0:ty1, tx0:tx1], a[..., 0])
        covered += float(a.sum()) / (ss * ss)
    cov = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_AREA)
    pm = cv2.resize(color * (alpha[..., None] > 0), (W, H), interpolation=cv2.INTER_AREA)
    return cov, pm


class SpriteBank:
    """AI-Hub 실제 쓰레기 조각(RGBA PNG) — build_aihub_debris.py 의 sprites/<split>/ (학습 분할만 쓸 것)."""

    def __init__(self, specs):
        from pathlib import Path
        self.paths = []
        for s in ([specs] if isinstance(specs, (str, bytes)) else specs):
            self.paths += sorted(str(q) for q in Path(s).glob("*.png"))
        if not self.paths:
            raise ValueError("스프라이트가 없습니다")

    def get(self, rng) -> np.ndarray:
        s = cv2.imread(self.paths[rng.integers(len(self.paths))], cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(s, cv2.COLOR_BGRA2RGBA)


def sprite_source_id(path):
    """Original image identity; object class names may themselves contain underscores."""
    import re
    from pathlib import Path
    stem = Path(path).stem
    match = re.match(r"^(FD_\d+)(?:_|$)", stem)
    if match:
        return match.group(1)
    return stem.rsplit("_", 1)[0]


class MixedSpriteBank(SpriteBank):
    """v17: 여러 스프라이트 출처를 가중치로 섞는다. spec = [(폴더, 가중치, split)], split: all | train | val
    (val = 원본 영상 id 의 crc32 % 5 == 0. 506 파일명 'FD_102_00_Glass' → id 'FD_102')."""

    def __init__(self, spec):
        import zlib
        from pathlib import Path
        self.groups, w = [], []
        for folder, weight, split in spec:
            ps = sorted(str(q) for q in Path(folder).glob("*.png"))
            if split != "all":
                ps = [p for p in ps if (zlib.crc32(sprite_source_id(p).encode()) % 5 == 0) == (split == "val")]
            if ps:
                self.groups.append(ps)
                w.append(float(weight))
        if not self.groups:
            raise ValueError("스프라이트가 없습니다: %r" % (spec,))
        self.w = np.array(w) / sum(w)
        self.paths = [p for g in self.groups for p in g]

    def get(self, rng) -> np.ndarray:
        g = self.groups[rng.choice(len(self.groups), p=self.w)]
        s = cv2.imread(g[rng.integers(len(g))], cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(s, cv2.COLOR_BGRA2RGBA)


_CONF_BANKS = {}


def _confuser_bank(dirs):
    key = tuple(dirs)
    if key not in _CONF_BANKS:                                   # 워커 프로세스마다 한 번만 목록을 읽는다
        _CONF_BANKS[key] = MixedSpriteBank([(d, 1.0, "all") for d in dirs])
    return _CONF_BANKS[key]


class RealBank:
    """AI-Hub 실측 0.3 m 타일(+GT) — build_aihub_debris.py 의 real03/<split>/ (학습 분할만). 쓰레기를 포함하는 256 창을 뽑는다."""

    def __init__(self, specs):
        from pathlib import Path
        self.items = []
        for s in ([specs] if isinstance(specs, (str, bytes)) else specs):
            for q in sorted(Path(s).glob("*_img.png")):
                g = q.with_name(q.name.replace("_img.png", "_gt.png"))
                if g.exists():
                    self.items.append((str(q), str(g)))
        if not self.items:
            raise ValueError("실측 타일이 없습니다")

    def sample(self, rng):
        for _ in range(20):
            ip, gp = self.items[rng.integers(len(self.items))]
            img = cv2.cvtColor(cv2.imread(ip, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE) > 127
            if gt.any() and min(img.shape[:2]) >= H:
                ys, xs = np.nonzero(gt)
                j = rng.integers(len(ys))
                y = int(np.clip(ys[j] - rng.integers(16, H - 16), 0, img.shape[0] - H))
                x = int(np.clip(xs[j] - rng.integers(16, W - 16), 0, img.shape[1] - W))
                ci, cg = img[y:y + H, x:x + W], gt[y:y + H, x:x + W]
                k = rng.integers(4)
                ci, cg = np.rot90(ci, k), np.rot90(cg, k)
                if rng.random() < 0.5:
                    ci, cg = ci[:, ::-1], cg[:, ::-1]
                if cg.any():
                    return np.ascontiguousarray(ci), np.ascontiguousarray(cg)
        return None


# --------------------------------------------------------------------------- 하드 네거티브(정답 제외 대상)
def add_hard_negative(rng, img: np.ndarray, water: np.ndarray, cfg: Optional[SynthCfg] = None) -> np.ndarray:
    cfg = cfg or SynthCfg()
    f = img.astype(np.float32)
    kind = rng.choice(list(cfg.hardneg_kinds), p=list(cfg.hardneg_kinds.values()))
    land = ~water
    hi = np.percentile(f, 99.5)
    if kind == "foam" and water.mean() > 0.05:                  # 물가의 흰 선형 무늬(연속적·무채색)
        d = ndi.distance_transform_edt(water)
        line = np.exp(-((d - rng.uniform(2, 25)) / rng.uniform(1.0, 3.0)) ** 2) * (smooth_noise(rng, (H, W), 25) > 0.45)
        a = np.clip(line * rng.uniform(0.4, 0.95), 0, 1)[..., None]
        f = f * (1 - a) + a * hi * np.array([1.0, 1.0, 1.0])
    elif kind == "gravel" and land.mean() > 0.1:                # 밝은 자갈·조개 무더기(무채색 점, 넓고 고르게)
        region = (smooth_noise(rng, (H, W), 45) > 0.6) & land
        dots = (rng.random((H, W)) < rng.uniform(0.08, 0.35)) & region
        a = cv2.GaussianBlur(dots.astype(np.float32), (0, 0), rng.uniform(0.5, 0.9))[..., None] * rng.uniform(0.5, 1.0)
        tint = np.array([1.0, rng.uniform(0.95, 1.0), rng.uniform(0.85, 0.97)])
        if cfg.neg_tint > 0:                                    # v4: 약한 색조 허용 → 색만으로는 자갈/쓰레기를 못 가름
            tint = tint * (1 + rng.uniform(-cfg.neg_tint, cfg.neg_tint, 3))
        f = f * (1 - a) + a * hi * tint * rng.uniform(0.8, 1.0)
    elif kind == "wrack" and water.mean() > 0.02:               # 해조류·유목의 어두운 띠
        d = ndi.distance_transform_edt(land)
        band = (np.abs(d - rng.uniform(2, 20)) < rng.uniform(1, 4)) & (smooth_noise(rng, (H, W), 30) > 0.4)
        a = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 0.8)[..., None] * rng.uniform(0.4, 0.85)
        f = f * (1 - a) + a * np.array([50, 42, 30], np.float32) * rng.uniform(0.6, 1.3)
    elif kind == "shellband" and land.mean() > 0.05:            # v5: 해안선 쌍둥이 음성 — 쓰레기 띠와 같은 위치·모양을 '균일한' 흰 자갈·조개 알갱이로
        rho = accumulation_field(rng, cfg, water, tiny=False)
        if rho is not None:
            region = rho > 0
            grains = (rng.random((H, W)) < rng.uniform(0.3, 0.85)) & region
            a = cv2.GaussianBlur(grains.astype(np.float32), (0, 0), rng.uniform(0.5, 0.8)) * rng.uniform(0.7, 1.0)
            a = np.clip(a * 1.6, 0, 1)[..., None]
            tone = hi * rng.uniform(0.85, 1.0) * (1 + rng.uniform(-cfg.neg_tint, cfg.neg_tint, 3) if cfg.neg_tint > 0 else 1)
            f = f * (1 - a) + a * tone * (1 + rng.normal(0, 0.02, (H, W, 1)))   # 알갱이 밝기 편차 ±2% (쓰레기는 흰·회·검·색이 섞임)
    elif kind == "gen_confuser":                                # v17: SDXL 생성 혼동물을 쓰레기 자리(집적대 밀도장)에 — GT 없음
        rho = accumulation_field(rng, cfg, water, tiny=False) if cfg.confuser_dirs else None
        if rho is not None and rho.any():
            cov, pm = render_sprites(rng, cfg, rho, _confuser_bank(cfg.confuser_dirs))
            sig = rng.uniform(*cfg.psf_sigma)
            cov_b = cv2.GaussianBlur(cov, (0, 0), sig)[..., None]
            pm_b = cv2.GaussianBlur(pm, (0, 0), sig)
            f = f * (1 - cov_b) + pm_b * 255.0 * rng.uniform(0.9, 1.1)
    elif kind == "buoys" and water.mean() > 0.1:                # v4: 양식장 부표 줄(남해형) — 물 위 흰 점의 규칙 배열
        th, sp, rs = rng.uniform(0, np.pi), rng.uniform(3.0, 8.0), rng.uniform(4.0, 12.0)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        u = (xx - W / 2) * np.cos(th) + (yy - H / 2) * np.sin(th)
        v = -(xx - W / 2) * np.sin(th) + (yy - H / 2) * np.cos(th)
        grid = (np.abs(u - sp * np.round(u / sp)) < 0.6) & (np.abs(v - rs * np.round(v / rs)) < 0.6)
        farm = (smooth_noise(rng, (H, W), 60) > rng.uniform(0.35, 0.6)) & water
        dots = grid & farm & (rng.random((H, W)) < rng.uniform(0.6, 0.95))
        a = np.clip(cv2.GaussianBlur(dots.astype(np.float32), (0, 0), rng.uniform(0.5, 0.9)) * 2.0 * rng.uniform(0.6, 1.0), 0, 1)[..., None]
        f = f * (1 - a) + a * hi * np.array([1.0, 1.0, rng.uniform(0.92, 1.0)])
    elif kind == "scatter":                                     # v26(H2): 집적대를 이루지 않은 흩어진 조각 — 양성과 같은 조각·색·밝기·PSF 로 그리되 GT 없음
        n = int(rng.integers(cfg.scatter_n[0], cfg.scatter_n[1] + 1))   # (대회: "검출 단위는 한 점 한 점이 아니라 모여 이룬 덩어리와 띠") 육지·물 가리지 않음
        cand = np.argwhere(land) if (land.mean() > 0.05 and rng.random() < 0.8) else np.argwhere(np.ones((H, W), bool))
        if cfg.scatter_shore_p > 0 and water.mean() > 0.02 and land.mean() > 0.05 and rng.random() < cfg.scatter_shore_p:
            dl = ndi.distance_transform_edt(land)                     # v27: 해안선 따라 밀려온 선
            band = land & (dl > 2) & (dl < rng.uniform(8, 30))
            if band.sum() > 50:
                cand = np.argwhere(band)
        gap = rng.uniform(cfg.scatter_gap, cfg.scatter_gap_hi) if cfg.scatter_gap_hi > cfg.scatter_gap else float(cfg.scatter_gap)
        pts, gap2 = [], gap ** 2
        for _ in range(n * 25):
            if len(pts) >= n:
                break
            y, x = cand[rng.integers(len(cand))]
            if 2 <= y < H - 2 and 2 <= x < W - 2 and all((y - py) ** 2 + (x - px) ** 2 >= gap2 for py, px in pts):
                pts.append((int(y), int(x)))
        if len(pts) >= 2:
            rho = np.zeros((H, W), np.float32)
            for y, x in pts:
                rho[y, x] = rng.uniform(*cfg.scatter_cov)           # 자리당 피복 1~6 화소(v26; v27 은 설정) — 눈에 띄되 집적대는 아님
            rho = cv2.GaussianBlur(rho, (0, 0), 0.9)                # 조각이 자리 ±1~2 px 안에 떨어지게
            cov, pm = render_pieces(rng, cfg, rho, *zone_palette(rng, cfg))
            white = float(np.clip(np.percentile(f / 255.0, 99.0), 0.55, 1.0))
            shade = min(1.0, white * rng.uniform(*cfg.piece_shade))
            sig = rng.uniform(*cfg.psf_sigma)
            cov_b = cv2.GaussianBlur(cov, (0, 0), sig)[..., None]
            pm_b = cv2.GaussianBlur(pm, (0, 0), sig)
            f = f * (1 - cov_b) + pm_b * shade * 255.0
    else:                                                       # 물고임/사주: 부드러운 어두운·밝은 얼룩
        blob = cv2.GaussianBlur((smooth_noise(rng, (H, W), 35) > 0.62).astype(np.float32), (0, 0), 2.0)[..., None]
        f = f * (1 - 0.35 * blob) + 0.35 * blob * f.mean((0, 1)) * rng.uniform(0.5, 1.5)
    return np.clip(f, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- 장면 톤
def scene_look(rng, img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    f = img.astype(np.float32) / 255.0
    f = f ** np.exp(rng.normal(0, 0.22 * strength))                                    # 감마
    f = f * np.exp(rng.normal(0, 0.07 * strength, 3))                                  # 채널 이득(화이트밸런스)
    g = f.mean(-1, keepdims=True)
    f = g + (f - g) * np.exp(rng.normal(0, 0.25 * strength))                           # 채도
    if rng.random() < 0.35 * strength:                                                 # 누렇고 뿌연 연무
        hz = rng.uniform(0.05, 0.28)
        f = f * (1 - hz) + hz * np.array([0.78, 0.74, 0.62], np.float32)
    f = (f - 0.5) * np.exp(rng.normal(0, 0.12 * strength)) + 0.5 + rng.normal(0, 0.03 * strength)
    if rng.random() < 0.3:                                                             # 팬샤픈 풍 언샤프
        f = f + rng.uniform(0.3, 0.9) * (f - cv2.GaussianBlur(f, (0, 0), 1.0))
    return np.clip(f * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- v4: 색 번짐 · JPEG · 장면 위성 룩 · 불변성 증강
def chroma_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """색(Cr/Cb)만 번지게(D1): 팬샤픈(1.2 m 다중분광 색 → 0.3 m)·JPEG 4:2:0 흉내. 휘도는 그대로."""
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y[..., 1:] = cv2.GaussianBlur(y[..., 1:], (0, 0), float(sigma))
    return cv2.cvtColor(np.clip(y + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def chroma_shift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """색(Cr/Cb)만 (dy, dx) 화소 이동(서브픽셀 쌍선형, 반사 경계) — 팬샤픈 MS-pan 정합 오차 흉내(v15). 휘도는 그대로."""
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    mat = np.float32([[1, 0, dx], [0, 1, dy]])
    y[..., 1:] = cv2.warpAffine(np.ascontiguousarray(y[..., 1:]), mat, (img.shape[1], img.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.cvtColor(np.clip(y + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def jpeg_roundtrip(img: np.ndarray, q: int) -> np.ndarray:
    params = [cv2.IMWRITE_JPEG_QUALITY, int(q)]
    if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):          # 4:2:0 명시(기본값도 4:2:0)
        params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420]
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img[..., ::-1]), params)
    return np.ascontiguousarray(cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]) if ok else img


def sat_scene(rng, s: np.ndarray) -> np.ndarray:
    """장면(1024²) 단위 위성 룩(D2): 물을 어둡고 푸르게(평가셋 '아주 어두운 남색~검정 바다') + 2~98% 스트레치
    (하한 밴드별·이득 RGB 공통 — 자매 주제 1 의 8bit 변환 규약). 주최 측 학습 영상 통계(BN 역산 평균 RGB 48/59/64) 쪽으로 옮긴다."""
    small = cv2.resize(s, (256, 256), interpolation=cv2.INTER_AREA)
    wm = water_mask(small)
    f = s.astype(np.float32)
    if wm.any():
        wmf = cv2.resize(wm.astype(np.uint8), (s.shape[1], s.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        gain = rng.uniform(0.25, 0.6) * np.array([rng.uniform(0.7, 0.9), rng.uniform(0.85, 0.97), rng.uniform(1.05, 1.25)], np.float32)
        f[wmf] *= gain
    sm = cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA)
    lo = np.percentile(sm, 2, axis=(0, 1))
    hi = float(np.percentile(sm, 98, axis=(0, 1)).max())
    f = (f - lo) / max(1.0, hi - float(lo.min())) * 255.0
    return np.clip(f, 0, 255).astype(np.uint8)


def crop_contrast(rng, img: np.ndarray, cfg: SynthCfg) -> np.ndarray:
    """v5 대비 불변 증강: 크롭 단위 퍼센타일 스트레치(하한 밴드별·이득 공통) 또는 대비 증폭. 쓰레기가 합성된 뒤에 걸어
    양성·음성이 같은 변환을 받는다 → '대비가 센 질감 = 쓰레기' 지름길을 끊는다."""
    f = img.astype(np.float32)
    if cfg.crop_stretch_p > 0 and rng.random() < cfg.crop_stretch_p:
        lo = np.percentile(f, rng.uniform(0.5, 3.0), axis=(0, 1))
        hi = float(np.percentile(f, rng.uniform(97.0, 99.5), axis=(0, 1)).max())
        f = (f - lo) / max(1.0, hi - float(lo.min())) * 255.0
    elif cfg.contrast_p > 0 and rng.random() < cfg.contrast_p:
        m = f.mean((0, 1), keepdims=True)
        f = m + (f - m) * rng.uniform(1.2, 2.0)
    return np.clip(f, 0, 255).astype(np.uint8)


def invariance_aug(rng, img: np.ndarray, gray_p: float = 0.2, jitter_p: float = 0.5, hue: float = 0.05,
                   sat: Tuple[float, float] = (0.5, 1.3)) -> np.ndarray:
    """학습 전용 불변성 증강(v4 M2): 흑백·색조·채도 흔들기 → 색 점이 아니라 휘도 질감·위치·모양 단서를 쓰게 한다."""
    if rng.random() < gray_p:
        g = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        return np.repeat(np.clip(g + 0.5, 0, 255).astype(np.uint8)[..., None], 3, -1)
    if rng.random() < jitter_p:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)          # H: 0~180
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-hue, hue) * 180.0) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(*sat), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img


# --------------------------------------------------------------------------- 샘플 1장
def finish(rng, img: np.ndarray, cfg: SynthCfg) -> np.ndarray:
    """합성 뒤 공통 꼬리: 색 밀림·번짐(센서) → 장면 톤 → 잡음 → 대비 증강(8bit 변환) → JPEG(전달). make_sample 과 실측 양성이 함께 쓴다."""
    if cfg.chroma_shift_p > 0 and rng.random() < cfg.chroma_shift_p:   # v15: 색-밝기 어긋남(방향 무작위, GT 는 휘도=조각 위치 그대로)
        ang, mag = rng.uniform(0, 2 * np.pi), rng.uniform(*cfg.chroma_shift_px)
        img = chroma_shift(img, mag * np.sin(ang), mag * np.cos(ang))
    if cfg.chroma_p > 0 and rng.random() < cfg.chroma_p:     # v4 D1: 센서 단계에서 색 해상도 저하(배경·쓰레기 모두)
        img = chroma_blur(img, rng.uniform(*cfg.chroma_sigma))
    img = scene_look(rng, img)
    ns = rng.uniform(*cfg.noise_std)
    if ns > 0:
        img = np.clip(img.astype(np.float32) + rng.normal(0, ns, img.shape), 0, 255).astype(np.uint8)
    if cfg.crop_stretch_p > 0 or cfg.contrast_p > 0:          # v5: 대비 불변 증강(8bit 변환 단계)
        img = crop_contrast(rng, img, cfg)
    if cfg.jpeg_p > 0 and rng.random() < cfg.jpeg_p:         # v4 D1: 전달 단계 JPEG 4:2:0
        img = jpeg_roundtrip(img, int(rng.integers(cfg.jpeg_q[0], cfg.jpeg_q[1] + 1)))
    return img


def make_sample(rng, bg: np.ndarray, cfg: SynthCfg = SynthCfg(), force_pos: Optional[bool] = None, sprites=None):
    """bg: (256,256,3) uint8 실제 해안 배경. → (img uint8, mask bool, meta dict). sprites: v6 실제 조각 뱅크."""
    img = bg.copy()
    water = water_mask(img)
    if rng.random() < cfg.p_hardneg:
        img = add_hard_negative(rng, img, water, cfg)
    is_pos = (rng.random() < cfg.p_pos) if force_pos is None else force_pos
    mask = np.zeros((H, W), bool)
    n_acc = 0
    if is_pos:
        f = img.astype(np.float32) / 255.0
        white = float(np.percentile(f, 99.0))                   # 장면의 밝은 끝 = 흰 조각의 밝기
        white = float(np.clip(white, 0.55, 1.0))
        tiny = rng.random() < cfg.tiny_p
        for _ in range(1 if tiny else rng.integers(cfg.n_accum[0], cfg.n_accum[1] + 1)):
            rho = accumulation_field(rng, cfg, water, tiny)
            if rho is None:
                continue
            zone_shade = None
            if sprites is not None and cfg.sprite_p > 0 and rng.random() < cfg.sprite_p:
                cov, pm = render_sprites(rng, cfg, rho, sprites)       # v6: 실제 쓰레기 조각
                zone_shade = rng.uniform(0.9, 1.1)                    # 실제 밝기 유지(장면 흰색 기준으로 다시 곱하면 실측보다 어두워짐)
            else:
                cov, pm = render_pieces(rng, cfg, rho, *zone_palette(rng, cfg))
            if zone_shade is None:
                shade = white * rng.uniform(*cfg.piece_shade)
                if tuple(cfg.piece_shade) != (0.85, 1.05):          # v6 만 상한 1.0 (이전 레시피 출력은 그대로 유지)
                    shade = min(1.0, shade)
            else:
                shade = zone_shade
            sig = rng.uniform(*cfg.psf_sigma)
            cov_b = cv2.GaussianBlur(cov, (0, 0), sig)[..., None]
            pm_b = cv2.GaussianBlur(pm, (0, 0), sig)
            f = f * (1 - cov_b) + pm_b * shade
            mask |= cov >= cfg.cov_thr
            n_acc += 1
        img = np.clip(f * 255.0, 0, 255).astype(np.uint8)
    img = finish(rng, img, cfg)
    return img, mask, dict(pos=bool(mask.any()), n_acc=n_acc, area=int(mask.sum()))


def draw(rng, bank, cfg: SynthCfg = SynthCfg(), mined=None, sprites=None, real=None):
    """양성률을 cfg.p_pos 로 '정확히' 맞춰 뽑는다. 전부 바다인 배경처럼 양성을 만들 수 없는 크롭은 다시 뽑는다.
    (존재 판정 임계는 양성률에 민감하므로 학습·검증의 양성률이 평가셋과 어긋나면 임계가 틀어진다)
    mined: 실배경 헛경보 채굴 뱅크(v4 D4). 음성 표본의 cfg.mined_p 만큼을 여기서 뽑는다."""
    want = rng.random() < cfg.p_pos
    if not want and mined is not None and cfg.mined_p > 0 and rng.random() < cfg.mined_p:
        return make_sample(rng, mined.crop(rng), cfg, force_pos=False)
    if want and real is not None and cfg.real_p > 0 and rng.random() < cfg.real_p:   # v6: 실측 양성
        r = real.sample(rng)
        if r is not None:
            img, m = r
            return finish(rng, img, cfg), m, dict(pos=True, n_acc=-1, area=int(m.sum()))
    for _ in range(8):
        img, m, meta = make_sample(rng, bank.crop(rng), cfg, force_pos=want, sprites=sprites)
        if bool(m.any()) == want:
            break
    return img, m, meta


# --------------------------------------------------------------------------- 배경 뱅크
def background_paths(specs) -> list:
    """배경 지정(폴더 또는 경로 목록 .txt, 여러 개 가능) → 장면 파일 경로 목록.
    .txt 의 상대경로는 그 .txt 가 있는 폴더 기준이다."""
    from pathlib import Path
    out = []
    for spec in ([specs] if isinstance(specs, (str, bytes)) or not hasattr(specs, "__iter__") else specs):
        sp = Path(spec)
        if sp.suffix == ".txt":
            out += [(sp.parent / l.strip()).resolve() for l in sp.read_text(encoding="utf-8").splitlines() if l.strip()]
        else:
            out += sorted(sp.glob("*.png")) + sorted(sp.glob("*.jpg"))
    return out


class BackgroundBank:
    """큰 장면(PNG/JPG) 경로를 들고 있다가, 크롭할 때마다 한 장씩 읽어 256 크롭을 뽑는다.
    (v1 은 전 장면을 워커마다 메모리에 올렸다 → 장면 수천 장이면 워커 수 x 수 GB. v2 부터 지연 로딩)
    장면 단위로 train/val 을 가른다(누수 방지)."""

    def __init__(self, paths, sat_p: float = 0.0, center_p: float = 0.0):
        self.paths = [str(p) for p in paths]
        self.sat_p, self.center_p = float(sat_p), float(center_p)   # v4 D2 (기본 0 = v1 동작)
        if not self.paths:
            raise ValueError("배경 장면이 없습니다")

    def _load(self, rng) -> np.ndarray:
        for _ in range(20):
            s = cv2.imread(self.paths[rng.integers(len(self.paths))], cv2.IMREAD_COLOR)
            if s is not None and min(s.shape[:2]) >= 300:
                s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
                if self.sat_p > 0 and rng.random() < self.sat_p:
                    s = sat_scene(rng, s)
                return s
        raise RuntimeError("읽을 수 있는 배경 장면이 없습니다")

    def crop(self, rng) -> np.ndarray:
        s = self._load(rng)
        z = rng.uniform(0.8, 1.25)                              # GSD 흔들기(0.24~0.375 m 상당)
        k = int(round(256 * z))
        if self.center_p > 0 and rng.random() < self.center_p:  # 장면 중앙(해안선 좌표) 쪽 → 물·뭍이 함께 든 크롭 비중↑
            cy = float(np.clip(rng.normal(s.shape[0] / 2, s.shape[0] * 0.15), k / 2, s.shape[0] - k / 2))
            cx = float(np.clip(rng.normal(s.shape[1] / 2, s.shape[1] * 0.15), k / 2, s.shape[1] - k / 2))
            y, x = int(cy - k / 2), int(cx - k / 2)
        else:
            y, x = rng.integers(0, s.shape[0] - k + 1), rng.integers(0, s.shape[1] - k + 1)
        c = s[y:y + k, x:x + k]
        c = cv2.resize(c, (W, H), interpolation=cv2.INTER_AREA if z > 1 else cv2.INTER_LINEAR)
        c = np.rot90(c, rng.integers(4))
        if rng.random() < 0.5:
            c = c[:, ::-1]
        return np.ascontiguousarray(c)


class MinedBank:
    """실배경 헛경보 채굴 크롭(256², 전부 음성) — v4 D4. mine_hardneg.py 가 만든 폴더(들)를 읽고 회전·반전만 한다."""

    def __init__(self, specs):
        self.paths = [str(p) for p in background_paths(specs)]
        if not self.paths:
            raise ValueError("채굴 음성 크롭이 없습니다")

    def crop(self, rng) -> np.ndarray:
        for _ in range(20):
            s = cv2.imread(self.paths[rng.integers(len(self.paths))], cv2.IMREAD_COLOR)
            if s is not None:
                s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
                if s.shape[:2] != (H, W):
                    s = cv2.resize(s, (W, H), interpolation=cv2.INTER_AREA)
                s = np.rot90(s, rng.integers(4))
                if rng.random() < 0.5:
                    s = s[:, ::-1]
                return np.ascontiguousarray(s)
        raise RuntimeError("읽을 수 있는 채굴 크롭이 없습니다")


if __name__ == "__main__":                                       # 눈으로 보는 점검용 시트
    import sys
    from pathlib import Path
    bg_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    bank = BackgroundBank(background_paths([str(bg_dir)]))
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        tiles = []
        for j in range(6):
            img, m, meta = make_sample(rng, bank.crop(rng), force_pos=(j % 3 != 2))
            ov = img.copy()
            edge = m & ~ndi.binary_erosion(m)
            ov[edge] = (255, 0, 255)
            tiles += [img, ov]
        rows.append(np.concatenate(tiles, 1))
    sheet = np.concatenate(rows, 0)
    cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("saved", out, sheet.shape)
    zooms = []                                                  # 집적대 주변 96x96 을 4배 확대(좌: 영상, 우: GT)
    while len(zooms) < 24:
        img, m, meta = make_sample(rng, bank.crop(rng), force_pos=True)
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        cy, cx = int(np.clip(np.median(ys), 48, H - 48)), int(np.clip(np.median(xs), 48, W - 48))
        a = img[cy - 48:cy + 48, cx - 48:cx + 48]
        b = np.repeat((m[cy - 48:cy + 48, cx - 48:cx + 48] * 255).astype(np.uint8)[..., None], 3, -1)
        zooms.append(cv2.resize(np.concatenate([a, b], 1), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
    z = np.concatenate([np.concatenate(zooms[i * 4:(i + 1) * 4], 1) for i in range(6)], 0)
    cv2.imwrite(str(out).replace(".jpg", "_zoom.jpg"), cv2.cvtColor(z, cv2.COLOR_RGB2BGR))
