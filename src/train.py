#!/usr/bin/env python
"""train.py — 합성 데이터만으로 학습하는 새 베이스라인 (UNet + 존재 판정 보조 헤드, 입력 2배 확대)

    python train.py --bg_dir <배경 장면 폴더 또는 경로목록.txt ...> --out runs/v1 --encoder resnet34 --steps 20000
    (중단됐으면 같은 명령에 --resume 을 붙이면 out/last.pt 부터 이어서 학습)
    python train.py --recipe v4 --mined <채굴 음성 폴더> ...     # v4 (docs/v4_병목분석_개선계획.md)
    python train.py --recipe v5 --init runs/v4_r34/weights_fp16.pt --mined <v2 채굴> <v4 채굴> --steps 8000 ...   # v5 (§4)

레시피
  v1 (기본)  v1/v2 와 같은 합성·검증·손실
  v4         D1~D4 합성(synth.cfg_v4: 색 지름길 끊기·위성 룩·집적대 면적·채굴 음성) + M1 강건성 검증(valmix) +
             M2 불변성 증강(흑백·색조·채도) + M4 지표 맞춤 손실(1 px 고리 무시 BCE + 허용-F 대리). M6 색 정규화는 --input_chroma_sigma
  v5         v4 + 대비 불변 증강(크롭 스트레치·대비 증폭) + 해안선 쌍둥이 음성(shellband) + 양성 이질성 보강 + v5 강건성 검증
  v6         v5 + AI-Hub 71363 실측 쓰레기(--sprites 조각 이식 35%, --real 실측 양성 10%). 학습 분할(40곳) 자산만 쓸 것

핵심 설계
  * 데이터: synth.make_sample 로 on-the-fly 생성(무한). 배경 '장면' 단위로 train/val 분리.
  * 모델: smp.Unet(encoder, classes=1, aux_params=분류 헤드). 입력을 2배 확대해 1~3 화소 구조를 살리고,
          추론 때는 확률을 2x2 평균으로 원해상도에 되돌린다(노트북의 Net 과 state_dict 키가 같다).
  * 손실: BCE + Dice(분할) + 0.5·BCE(존재). 존재 라벨 = GT 에 1 화소라도 있으면 양성(지표 정의 그대로).
  * 선택: 고정 합성 검증셋에서 '대회 지표 그대로' 점수를 계산하고, mask_thr·presence_thr 를 격자 탐색해
          최고 점수 체크포인트와 그 임계값을 config.json 에 기록한다.
"""
import argparse
import ast
import json
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

import metric as M
from chroma import canon_chroma
from postproc import decide, inflate_to_min_area
from synth import (BackgroundBank, MinedBank, MixedSpriteBank, RealBank, SpriteBank, SynthCfg, background_paths, cfg_v4, cfg_v5, cfg_v6, cfg_v17, cfg_v26, cfg_v27, cfg_v28, make_pair, draw,
                   invariance_aug, V3_PALETTE, V5_PALETTE)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Net(torch.nn.Module):
    def __init__(self, arch, encoder, upscale=2, encoder_weights="imagenet", aux=True, vfm_tokens=32, vfm_rein_pos=0, vfm_anyup=""):
        super().__init__()
        if arch == "VFM":                                   # 얼린 기반모델 + Rein 어댑터 + 고해상 디코더(vfm_net.py). encoder = timm 이름
            from vfm_net import VFMBody
            self.body = VFMBody(encoder, pretrained=encoder_weights is not None, tokens=vfm_tokens, rein_pos=vfm_rein_pos,
                                anyup=bool(vfm_anyup), anyup_ckpt=vfm_anyup if vfm_anyup and vfm_anyup != "1" else None)
        else:
            self.body = getattr(smp, arch)(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=1,
                                           aux_params=dict(classes=1, dropout=0.2) if aux else None)
        self.up = upscale

    def forward(self, x):                                   # 학습용: 확대 해상도의 logit 을 그대로 돌려준다
        if self.up > 1:
            x = F.interpolate(x, scale_factor=self.up, mode="bilinear", align_corners=False)
        out = self.body(x)
        return out if isinstance(out, tuple) else (out, None)


class SynthStream(torch.utils.data.IterableDataset):
    """무한 합성 스트림. 샘플의 난수를 (seed, 전역 배치 번호, 배치 내 위치)로 정한다 →
    워커 수를 바꾸거나 중간에 끊고 --resume 해도 step 마다 '같은 배치'가 나온다(재현성).
    워커 w 는 배치 start+w, start+w+W, ... 를 만들고, DataLoader 가 워커를 순서대로 돌며 꺼내므로 전역 순서가 유지된다."""

    def __init__(self, paths, cfg, seed, batch, start_batch=0, recipe="v1", mined=None, chroma_sigma=0.0, sprites=None, real=None):
        self.paths, self.cfg, self.seed, self.batch, self.start = paths, cfg, seed, batch, start_batch
        self.recipe, self.mined, self.chroma_sigma, self.sprites, self.real = recipe, mined, chroma_sigma, sprites, real

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        bank = BackgroundBank(self.paths, sat_p=self.cfg.sat_p, center_p=self.cfg.center_p)
        mined = MinedBank(self.mined) if self.mined else None
        sprites = (MixedSpriteBank(self.sprites) if self.sprites and isinstance(self.sprites[0], (tuple, list))
                   else SpriteBank(self.sprites) if self.sprites else None)
        real = RealBank(self.real) if self.real else None
        b = self.start + wid
        while True:
            pend = None
            for i in range(self.batch):
                rng = np.random.default_rng([self.seed, b, i])
                if pend is not None:                                # v27: 짝의 음성 쪽을 같은 배치 바로 다음 칸에
                    (img, m), pend = pend, None
                else:
                    pair = None
                    if self.cfg.pair_p > 0 and i < self.batch - 1 and rng.random() < self.cfg.pair_p:
                        pair = make_pair(rng, bank.crop(rng), self.cfg)
                    if pair is not None:
                        (img, m), pend = pair
                    else:
                        img, m, _ = draw(rng, bank, self.cfg, mined, sprites, real)
                if self.recipe in ("v4", "v5", "v6", "v17", "v26", "v27", "v28"):
                    img = invariance_aug(rng, img)                      # M2
                img = canon_chroma(img, self.chroma_sigma)              # M6 (σ=0 이면 그대로)
                x = ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
                yield torch.from_numpy(x), torch.from_numpy(m[None].astype(np.float32))
            b += nw


def make_val(paths, cfg, n, seed):
    rng, bank = np.random.default_rng(seed), BackgroundBank(paths)
    xs, ms = [], []
    for _ in range(n):
        img, m, _ = draw(rng, bank, cfg)
        xs.append(img); ms.append(m)
    return np.stack(xs), np.stack(ms)


@torch.no_grad()
def predict(net, imgs, bs=64):
    net.eval()
    probs, press = [], []
    for s in range(0, len(imgs), bs):
        x = torch.from_numpy(((imgs[s:s + bs].astype(np.float32) / 255.0 - MEAN) / STD).transpose(0, 3, 1, 2)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16 if hasattr(net.body, "vfm") else torch.float16):
            logit, cls = net(x)
        p = torch.sigmoid(logit.float())
        if net.up > 1:
            p = F.avg_pool2d(p, net.up)
        probs.append(p[:, 0].cpu().numpy()); press.append(torch.sigmoid(cls.float()).flatten().cpu().numpy())
    net.train()
    return np.concatenate(probs), np.concatenate(press)


MASK_GRID = {"v1": (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6),
             "v4": (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)}      # M1: v1·v2 는 하한 0.3 에 붙어 있었음
MASK_GRID["v5"] = MASK_GRID["v6"] = MASK_GRID["v17"] = MASK_GRID["v26"] = MASK_GRID["v27"] = MASK_GRID["v28"] = MASK_GRID["v4"]
PRES_GRID_V5 = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7)
PRES_GRID = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)


def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d > 0 else 0.0


def eval_at(probs, press, gts, mt, pt, inflate_to=56):
    """postproc.decide 와 같은 결정으로 대회 점수. 존재 판정이 양성이면 마스크(50 px 미만은 inflate), 음성이면 빈 마스크."""
    gt_pos = gts.reshape(len(gts), -1).any(1)
    fi = np.zeros(len(gts))
    for i in np.flatnonzero(gt_pos):
        mk = probs[i] >= mt
        if int(mk.sum()) < M.MIN_POS_PX:
            mk = inflate_to_min_area(mk, probs[i], inflate_to)
        fi[i] = M.patch_prf(mk, gts[i], M.MIN_POS_PX)[2]
    pp = press >= pt
    tp, fp = int((pp & gt_pos).sum()), int((pp & ~gt_pos).sum())
    fn, tn = int((~pp & gt_pos).sum()), int((~pp & ~gt_pos).sum())
    f1 = 0.5 * (_f1(tp, fp, fn) + _f1(tn, fn, fp))
    shape = float(fi[pp & gt_pos].sum()) / max(1, int(gt_pos.sum()))
    return dict(score=0.5 * f1 + 0.5 * shape, f1=f1, shape=shape, tp=tp, fp=fp, fn=fn, tn=tn), fi


def tune(probs, press, gts, mask_grid=MASK_GRID["v1"], pres_grid=PRES_GRID):
    """대회 지표를 직접 최대화하는 (mask_thr, presence_thr) 격자 탐색.
    형상 F 는 존재 임계와 무관하므로 mask_thr 마다 한 번만 계산한다(이전 구현과 결과 동일, 격자 수만큼 빠름)."""
    gt_pos = gts.reshape(len(gts), -1).any(1)
    n_pos = max(1, int(gt_pos.sum()))
    best = dict(score=-1)
    for mt in mask_grid:
        _, fi = eval_at(probs, press, gts, mt, 2.0)                 # 형상 F 만 필요(pt 무관)
        for pt in pres_grid:
            pp = press >= pt
            tp, fp = int((pp & gt_pos).sum()), int((pp & ~gt_pos).sum())
            fn, tn = int((~pp & gt_pos).sum()), int((~pp & ~gt_pos).sum())
            f1 = 0.5 * (_f1(tp, fp, fn) + _f1(tn, fn, fp))
            shape = float(fi[pp & gt_pos].sum()) / n_pos
            score = 0.5 * f1 + 0.5 * shape
            if score > best["score"]:
                best = dict(score=float(score), f1=float(f1), shape=float(shape), mask_thr=mt, presence_thr=pt)
    return best


def split_scenes(paths, seed):
    """장면 단위 train/val 분할(검증 20%). mine_hardneg.py 도 같은 분할을 써서 검증 장면을 채굴에서 뺀다."""
    rs = np.random.default_rng(seed).permutation(len(paths))
    n_val = max(2, len(paths) // 5)
    return [paths[i] for i in rs[n_val:]], [paths[i] for i in rs[:n_val]]


def parse_synth(items):
    """--synth key=value 목록 → SynthCfg 덮어쓰기 dict. 없는 키는 바로 멈춘다."""
    names, out = {f.name for f in fields(SynthCfg)}, {}
    for it in items:
        k, _, v = it.partition("=")
        if k not in names:
            raise SystemExit(f"--synth: SynthCfg 에 없는 키 {k}")
        out[k] = {"V3": V3_PALETTE, "V5": V5_PALETTE}.get(v) or ast.literal_eval(v)
        if k == "palette":
            out[k] = dict(out[k])
    return out


def tolerant_f_loss(prob, gt, eps=1e-6, beta=1.0):
    """대회 형상 점수(1 px 허용 F)의 미분가능 대리(M4). prob·gt: (B,1,H,W) 원해상도. 정답 양성 영상만 평균.
    beta < 1 이면 F_β 로 정밀도에 더 무게(넘치게 칠하는 것을 더 감점). beta = 1 은 기존과 같다."""
    d_gt, d_pr = F.max_pool2d(gt, 3, 1, 1), F.max_pool2d(prob, 3, 1, 1)
    P = (prob * d_gt).sum((1, 2, 3)) / (prob.sum((1, 2, 3)) + eps)
    R = (gt * d_pr).sum((1, 2, 3)) / (gt.sum((1, 2, 3)) + eps)
    b2 = beta * beta
    f = (1 + b2) * P * R / (b2 * P + R + eps)
    pos = gt.sum((1, 2, 3)) > 0
    return (1 - f[pos]).mean() if pos.any() else prob.sum() * 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg_dir", required=True, nargs="+", help="폴더 또는 경로 목록 .txt (여러 개 가능)"); ap.add_argument("--out", required=True)
    ap.add_argument("--pos_weight", type=float, default=3.0, help="마스크 BCE 양성 화소 가중치(기존 3). 낮추면 덜 넓게 칠한다")
    ap.add_argument("--pres_pos_weight", type=float, default=1.0, help="v27: 존재 BCE 양성 가중(정탐 1 = 오탐 1.8 값)")
    ap.add_argument("--distill", default="", help="v27: 존재 증류 교사 가중치(같은 구조, 예: runs/v4_r34/weights_fp16.pt)")
    ap.add_argument("--distill_w", type=float, default=1.0)
    ap.add_argument("--tolf_beta", type=float, default=1.0, help="1 px 허용 F 대리의 β(기존 1). <1 이면 정밀도에 더 무게")
    ap.add_argument("--vfm_rein_pos", type=int, default=0, help="v22: Rein 원본의 위치별 선형 항 저랭크(0=끔)")
    ap.add_argument("--vfm_anyup", default="", help="v22: AnyUp 사전학습 가중치 경로(빈칸=끔)")
    ap.add_argument("--vfm_tokens", type=int, default=32, help="VFM 백본 토큰 격자(32 → 512 px, 48 → 768 px 입력)")
    ap.add_argument("--arch", default="Unet"); ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--upscale", type=int, default=2); ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=24); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--val_n", type=int, default=600)
    ap.add_argument("--eval_every", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="out/last.pt 가 있으면 그 지점부터 이어서 학습")
    ap.add_argument("--ckpt_every", type=int, default=500, help="재개용 체크포인트(last.pt) 저장 간격(step)")
    ap.add_argument("--recipe", default="v1", choices=["v1", "v4", "v5", "v6", "v17", "v26", "v27", "v28"], help="v1: 기존 / v4: D1~D4 + M1·M2·M4 / v5: v4 + §4 / v6: v5 + AI-Hub 실측")
    ap.add_argument("--sprites", nargs="*", default=None, help="v6: AI-Hub 쓰레기 조각 폴더(derived/sprites/train)")
    ap.add_argument("--real", nargs="*", default=None, help="v6: AI-Hub 실측 0.3 m 타일 폴더(derived/real03/train)")
    ap.add_argument("--init_partial", action="store_true", help="--init 에서 이름·모양이 같은 가중치만 불러오고 새 모듈은 새로 초기화")
    ap.add_argument("--init", default=None, help="이 가중치(weights_fp16.pt)에서 이어 학습(새 LR 스케줄). --resume 과 다름")
    ap.add_argument("--save_every_eval", action="store_true", help="검증 때마다 weights_step<N>.pt 저장(사후에 실측으로 체크포인트 고르기)")
    ap.add_argument("--ema", type=float, default=0.0, help="가중치 지수이동평균 감쇠(예: 0.999). >0 이면 검증·저장·선택은 EMA 가중치로")
    ap.add_argument("--mined", nargs="*", default=None, help="실배경 헛경보 채굴 폴더(들) — v4 D4 (mine_hardneg.py 산출)")
    ap.add_argument("--input_chroma_sigma", type=float, default=0.0, help="M6 입력 색 정규화 σ(학습·추론 동일 적용, 0=끔)")
    ap.add_argument("--bg_extra", nargs="*", default=None, help="학습에만 더하는 배경(폴더·목록 .txt). 검증 분할은 --bg_dir 로만 정해진다")
    ap.add_argument("--split_seed", type=int, default=None, help="장면 train/val 분할 시드(기본 = --seed). 시드만 바꾼 비교는 0 으로 고정")
    ap.add_argument("--synth", nargs="*", default=None,
                    help="레시피 합성 설정 덮어쓰기 key=value (값은 파이썬 리터럴, palette 는 V3|V5). 예: p_pos=0.5 clump_r='(7,32)'")
    ap.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay (v17 권장 0.05)")
    ap.add_argument("--label_smooth", type=float, default=0.0, help="존재 판정 BCE 라벨 스무딩(v17 권장 0.05)")
    ap.add_argument("--freeze_enc_steps", type=int, default=0, help="처음 N step 인코더 고정(LR 0) — ImageNet 특징 보존(v17 권장 2000)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    # 경로 점검: 틀린 경로(예: 비어 있는 셸 변수로 '/sprites/train')를 검증셋 생성 전에 바로 알린다
    for flag, specs in (("--bg_dir", a.bg_dir), ("--bg_extra", a.bg_extra or []), ("--mined", a.mined or []), ("--sprites", a.sprites or []), ("--real", a.real or [])):
        for sp in specs:
            n = len(background_paths([sp])) if flag != "--real" else len(list(Path(sp).glob("*_img.png")))
            if n == 0:
                raise SystemExit(f"{flag} 경로에 파일이 없습니다: {sp} (현재 폴더 {Path.cwd()})")
    n_sprites = sum(len(list(Path(s).glob("*.png"))) for s in (a.sprites or []))
    n_real = sum(len(list(Path(s).glob("*_img.png"))) for s in (a.real or []))
    paths = background_paths(a.bg_dir)
    split_seed = a.seed if a.split_seed is None else a.split_seed
    tr_paths, val_paths = split_scenes(paths, split_seed)
    if a.bg_extra:
        tr_paths = tr_paths + background_paths(a.bg_extra)
    cfg = {"v4": cfg_v4, "v5": cfg_v5, "v6": cfg_v6, "v17": cfg_v17, "v26": cfg_v26, "v27": cfg_v27, "v28": cfg_v28}.get(a.recipe, SynthCfg)()
    if a.synth:
        cfg = replace(cfg, **parse_synth(a.synth))
        print("SYNTH override", a.synth, flush=True)
    if a.recipe == "v28":                                       # v28: 실제 조각(생성·AI-Hub 506·71363 학습 분할) + 한국 실측 양성 타일(학습 분할)
        D = Path(__file__).resolve().parents[1] / "data"
        a.sprites = a.sprites or [(str(D / "gen_sprites/train"), 25, "all"), (str(D / "aihub_506_sample/derived/sprites"), 20, "train"),
                                  (str(D / "aihub_71363/derived/sprites/train"), 15, "all")]
        a.real = a.real or [str(D / "aihub_71363/derived/real03/train")]
    if a.recipe == "v6" and not (a.sprites and a.real):
        raise SystemExit("v6 는 --sprites 와 --real 이 필요합니다(AI-Hub 학습 분할 자산)")
    if a.recipe == "v17":                                       # 렌더러 혼합 + 학습에 안 쓴 스프라이트·혼동물·장면으로 만든 검증셋(LORO)
        D = Path(__file__).resolve().parents[1] / "data"
        a.sprites = [(str(D / "gen_sprites/train"), 25, "all"), (str(D / "aihub_506_sample/derived/sprites"), 20, "train"),
                     (str(D / "aihub_71363/derived/sprites/train"), 15, "all")]
        vspec = [(str(D / "gen_sprites/val"), 25, "all"), (str(D / "aihub_506_sample/derived/sprites"), 20, "val"),
                 (str(D / "aihub_71363/derived/sprites/train"), 15, "all")]
        vcfg = replace(cfg, confuser_dirs=(str(D / "gen_confusers/val"),))
        vrng, vbank, vspr = np.random.default_rng(12345), BackgroundBank(val_paths, sat_p=cfg.sat_p, center_p=cfg.center_p), MixedSpriteBank(vspec)
        xs, ms = [], []
        for _ in range(a.val_n):
            img, m, _ = draw(vrng, vbank, vcfg, None, vspr)
            xs.append(img); ms.append(m)
        vx, vm, vg = np.stack(xs), np.stack(ms), np.array(["loro"] * a.val_n)
    elif a.recipe in ("v4", "v5", "v6"):                          # M1 강건성 검증셋 (v6 은 v5 구성)
        import valmix
        vx, vm, vg = valmix.make_robust_val(val_paths, a.val_n, seed=12345, recipe="v5" if a.recipe == "v6" else a.recipe)
    else:
        vx, vm = make_val(val_paths, cfg, a.val_n, seed=12345)
        vg = np.array(["base"] * len(vx))
    if a.input_chroma_sigma > 0:
        vx = np.stack([canon_chroma(im, a.input_chroma_sigma) for im in vx])
    mask_grid = MASK_GRID[a.recipe]
    n_mined = len(background_paths(a.mined)) if a.mined else 0
    print(f"recipe {a.recipe} | scenes train {len(tr_paths)} / val {len(val_paths)} | val {len(vx)} pos rate {vm.any((1, 2)).mean():.3f} "
          f"groups {dict(zip(*np.unique(vg, return_counts=True)))} | mined negatives {n_mined} | sprites {n_sprites} | real tiles {n_real} "
          f"| input_chroma_sigma {a.input_chroma_sigma}", flush=True)

    net = Net(a.arch, a.encoder, a.upscale, encoder_weights=None if a.init else "imagenet", vfm_tokens=a.vfm_tokens,
              vfm_rein_pos=a.vfm_rein_pos, vfm_anyup=a.vfm_anyup).cuda()
    if a.init:
        sd = torch.load(a.init, map_location="cpu", weights_only=True)
        sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
        own = net.state_dict()
        if "body.grid_tokens" in own:                          # v22: 격자를 바꿔 이어 학습(점진 해상도) — 표식은 새 모델 것을 쓴다
            sd["body.grid_tokens"] = own["body.grid_tokens"]
        else:
            sd.pop("body.grid_tokens", None)
        if a.init_partial:                                     # v22: 새 모듈(위치별 어댑터·AnyUp 가지)은 새로 초기화, 나머지는 불러옴
            keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
            new_k = [k for k in own if k not in keep]; drop = [k for k in sd if k not in keep]
            net.load_state_dict(keep, strict=False)
            print(f"INIT(partial) from {a.init}: 불러옴 {len(keep)} / 새로 초기화 {len(new_k)} {sorted({k.split('.')[1] + '.' + k.split('.')[2] for k in new_k})[:6]} / 버림 {len(drop)}", flush=True)
        else:
            net.load_state_dict(sd)
            print(f"INIT from {a.init}", flush=True)
    is_enc = lambda n: ".encoder." in n or ".rein." in n                # VFM: 얼린 백본은 제외, 어댑터가 인코더 몫
    enc = [p for n, p in net.named_parameters() if p.requires_grad and is_enc(n)]
    rest = [p for n, p in net.named_parameters() if p.requires_grad and not is_enc(n)]
    print(f"학습 파라미터 {sum(p.numel() for p in enc + rest) / 1e6:.1f}M / 전체 {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW([dict(params=enc, lr=a.lr * 0.3), dict(params=rest, lr=a.lr)], weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr * 0.3, a.lr], total_steps=a.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")
    best, log, start = dict(score=-1), [], 0
    ck_path = out / "last.pt"
    ema_state = None
    if a.resume and ck_path.exists():
        ck = torch.load(ck_path, map_location="cuda", weights_only=False)
        if ck["steps"] != a.steps:
            raise SystemExit(f"--steps 가 체크포인트({ck['steps']})와 다릅니다. 같은 값으로 재개하세요(LR 스케줄이 steps 에 묶여 있음).")
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        if ck.get("ema") is not None:
            ema_state = ck["ema"]
        best, log, start = ck["best"], ck["log"], ck["step"]
        print(f"RESUME from step {start} (best {best['score']:.4f})", flush=True)
    elif a.resume:
        print("RESUME: last.pt 없음 → 처음부터 시작", flush=True)

    # EMA(선택): 학습 가중치의 지수이동평균 사본. BN 통계는 학습 모델 것을 그대로 복사한다.
    ema = None
    if a.ema > 0:
        import copy
        ema = copy.deepcopy(net).eval()
        for q in ema.parameters():
            q.requires_grad_(False)
        if ema_state is not None:
            ema.load_state_dict(ema_state)

    def ema_update(step):
        d = min(a.ema, (1 + step) / (10 + step))              # 초반엔 빨리 따라가게(워밍업)
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(d).add_(pn.detach(), alpha=1 - d)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)

    def eval_model():
        return ema if ema is not None else net

    def save_ckpt(step):                                    # 재개용(fp32 전체 상태). 쓰는 도중 죽어도 기존 파일이 안 깨지게 임시파일 → 교체
        tmp = out / "last.pt.tmp"
        torch.save(dict(model=net.state_dict(), ema=(ema.state_dict() if ema is not None else None), opt=opt.state_dict(),
                        sched=sched.state_dict(), scaler=scaler.state_dict(),
                        step=step, steps=a.steps, best=best, log=log), tmp)
        tmp.replace(ck_path)

    teacher = None
    if a.distill:
        teacher = Net(a.arch, a.encoder, a.upscale, encoder_weights=None).cuda().eval()
        tsd = torch.load(a.distill, map_location="cpu", weights_only=True)
        teacher.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in tsd.items()})
        for q in teacher.parameters():
            q.requires_grad_(False)
        print(f"DISTILL teacher {a.distill} w {a.distill_w}", flush=True)
    # 배치 번호 = step-1 → 재개하면 끊긴 바로 다음 배치부터 같은 데이터가 이어진다(워커 수와 무관)
    dl = torch.utils.data.DataLoader(SynthStream(tr_paths, cfg, a.seed, a.batch, start_batch=start, recipe=a.recipe,
                                                 mined=a.mined, chroma_sigma=a.input_chroma_sigma, sprites=a.sprites, real=a.real),
                                     batch_size=a.batch,
                                     num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)   # 기본이 워커 순서대로 꺼냄
    t0 = time.time()
    net.train()
    for step, (x, m) in enumerate(dl, start + 1):
        if step > a.steps:
            break
        x, m = x.cuda(non_blocking=True), m.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16 if hasattr(net.body, "vfm") else torch.float16):
            logit, cls = net(x)
        logit, cls = logit.float(), cls.float().flatten()
        mu = F.interpolate(m, scale_factor=a.upscale, mode="nearest") if a.upscale > 1 else m
        pw = torch.tensor(a.pos_weight, device=logit.device)
        p = torch.sigmoid(logit)
        inter, den = (p * mu).sum((1, 2, 3)), p.sum((1, 2, 3)) + mu.sum((1, 2, 3))
        dice = (1 - (2 * inter + 1.0) / (den + 1.0)).mean()
        ptgt = (m.amax((1, 2, 3)) > 0).float()
        if a.label_smooth > 0:
            ptgt = ptgt * (1 - a.label_smooth) + 0.5 * a.label_smooth
        pres = F.binary_cross_entropy_with_logits(cls, ptgt, pos_weight=torch.tensor(a.pres_pos_weight, device=cls.device))
        if teacher is not None:                                     # v27: 양성 잊지 않기 — 정답 양성에서 v4 보다 낮아진 만큼 벌점
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                _, tcls = teacher(x)
            gap = F.relu(torch.sigmoid(tcls.float().flatten()) - torch.sigmoid(cls))
            pres = pres + a.distill_w * (gap * (ptgt > 0.5).float()).sum() / (ptgt > 0.5).float().sum().clamp_min(1)
        if a.recipe in ("v4", "v5", "v6", "v17", "v26", "v27", "v28"):               # M4: 지표가 공짜로 봐주는 GT 바깥 1 px 고리는 BCE 에서 제외 + 허용-F 대리
            ring = F.max_pool2d(m, 3, 1, 1) - m
            w = 1.0 - (F.interpolate(ring, scale_factor=a.upscale, mode="nearest") if a.upscale > 1 else ring)
            bce = (F.binary_cross_entropy_with_logits(logit, mu, pos_weight=pw, reduction="none") * w).sum() / w.sum().clamp_min(1)
            tolf = tolerant_f_loss(F.avg_pool2d(p, a.upscale) if a.upscale > 1 else p, m, beta=a.tolf_beta)
            loss = bce + tolf + 0.5 * dice + 0.5 * pres
        else:
            bce = F.binary_cross_entropy_with_logits(logit, mu, pos_weight=pw)
            tolf = torch.zeros((), device=logit.device)
            loss = bce + dice + 0.5 * pres
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if step <= a.freeze_enc_steps:                       # v17: 초반 인코더 고정(ImageNet 특징 보존)
            opt.param_groups[0]["lr"] = 0.0
        scaler.step(opt); scaler.update(); sched.step()
        if ema is not None:
            ema_update(step)
        if step % 50 == 0:
            el = time.time() - t0
            spi = el / (step - start)
            eta = spi * (a.steps - step)
            print(f"step {step}/{a.steps} ({100 * step / a.steps:.1f}%) loss {loss.item():.4f} bce {bce.item():.4f} "
                  f"dice {dice.item():.4f} tolF {tolf.item():.4f} pres {pres.item():.4f} {spi:.3f}s/it "
                  f"elapsed {el / 60:.1f}m ETA {eta / 60:.1f}m best {best['score']:.4f}", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            pr, pc = predict(eval_model(), vx)
            r = tune(pr, pc, vm, mask_grid, PRES_GRID_V5 if a.recipe in ("v5", "v6", "v17") else PRES_GRID)
            if len(set(vg)) > 1:                                    # 성분별 점수(같은 임계) — 어느 외관에서 약한지 기록
                r["groups"] = {g: round(eval_at(pr[vg == g], pc[vg == g], vm[vg == g], r["mask_thr"], r["presence_thr"])[0]["score"], 4)
                               for g in dict.fromkeys(vg)}
            r["step"] = step; log.append(r)
            if a.save_every_eval:
                torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in eval_model().state_dict().items()}, out / f"weights_step{step}.pt")
            print("VAL", json.dumps(r), flush=True)
            if r["score"] > best["score"]:
                best = r
                sd = {k: (v.half() if v.is_floating_point() else v) for k, v in eval_model().state_dict().items()}
                torch.save(sd, out / "weights_fp16.pt")
                conf = dict(arch=a.arch, encoder=a.encoder, upscale=a.upscale, vfm_tokens=a.vfm_tokens, vfm_rein_pos=a.vfm_rein_pos, vfm_anyup=bool(a.vfm_anyup), pos_weight=a.pos_weight, tolf_beta=a.tolf_beta, aux_presence=True, weights=["weights_fp16.pt"],
                            mask_thr=r["mask_thr"], presence_thr=r["presence_thr"], inflate_to=56, tta="d4", fp16=True, batch=32, mean=MEAN.tolist(), std=STD.tolist(),
                            time_budget_s=5700, input_chroma_sigma=a.input_chroma_sigma, recipe=a.recipe,
                            notes=dict(val=r, steps=a.steps, seed=a.seed, split_seed=split_seed, synth_override=a.synth, bg_extra=a.bg_extra, ema=a.ema, synth=str(cfg), bg=a.bg_dir, mined=a.mined, n_mined=n_mined,
                                       val_kind="loro" if a.recipe == "v17" else ("valmix" if a.recipe in ("v4", "v5", "v6", "v26", "v27", "v28") else "base"), init=a.init,
                                       sprites=a.sprites, real=a.real,
                                       n_scenes=dict(train=len(tr_paths), val=len(val_paths))))
                (out / "config.json").write_text(json.dumps(conf, ensure_ascii=False, indent=1), encoding="utf-8")
            (out / "val_log.json").write_text(json.dumps(log, indent=1))
        if step % a.ckpt_every == 0 or step % a.eval_every == 0 or step == a.steps:
            save_ckpt(step)
        if step >= a.steps:
            break
    (out / "val_log.json").write_text(json.dumps(log, indent=1))
    print("BEST", json.dumps(best))


if __name__ == "__main__":
    main()
