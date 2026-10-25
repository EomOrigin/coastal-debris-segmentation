"""vfm_net.py — 얼린 비전 기반모델(DINOv3 위성판 / DINOv2) + Rein 방식 경량 어댑터 + 고해상 상세 가지 디코더.

합성→실제 일반화 분할 연구(Rein CVPR 2024, Rein++ 2025, DGSS 서베이 2025)의 공통 결론: 기반모델 백본은 얼리고
층마다 작은 어댑터만 학습하면(백본의 1% 남짓) 사전학습이 가진 실영상 지식이 합성 데이터에 덮어써지지 않아
GTA→실도시 mIoU 가 ResNet 계열 48% → 68% 로 오른다. 우리 문제(합성 학습 → 실제 위성 평가)와 같은 구조다.

- 백본: timm 이름. 'vit_large_patch16_dinov3.sat493m'(Maxar 0.6 m 위성 4.9억 장, DINOv3 라이선스: 상업·파생 허용, "Built with DINOv3" 표기)
        또는 'vit_large_patch14_reg4_dinov2.lvd142m'(Apache-2.0).
- 어댑터(Rein 간소판): 블록마다 학습 토큰 100개(저랭크) — 특징과 토큰의 유사도로 잔차 보정, 게이트 0 에서 시작(처음엔 원래 특징 그대로).
- 디코더: 4개 층 특징으로 피라미드(stride 4·8·16·32) → 위에서 아래로 합치며, 원해상도 상세 CNN 가지(stride 1·2·4)를 붙여 1 px 경계를 낸다.
- train.Net 이 arch="VFM" 일 때 body 로 쓴다. 입력은 Net 이 upscale 배 키운 정규화 영상(ImageNet 평균·표준편차), 출력은 (같은 해상도 마스크 logit, 존재 logit).
"""
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

IMN_MEAN = (0.485, 0.456, 0.406)
IMN_STD = (0.229, 0.224, 0.225)
ANYUP_MB = 8                      # AnyUp 한 번에 처리하는 장 수
ANYUP_CHUNK = 4096                # AnyUp 쿼리(출력 화소)를 이만큼씩 나눠 주의를 계산 — 메모리 절약(결과는 같음)
TOKENS = 32                       # 백본 입력을 32x32 토큰 격자로 맞춘다(p16 → 512 px, p14 → 448 px)


class Rein(nn.Module):
    def __init__(self, dim, m=100, r=16, v=64, pos_r=0):
        super().__init__()
        self.a = nn.Parameter(torch.randn(m, r) * 0.02)
        self.b = nn.Parameter(torch.randn(r, dim) * 0.02)
        self.v1 = nn.Linear(dim, v)
        self.v2 = nn.Linear(v, dim)
        self.gate = nn.Parameter(torch.zeros(1))
        if pos_r:                                                    # v22: Rein 원본의 (Δf̄ + f)·W_f 중 특징 자체의 선형 변환 f·W_f (저랭크 pos_r)
            self.pu = nn.Parameter(torch.randn(dim, pos_r) / math.sqrt(dim))   # 위치(토큰)마다 다른 보정을 만든다 — v19 간소판에는 없던 부분
            self.pv = nn.Parameter(torch.zeros(pos_r, dim))          # 0 초기화: 처음엔 v19 와 출력이 같다

    def forward(self, f):                                            # f: (B, N, C) 블록 출력(접두 토큰 포함)
        t = self.a @ self.b                                          # (m, C)
        s = torch.softmax((f @ t.t().to(f.dtype)) / math.sqrt(f.shape[-1]), dim=-1)
        out = f + self.gate.to(f.dtype) * (s @ self.v2(F.gelu(self.v1(t))).to(f.dtype))
        if hasattr(self, "pv"):
            out = out + (f @ self.pu.to(f.dtype)) @ self.pv.to(f.dtype)
        return out


def _rein_hook(mod, inp, out):                                       # 모듈 함수 + mod.rein → deepcopy(EMA) 해도 사본은 자기 어댑터를 쓴다
    return mod.rein(out)


def cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.GELU())


class VFMBody(nn.Module):
    def __init__(self, name, pretrained=True, dec=256, tokens=TOKENS, rein_pos=0, anyup=False, anyup_ckpt=None):
        super().__init__()
        if tokens != TOKENS:                                          # v21: 격자를 바꾼 모델은 표식 버퍼를 둔다 → 다른 격자로 만든 모델에
            self.register_buffer("grid_tokens", torch.tensor(int(tokens)))   # strict 로드하면 오류가 나서 조용히 틀리지 않는다
        self.vfm = timm.create_model(name, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        for p in self.vfm.parameters():
            p.requires_grad_(False)
        cfg = self.vfm.pretrained_cfg
        ps = self.vfm.patch_embed.patch_size
        self.res = int(tokens) * (ps[0] if isinstance(ps, (tuple, list)) else ps)
        C, L = self.vfm.embed_dim, len(self.vfm.blocks)
        self.idx = [L // 4 - 1, L // 2 - 1, 3 * L // 4 - 1, L - 1]
        for blk in self.vfm.blocks:
            blk.rein = Rein(C, pos_r=rein_pos)
            blk.register_forward_hook(_rein_hook)
        m = torch.tensor(cfg.get("mean", IMN_MEAN)).view(1, 3, 1, 1)
        s = torch.tensor(cfg.get("std", IMN_STD)).view(1, 3, 1, 1)
        self.register_buffer("in_a", torch.tensor(IMN_STD).view(1, 3, 1, 1) / s, persistent=False)
        self.register_buffer("in_b", (torch.tensor(IMN_MEAN).view(1, 3, 1, 1) - m) / s, persistent=False)
        self.lat = nn.ModuleList([nn.Sequential(nn.Conv2d(C, dec, 1, bias=False), nn.GroupNorm(32, dec)) for _ in range(4)])
        self.td = nn.ModuleList([cbr(dec, dec) for _ in range(3)])        # s32→s16, s16→s8, s8→s4
        self.d1, self.d2, self.d4 = cbr(3, 32), cbr(32, 64, 2), cbr(64, 96, 2)
        self.f4 = cbr(dec + 96, 128)
        self.f2 = cbr(128 + 64, 64)
        self.f1 = cbr(64 + 32, 32)
        self.out = nn.Conv2d(32, 1, 1)
        if anyup:                                                    # v22: AnyUp(ICLR 2026, CC BY 4.0) — 영상 밝기 경계를 길잡이로 DINO 특징을 원해상도로 복원
            from anyup.model import AnyUp
            self.anyup = AnyUp()
            if anyup_ckpt:
                self.anyup.load_state_dict(torch.load(anyup_ckpt, map_location="cpu", weights_only=True))
            for p in self.anyup.parameters():
                p.requires_grad_(False)                              # 사전학습 업샘플러는 얼림(인코더와 무관하게 일반화하는 성질 유지)
            self.any_fuse = nn.Conv2d(dec, 128, 1)
            nn.init.zeros_(self.any_fuse.weight); nn.init.zeros_(self.any_fuse.bias)   # 0 초기화: 처음엔 v19 와 출력이 같다
        self.cls = nn.Sequential(nn.Linear(C, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x):                                            # x: (B,3,H,W) ImageNet 정규화, H=W=256*upscale
        H, W = x.shape[-2:]
        v = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False) * self.in_a + self.in_b
        feats = self.vfm.forward_intermediates(v, indices=self.idx, norm=False, output_fmt="NCHW", intermediates_only=True)
        p4 = F.interpolate(self.lat[0](feats[0]), scale_factor=4, mode="bilinear", align_corners=False)
        p8 = F.interpolate(self.lat[1](feats[1]), scale_factor=2, mode="bilinear", align_corners=False)
        p16 = self.lat[2](feats[2])
        p32 = F.avg_pool2d(self.lat[3](feats[3]), 2)
        y = self.td[0](F.interpolate(p32, scale_factor=2, mode="bilinear", align_corners=False) + p16)
        y = self.td[1](F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False) + p8)
        y = self.td[2](F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False) + p4)   # stride 4 of 512 = 128
        e1 = self.d1(x); e2 = self.d2(e1); e4 = self.d4(e2)
        y = self.f4(torch.cat([F.interpolate(y, size=e4.shape[-2:], mode="bilinear", align_corners=False), e4], 1))
        y = F.interpolate(y, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        if hasattr(self, "anyup"):                                   # 원해상도(256) 특징을 AnyUp 으로 만들어 stride 2 단계에 더한다
            with torch.no_grad():
                img = F.avg_pool2d(x.float(), x.shape[-1] // e2.shape[-1]) if x.shape[-1] != e2.shape[-1] else x.float()
                lf = self.lat[3](feats[3]).float()
                nk = lf.shape[-2] * lf.shape[-1]                       # AnyUp 은 8장씩(내부 합성곱·주의 텐서의 32비트 색인 한계), 쿼리도 나눠 계산 — 결과는 같다
                chunk = max(256, min(ANYUP_CHUNK, (1 << 29) // (ANYUP_MB * 4 * nk)))
                a_hr = torch.cat([self.anyup(img[i:i + ANYUP_MB], lf[i:i + ANYUP_MB], output_size=tuple(e2.shape[-2:]), q_chunk_size=chunk)
                                  for i in range(0, img.shape[0], ANYUP_MB)])
            y = y + self.any_fuse(a_hr.to(y.dtype))
        y = self.f2(torch.cat([y, e2], 1))
        y = self.f1(torch.cat([F.interpolate(y, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        logit = self.out(y)
        if logit.shape[-2:] != (H, W):
            logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        cls = self.cls(feats[3].float().mean((2, 3)))
        return logit, cls
