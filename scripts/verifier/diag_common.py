"""diag_common.py — 진단 공용: 우리 모델 로드, 제출 노트북과 같은 추론·결정 규칙, 채점.

우리 모델: predict.ipynb 와 같은 D4 TTA 확률 평균 + postproc.decide(config 의 mask_thr / presence_thr / inflate_to).
(원본에 있던 주최 베이스라인 로드 함수는 주최 제공 가중치를 쓰므로 이 저장소에서는 뺐다.)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]                       # 저장소 루트
NB = ROOT / "src"
sys.path.insert(0, str(NB))

import metric as M                                                # noqa: E402
from postproc import decide                                       # noqa: E402
from train import Net, MEAN, STD                                  # noqa: E402


def load_ours(run_name):
    run = NB / "runs" / run_name
    cfg = json.loads((run / "config.json").read_text())
    net = Net(cfg["arch"], cfg["encoder"], cfg["upscale"], encoder_weights=None).cuda().eval()
    sd = torch.load(run / cfg["weights"][0], map_location="cpu", weights_only=True)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    return net, cfg


def _norm(imgs):
    x = (np.asarray(imgs, np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().cuda()


@torch.no_grad()
def predict_ours(net, imgs, tta="d4", bs=48):
    """→ probs (N,H,W), pres (N,) — 노트북 predict_batch 와 같은 계산."""
    views = [(k, f) for f in (False, True) for k in range(4)]
    views = {"none": views[:1], "flip": [views[0], views[4]], "d4": views}[tta]
    P, C = [], []
    for s in range(0, len(imgs), bs):
        x = _norm(imgs[s:s + bs])
        ap, ac = 0.0, 0.0
        for k, flip in views:
            xt = torch.rot90(x.flip(-1) if flip else x, k, (-2, -1))
            with torch.autocast("cuda", dtype=torch.float16):
                logit, cls = net(xt)
            p = torch.sigmoid(logit.float())
            if net.up > 1:
                p = F.avg_pool2d(p, net.up)
            p = torch.rot90(p[:, 0], -k, (-2, -1))
            ap = ap + (p.flip(-1) if flip else p)
            ac = ac + torch.sigmoid(cls.float()).flatten()
        P.append((ap / len(views)).cpu().numpy()); C.append((ac / len(views)).cpu().numpy())
    return np.concatenate(P), np.concatenate(C)


def decide_all(probs, press, cfg, **override):
    c = dict(cfg, **override)
    return np.stack([decide(p, q, c) for p, q in zip(probs, press)])


def score(pred, gt):
    r = M.score_arrays(pred, gt)
    keep = ("score", "f1", "f1_pos", "f1_neg", "shape", "shape_detected_mean", "tp", "fp", "fn", "tn")
    return {k: (round(float(r[k]), 4) if isinstance(r[k], float) else int(r[k])) for k in keep if k in r}
