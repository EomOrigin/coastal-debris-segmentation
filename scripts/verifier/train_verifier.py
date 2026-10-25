"""train_verifier.py — feats.npz(DINOv2 특징 + 1차 스칼라)로 2차 분류 헤드(MLP) 학습·검증.

그룹(장면) 단위 80/20 분할. 검증에서 '양성 유지율 99.5% / 99%' 가 되는 임계 τ 를 정하고,
그때 음성 종류별(실제 채굴·와덴해 갯벌·태안·합성 혼동물) 오탐 제거율을 보고한다.
    python train_verifier.py   → verifier_head.pt, verifier.json
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
torch.manual_seed(0)


class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    src = sys.argv[sys.argv.index("--feats") + 1] if "--feats" in sys.argv else "feats.npz"
    z = np.load(HERE / src, allow_pickle=False)
    x = np.concatenate([z["feat"].astype(np.float32), z["scal"]], 1)
    y, kind, group = z["label"].astype(np.float32), z["kind"], z["group"]
    val = np.array([zlib.crc32(g.encode()) % 5 == 0 for g in group])
    mu, sd = x[~val].mean(0), x[~val].std(0) + 1e-6
    xt = torch.from_numpy((x - mu) / sd)
    yt = torch.from_numpy(y)
    tr_i, va_i = np.flatnonzero(~val), np.flatnonzero(val)
    print(f"표본 {len(y)} (양성 {int(y.sum())}) | 학습 {len(tr_i)} / 검증 {len(va_i)}")
    head = Head(x.shape[1]).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    pos_w = torch.tensor(float((y[tr_i] == 0).sum() / max((y[tr_i] == 1).sum(), 1))).cuda()
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    ep = 40
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=ep * (len(tr_i) // 512 + 1))
    for e in range(ep):
        head.train()
        perm = np.random.default_rng(e).permutation(tr_i)
        for s in range(0, len(perm), 512):
            b = perm[s:s + 512]
            xb = xt[b].cuda() + 0.05 * torch.randn(len(b), x.shape[1], device="cuda")     # 특징 잡음 증강
            loss = lossf(head(xb), yt[b].cuda())
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    head.eval()
    with torch.no_grad():
        p = torch.sigmoid(head(xt.cuda())).cpu().numpy()
    pv, yv, kv = p[va_i], y[va_i], kind[va_i]
    pos = pv[yv == 1]
    rep = {}
    for keep in (0.998, 0.995, 0.99, 0.98):
        tau = float(np.quantile(pos, 1 - keep))
        r = {"tau": float(f"{tau:.6e}"), "pos_keep": round(float((pos >= tau).mean()), 4)}
        for k in sorted(set(kv[yv == 0])):
            m = (kv == k) & (yv == 0)
            r[k] = f"{int((pv[m] < tau).sum())}/{int(m.sum())} 제거"
        real = (yv == 0) & ~np.char.startswith(kv.astype(str), "synth")
        r["실제음성_전체_제거율"] = round(float((pv[real] < tau).mean()), 3)
        rd = (kv == "real_debris")
        r["태안실측양성_유지"] = f"{int((pv[rd] >= tau).sum())}/{int(rd.sum())}"
        rep[f"keep{keep}"] = r
        print(json.dumps(r, ensure_ascii=False))
    torch.save({"state": head.state_dict(), "mu": torch.from_numpy(mu), "sd": torch.from_numpy(sd), "d": x.shape[1]}, HERE / "verifier_head.pt")
    (HERE / "verifier.json").write_text(json.dumps({"report": rep, "tau_conservative": rep["keep0.995"]["tau"],
                                                    "tau_aggressive": rep["keep0.99"]["tau"]}, ensure_ascii=False, indent=1))
    print(f"저장: verifier_head.pt, verifier.json | tau(99.5) {rep['keep0.995']['tau']:.3e}  tau(99) {rep['keep0.99']['tau']:.3e}")


if __name__ == "__main__":
    main()
