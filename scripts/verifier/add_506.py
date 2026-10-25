"""add_506.py — AI-Hub 506 샘플 스프라이트(실제 부유쓰레기 색·질감)로 만든 집적대를 2차 분류기 양성으로 추가.
v4 레시피 그대로 렌더하되 집적대를 전부 506 스프라이트로 채운다(sprite_p=1). 스프라이트는 라벨 균등 추출(부표 쏠림 방지).
GT 는 기존 피복률 규칙(cov ≥ 0.5). 1차(v4) 존재 ≥ 0.4 만 남겨 feats.npz 에 붙인다 → feats_all.npz
    python add_506.py [개수=4000]"""
import sys
from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_verifier_data as B          # noqa: E402  (모델 로드·특징 함수 재사용)
S = B.S


class BalancedBank(S.SpriteBank):
    def __init__(self, folder):
        super().__init__(folder)
        self.by = {}
        for p in self.paths:
            self.by.setdefault(Path(p).stem.split("_", 3)[-1], []).append(p)
        self.labels = sorted(self.by)

    def get(self, rng):
        lst = self.by[self.labels[rng.integers(len(self.labels))]]
        s = cv2.imread(lst[rng.integers(len(lst))], cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(s, cv2.COLOR_BGRA2RGBA)


def src_506(bank, sprites, n, seed):
    rng = np.random.default_rng(seed)
    cfg = replace(S.cfg_v4(), sprite_p=1.0)
    for i in range(n):
        img, m, _ = S.make_sample(rng, bank.crop(rng), cfg, force_pos=True, sprites=sprites)
        if m.any():
            yield img, "aihub506_pos", f"a506_{seed}_{i // 50}"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 4000
    v4, dino = B.load_v4(), B.load_dino()
    tr, _ = B.split_scenes(S.background_paths([str(B.BG / "splits/v2_train.txt")]), 0)
    bank = S.BackgroundBank(tr, sat_p=0.6, center_p=0.5)
    sprites = BalancedBank(str(B.ROOT / "data/aihub_506_sample/derived/sprites"))
    store = dict(feat=[], scal=[], label=[], kind=[], group=[], pres=[], thumbs={})
    B.run_source("aihub506_pos", src_506(bank, sprites, n, 21), 1, B.POS_MIN, v4, dino, store)
    B.sheet(store)
    np.savez(HERE / "feats_506.npz", feat=np.concatenate(store["feat"]), scal=np.concatenate(store["scal"]),
             label=np.array(store["label"], np.int8), kind=np.array(store["kind"]), group=np.array(store["group"]),
             pres=np.concatenate(store["pres"]))
    print("feats_506.npz:", len(store["label"]))


def merge():
    z = dict(np.load(HERE / "feats.npz"))
    a = np.load(HERE / "feats_506.npz")
    for k in ("feat", "scal", "label", "kind", "group", "pres"):
        z[k] = np.concatenate([z[k], a[k]])
    np.savez(HERE / "feats_all.npz", **z)
    print("feats_all.npz: 음성", int((z["label"] == 0).sum()), "양성", int((z["label"] == 1).sum()))

if __name__ == "__main__":
    merge() if "merge" in sys.argv else main()
