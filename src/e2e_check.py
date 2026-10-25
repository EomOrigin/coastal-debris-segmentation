"""e2e_check.py — 합성 검증 입력 폴더를 만들고(정답 보관), 제출 노트북 산출 CSV 를 대회 지표로 채점한다.
    make    <입력폴더> <배경 폴더|목록.txt> <N>   합성 평가셋(양성 0.64)
    makeneg <입력폴더> <배경 폴더|목록.txt> <N>   실배경 원본 크롭만(전부 음성) → 헛경보율 측정용
    score   <입력폴더> <pred.csv ...>
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import metric as M
from synth import BackgroundBank, SynthCfg, background_paths, draw

cmd, root = sys.argv[1], Path(sys.argv[2])
if cmd in ("make", "makeneg"):
    bg = background_paths([sys.argv[3]]); n = int(sys.argv[4])
    rng, bank = np.random.default_rng(777), BackgroundBank(bg)
    (root / "images").mkdir(parents=True, exist_ok=True)
    gts = {}
    for i in range(n):
        if cmd == "make":
            img, m, _ = draw(rng, bank, SynthCfg())
        else:
            img, m = bank.crop(rng), np.zeros((256, 256), bool)
        pid = f"m{i + 1:04d}"
        Image.fromarray(img).save(root / "images" / f"{pid}.png"); gts[pid] = M.rle_encode(m)
    with (root / "patches.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["id"]); w.writerows([[k] for k in gts])
    (root.parent / (root.name + "_gt.json")).write_text(json.dumps(gts))
    print("made", n, "pos rate", np.mean([bool(v) for v in gts.values()]))
else:
    gts = {k: M.rle_decode(v) for k, v in json.loads((root.parent / (root.name + "_gt.json")).read_text()).items()}
    for p in sys.argv[3:]:
        r = M.score_csv(p, gts)
        print(p, {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items() if k in ("score", "f1", "f1_pos", "f1_neg", "shape", "tp", "fp", "fn", "tn")})
