#!/usr/bin/env python
"""make_splits_v2.py — v2 배경 장면을 학습/평가로 나누고 통합 장부를 만든다.

    python make_splits_v2.py        → backgrounds/splits/v2_train.txt, v2_test.txt, backgrounds/ledger_v2.csv

  * 평가(test) = 소스·지역별 8% (최소 1장). v1·v2 어느 모델도 학습에 쓰지 않은 장면 → 두 모델을 공정하게 비교하는 용도.
  * 학습(train) = 나머지 신규 장면 + v1 의 기존 46장면(scenes_1024).
  * 목록의 경로는 목록 파일 위치 기준 상대경로(컨테이너·호스트 어디서든 동작).
"""
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "backgrounds"
TEST_FRAC = 0.08


def main():
    rows = []
    for p in sorted((ROOT / "_meta").glob("ledger_*.csv")):
        rows += [r for r in csv.DictReader(open(p)) if (ROOT / r["source"] / r["file"]).exists()]
    rng = random.Random(2026)
    groups = {}
    for r in rows:
        groups.setdefault((r["source"], r["region"]), []).append(r)
    test = set()
    for key, g in sorted(groups.items()):
        g = sorted(g, key=lambda r: r["file"])
        rng.shuffle(g)
        test |= {r["file"] for r in g[:max(1, round(len(g) * TEST_FRAC))]}
    out = ROOT / "splits"; out.mkdir(exist_ok=True)
    legacy = sorted((ROOT / "scenes_1024").glob("*.png"))
    tr = [f"../scenes_1024/{p.name}" for p in legacy] + [f"../{r['source']}/{r['file']}" for r in rows if r["file"] not in test]
    te = [f"../{r['source']}/{r['file']}" for r in rows if r["file"] in test]
    (out / "v2_train.txt").write_text("\n".join(tr) + "\n")
    (out / "v2_test.txt").write_text("\n".join(te) + "\n")
    cols = list(rows[0].keys()) + ["split"]
    with open(ROOT / "ledger_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(dict(r, split="test" if r["file"] in test else "train"))
    by_src = {}
    for r in rows:
        s = by_src.setdefault(r["source"], [0, 0]); s[r["file"] in test] += 1
    print("train", len(tr), "(legacy", len(legacy), ") / test", len(te))
    for k, (a, b) in sorted(by_src.items()):
        print(f"  {k}: train {a} test {b}")


if __name__ == "__main__":
    main()
