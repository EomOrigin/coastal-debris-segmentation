"""make_package.py — 학습한 가중치로 제출 패키지를 채운다. (이 저장소에서 새로 쓴 보조 스크립트)

대회 때 패키지는 직전 패키지에 판정·분류기·마스크 멤버를 하나씩 바꿔 넣는 스크립트들로 만들었다
(make_submission → build_v16 → build_thr → build_v19 → build_v22 → make_ens). 그 스크립트들은 이전 패키지가 있어야
돌아가서 이 저장소에는 넣지 않았고, 대신 그 결과물인 최종 submission/assets/config.json 을 틀로 쓴다.
멤버 구성·가중치 비율·임계값은 그대로 두고, 가중치 파일만 weights_<sha256>.pt 이름으로 assets/model/ 에 복사하면서
config.json 의 파일 이름을 바꾼다. 대회 때와 같은 가중치를 넣으면 config.json 이 최종 제출본과 똑같이 나온다.

    python scripts/make_package.py \
        --presence src/runs/v4_r34/weights_fp16.pt \
        --mask src/runs/v22p/weights_step4000.pt src/runs/v22p_s2/weights_step1500.pt src/runs/v26mk2/weights_step1500.pt \
        --rescue src/runs/v19_d3/weights_step6000.pt \
        --verifier_backbone scripts/verifier/dinov2_vitb14_reg4_448_fp16.pt \
        --verifier_head scripts/verifier/verifier_head.pt [--tau 6.136577e-07] [--out submission] [--config_only]
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "submission"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presence", required=True, help="존재 판정 멤버(member_0) 가중치")
    ap.add_argument("--mask", required=True, nargs="+", help="마스크 멤버 가중치(config 순서: v22p, v22p_s2, v26mk2)")
    ap.add_argument("--rescue", required=True, help="구제 규칙 전용 멤버(v19_d3) 가중치")
    ap.add_argument("--verifier_backbone", required=True)
    ap.add_argument("--verifier_head", required=True)
    ap.add_argument("--tau", type=float, default=None, help="2차 분류기 임계(비우면 config 값 유지). 재학습했다면 verifier.json 의 tau_conservative")
    ap.add_argument("--out", default=str(TEMPLATE), help="패키지 폴더(기본: submission/ 을 그 자리에서 채움)")
    ap.add_argument("--config_only", action="store_true", help="가중치는 복사하지 않고 config.json 만 쓴다(이름 확인용)")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    if out != TEMPLATE.resolve():
        shutil.copytree(TEMPLATE, out, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.pt", "README.md"))
    cfg = json.loads((TEMPLATE / "assets/config.json").read_text(encoding="utf-8"))
    members = cfg["members"]
    pres = [m for m in members if float(m["presence_weight"]) > 0]
    masks = [m for m in members if float(m["mask_weight"]) > 0]
    resc = [m for m in members if float(m.get("rescue_weight", 0)) > 0]
    assert len(pres) == 1 and len(resc) == 1, "config 의 멤버 구성이 최종 제출본과 다릅니다"
    assert len(masks) == len(a.mask), f"마스크 가중치 {len(masks)}개가 필요합니다"

    model_dir = out / "assets/model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for m, w in [(pres[0], a.presence), *zip(masks, a.mask), (resc[0], a.rescue)]:
        name = f"weights_{sha(w)}.pt"
        m["weights"] = [name]
        if not a.config_only:
            shutil.copy2(w, model_dir / name)
        print(f"{m['name']:<22} {w} -> {name}")
    v = cfg["verifier"]
    if not a.config_only:
        shutil.copy2(a.verifier_backbone, model_dir / v["backbone"])
        shutil.copy2(a.verifier_head, model_dir / v["head"])
    if a.tau is not None:
        v["tau"] = a.tau
    (out / "assets/config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    print("config:", out / "assets/config.json", "| verifier tau", v["tau"])


if __name__ == "__main__":
    main()
