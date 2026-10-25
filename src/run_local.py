#!/usr/bin/env python
"""run_local.py — 채점 서버의 추론 단계를 흉내 내 노트북을 로컬 실행한다.
서버는 추론 단계에서 '!' '%' 로 시작하는 줄을 제거한 뒤 실행하므로, 여기서도 그렇게 지운다(→ 제출 셀이 절대 실행되지 않는다).
    python run_local.py <제출폴더> <입력폴더> <prediction.csv>
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

sub, inp, pred = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
nb = json.loads((sub / "predict.ipynb").read_text(encoding="utf-8"))
for c in nb["cells"]:
    if c["cell_type"] == "code":
        src = c["source"] if isinstance(c["source"], list) else c["source"].splitlines(keepends=True)
        c["source"] = [l for l in src if not l.lstrip().startswith(("!", "%"))]
tmp = sub / "_local_stripped.ipynb"
tmp.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
env = dict(os.environ, AIF_INPUT_DIR=str(inp), AIF_PREDICTION_PATH=str(pred))
try:
    subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", tmp.name, "--output", str(pred.with_suffix(".executed.ipynb")),
                    "--ExecutePreprocessor.timeout=7200"], cwd=sub, env=env, check=True)
finally:
    tmp.unlink(missing_ok=True)
print("prediction:", pred, pred.stat().st_size, "bytes")
