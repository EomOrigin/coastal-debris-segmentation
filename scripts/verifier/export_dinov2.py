"""export_dinov2.py — 2차 분류기 백본 파일(dinov2_vitb14_reg4_448_fp16.pt) 만들기.

대회 때 이 파일을 만든 코드는 남아 있지 않아 이 저장소에서 새로 썼다. 제출 노트북과 build_verifier_data.py 가 읽는 형식
(timm vit_base_patch14_reg4_dinov2, num_classes=0, img_size=448, float16 state_dict)에 맞췄고,
실제 제출 패키지의 파일과 텐서 값이 모두 같음을 확인했다.
    python export_dinov2.py [출력 경로=scripts/verifier/dinov2_vitb14_reg4_448_fp16.pt]
"""
import sys
from pathlib import Path

import timm
import torch

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "dinov2_vitb14_reg4_448_fp16.pt"
m = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=448)
torch.save({k: v.half() for k, v in m.state_dict().items()}, out)
print("saved", out)
