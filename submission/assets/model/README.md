# assets/model/ (가중치 미포함)

최종 제출 패키지에는 아래 7개 파일(합계 약 2.6 GB)이 있었습니다. 저장소에는 넣지 않았습니다.
`scripts/make_package.py` 가 학습 결과를 이 이름 규칙(`weights_<sha256>.pt`)으로 복사하고 `../config.json` 을 맞춥니다.

| 파일 (config.json 이름) | 크기 (바이트) | 내용 | 만드는 곳 |
|---|---|---|---|
| `weights_7dad6e90….pt` | 49,003,726 | 존재 판정 U-Net(ResNet-34) `v4_r34` | `scripts/train_final.sh v4` → `src/runs/v4_r34/weights_fp16.pt` |
| `weights_0b42cce5….pt` | 629,115,516 | 마스크 멤버 `v22p` | `train_final.sh v22p` → `weights_step4000.pt` |
| `weights_b7e0e17d….pt` | 629,115,516 | 마스크 멤버 `v22p_s2` | `train_final.sh v22p_s2` → `weights_step1500.pt` |
| `weights_5c653fca….pt` | 629,115,516 | 마스크 멤버 `v26mk2` | `train_final.sh v26mk2` → `weights_step1500.pt` |
| `weights_ea26d1a5….pt` | 620,960,472 | 구제 규칙 전용 `v19_d3` | `train_final.sh v19_d3` → `weights_step6000.pt` |
| `dinov2_vitb14_reg4_448_fp16.pt` | 172,685,588 | 2차 분류기 백본 (DINOv2 ViT-B/14 reg4, 448 입력, fp16) | `scripts/verifier/export_dinov2.py` |
| `verifier_head.pt` | 5,013,440 | 2차 분류기 MLP 헤드 (정규화 평균·표준편차 포함) | `scripts/verifier/train_verifier.py` |

다시 학습하면 sha256 이 달라지므로 파일 이름도 달라집니다. 이름은 `make_package.py` 가 config.json 에 함께 고쳐 씁니다.
