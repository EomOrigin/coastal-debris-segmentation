**한국어** | [English](README_EN.md)

# 위성 영상 해안 쓰레기 분할 — 학습 데이터 없이 합성 데이터로

AIFactory 「2026 국립공원 위성 모니터링 AI 챌린지」 주제 4(해안 쓰레기 탐지 및 규모 추정)에 혼자 참가해 낸 최종 제출 모델의 코드입니다.
최종 순위는 Private 리더보드 138팀 중 6위(0.8586)입니다.

## 개요

| 항목 | 내용 |
|---|---|
| 대회 | AIFactory #9307 「2026 국립공원 위성 모니터링 AI 챌린지 주제 4: 해안 쓰레기 탐지 및 규모 추정」 (2026-09-14 ~ 10-06) |
| 과제 | 0.3 m급 256×256 RGB 위성 패치에서 쓰레기 집적대 화소를 1로 칠한 마스크를 RLE 로 제출 (단일 클래스 분할) |
| 지표 | 0.5 × 존재 판정 macro-F1 (마스크가 50화소 이상이면 양성) + 0.5 × 형상 점수 (정답 양성 패치마다 1화소 허용 F 의 평균, 놓친 양성은 0) |
| 팀 | 다닐로바 (1인) |
| 결과 | **Private 0.8586483812, 138팀 중 6위** · Public 0.8320671333 (제출 시점 9위) |
| 최종 제출 | `newbase-v22ens3` (이 저장소의 `submission/`) |

## 문제

- 주최 측이 학습 데이터를 한 장도 주지 않았습니다. 평가 영상도 채점 서버에서만 열리고, 제출한 노트북이 서버에서 자동으로 추론합니다. 학습 데이터는 전부 직접 만들어야 했습니다.
- 평가 영상이 어떤 센서와 처리를 거쳤는지 모릅니다. 첫 모델(v1)은 제 합성 평가셋에서 0.902 였지만 Public 은 0.651 이었습니다. 합성 점수와 실제 점수 사이에 0.25 의 격차가 있었습니다.
- 진단해 보니 v1 은 '선명한 유채색 점 검출기'였습니다. 흰색·회색 조각만 있는 합성 양성에서는 0.263 까지 떨어졌습니다. 양성에는 늘 유채색 조각이 섞이고 가짜 음성(자갈·포말)은 무채색인 합성 설계가 "색이 있으면 쓰레기"라는 지름길을 가르쳤다고 봤고, v4 에서 이 지름길을 끊자 Public 이 +0.122 올랐습니다.
- 본 제출은 하루 3회이고 결과는 점수 하나뿐입니다. Public 은 194장(양성 124, 음성 70)입니다. 이 수는 공개 리더보드 점수들을 지표 식으로 역산해 얻었습니다. 장수가 적어 두 제출의 점수 차 표준편차가 0.015~0.02 쯤으로 추정됐고, 작은 개선은 잡음에 묻힙니다.
- 지표가 미탐(FN)에 더 가혹합니다. 놓친 양성은 F1 과 그 패치의 형상 점수를 함께 잃고, 오탐(FP)은 F1 만 잃습니다. 최종 상태에서 정탐 1개를 잃으면 −0.0058, 오탐 1개를 지우면 +0.0033 이었습니다. 오탐을 거르는 규칙은 정탐 1개를 잃을 때마다 오탐을 1.8개 넘게 지워야 이득입니다.

## 방법

### 1. 배경과 합성 데이터 (`src/synth.py`, `data/scripts/`)

- **배경**: OSM 해안선 좌표를 중심으로 공개 정사영상 5종(미국 NAIP·NOAA ERI, 뉴질랜드 LINZ, 일본 국토지리원, 프랑스 IGN)에서 1024×1024 @0.3 m 장면 3,950장을 받았습니다. 소스·지역별 8%(317장)는 어떤 모델도 학습에 쓰지 않는 시험 분할로 떼었습니다. 학습 목록(3,679장, 이전에 받은 46장 포함)은 장면 단위로 학습 2,944 / 검증 735 로 나눕니다.
- **쓰레기 합성**: 배경 크롭 위에 해안선을 따라가는 띠(길이 30~260 px, 폭 3~18 px)나 덩어리(반경 5~26 px) 모양의 밀도장을 만듭니다. 4배 초해상 캔버스에 조각을 그린 뒤 줄이고, 정답은 조각 피복률 0.5 이상인 화소입니다. 양성 비율은 0.64 입니다.
- **색 지름길 끊기**: 집적대의 25%는 무채색 조각만, 50%는 유채색 비율 5~45%로 만듭니다. 합성이 끝난 영상의 색(Cb/Cr)만 σ 0.3~2.5 로 번지게 하고(확률 0.7), JPEG(q 70~95, 확률 0.3)를 섞었습니다.
- **위성 룩**: 표본의 60%는 물을 어둡고 푸르게 바꾸고 장면 단위 2~98% 스트레치를 겁니다. 주최 베이스라인 가중치의 BN 통계로 역산한 평가 영상 평균색(RGB 48/59/64)을 기준으로 삼았습니다.
- **음성**: 포말, 자갈·조개, 해조류·유목, 물고임, 양식장 부표 줄을 정답 없이 합성합니다. 초기 모델(v2)이 실제 배경에서 쓰레기로 잘못 찍은 크롭 4,667장을 음성의 20%로 섞었습니다.
- **학습 증강**: 흑백(확률 0.2), 색조(±0.05)·채도(×0.5~1.3) 흔들기.

### 2. 판정과 마스크를 나눈 구조

최종 모델은 "이 패치에 쓰레기가 있나"(판정)와 "어디를 칠하나"(마스크)를 다른 모델이 맡습니다.
같은 판정에 마스크만 큰 모델로 바꾼 제출에서 Public 이 올랐고(v13a, +0.013), 큰 모델이 판정까지 맡으면 오히려 떨어졌기 때문입니다.

| 단계 | 모델 | 규칙 |
|---|---|---|
| 1차 판정 | U-Net(ResNet-34) + 존재 분류 헤드 `v4_r34`, 입력 2배 확대 | 회전·뒤집기 8방향 평균 존재 확률 ≥ 0.9 이면 양성 |
| 구제 | 얼린 DINOv3 분할 모델 `v19_d3` | 존재 확률 0.7~0.9 이고, 화소 확률 ≥ 0.9 인 화소가 50개 이상이면 양성 |
| 2차 분류기 | 얼린 DINOv2-B/14 reg4 (448 입력) 특징 + MLP | 1차 양성만 다시 보고 p < 6.14e-7 이면 음성 |
| 마스크 | 얼린 DINOv3 위성판 ViT-L/16 분할 모델 3개 (`v22p`, `v22p_s2`, `v26mk2`) | 8방향 TTA 확률을 1/3씩 평균, 임계 0.35 |
| 후처리 | — | 양성인데 마스크가 50화소 미만이면 확률 높은 화소부터 56화소까지 채움. 음성이면 빈 마스크 |

- **마스크 모델** (`src/vfm_net.py`): DINOv3 ViT-L/16 SAT-493M(위성 영상 4.9억 장으로 사전학습)은 얼리고, 블록마다 Rein 방식 어댑터(학습 토큰 100개, 게이트 0에서 시작)와 위치별 저랭크 보정항(랭크 64)만 학습합니다. 4개 층 특징으로 피라미드 디코더를 만들고 원해상도 CNN 가지를 붙였습니다. 얼린 AnyUp 업샘플러가 만든 원해상도 특징을 stride 2 단계에 더합니다. 입력은 768 px(토큰 격자 48)이고, 학습 파라미터는 전체 314.4M 중 10.5M 입니다.
- **손실** (`src/train.py`): 정답 바깥 1화소 고리를 뺀 BCE + 1화소 허용 F 의 미분 가능 대리 + 0.5·Dice + 0.5·존재 BCE. 마스크 모델 3개는 양성 화소 가중을 3에서 1로 낮추고 F_β(β 0.8)를 써서 정밀도 쪽으로 기울였습니다. Public 에서 잰 정밀도 0.71 이 재현율 0.82 보다 낮아, 마스크가 넘치게 칠한다고 봤기 때문입니다.
- **세 마스크 모델**: `v19_d3`(512 px, 6,000 스텝)에서 출발해 768 px 로 이어 학습한 `v22p`(4,000 스텝, 시드 1)와 `v22p_s2`(1,500 스텝, 시드 2), 그리고 `v22p_s2` 를 레시피 v26 으로 1,500 스텝 미세조정한 `v26mk2` 입니다. v26 레시피는 하드 네거티브의 절반을 '집적대를 이루지 않은 흩어진 조각'으로 바꾸고, 이것을 음성·양성 패치 모두에 정답 없이 깝니다. 그래서 `v26mk2` 는 집적대만 칠하는 쪽으로 기웁니다.
- **2차 분류기** (`scripts/verifier/`): 1차 판정 모델이 속는 표본만 모아 학습했습니다. 음성 21,369개(실제 해안 채굴 크롭 12,376, 네덜란드 갯벌 3,141, AI-Hub 태안 드론 타일 3,332, 합성 혼동물 2,520), 양성 9,870개(합성 집적대 6,824, AI-Hub 506 조각으로 합성한 집적대 2,901, AI-Hub 태안 실측 145)입니다. 장면 단위로 나눈 검증에서 양성 99.5%를 유지하는 τ 를 골랐습니다.
- **추론 안전장치** (`submission/predict.ipynb`): 시간 예산(5,700초)을 넘길 것 같으면 마스크 TTA 만 8 → 2 → 1회로 줄이고, 판정 경로는 항상 8방향으로 고정합니다. 마스크 모델이 오류를 내면 그 배치만 판정 모델 자체 마스크로 대신 냅니다.

## 검증 전략

1. **합성 검증셋**: 체크포인트와 임계값은 합성 검증셋 1,000장(valmix: 기본·색 번짐 σ1.5·σ2.0·위성 룩·무채색·저채도 6개 외관 묶음)에서 대회 지표를 그대로 계산해 골랐습니다.
2. **대리 지표의 한계**: 처음에는 '색 번짐 σ1.5' 점수가 Public 순서를 세 번 맞혔습니다(v1 < v6 < v4). 그런데 큰 백본(v10)에서는 +0.033 을 예측했고 Public 은 −0.007 이었습니다.
3. **검증셋 점검**: Public 점수가 있는 제출 7개를 패키지 그대로 검증셋 10종에 다시 채점했습니다. 차이가 0.015 이상인 13쌍은 valmix 가 모두 맞혔지만, ±0.01 수준의 차이를 가르는 합성 검증셋은 없었습니다. 검증용 변형을 학습 증강에도 넣었기 때문에, 합성 점수가 '제 합성기에 얼마나 맞췄나'를 재게 된 것입니다. 검증과 Public 의 차이는 v1 −0.02 에서 v11 +0.14 까지 커졌습니다.
4. **Public A/B**: 그래서 v13a 부터는 한 번에 판정이나 마스크 중 하나만 바꿔 냈습니다. 판정이 같으면 두 점수 차의 2배가 곧 형상 점수 차입니다.
5. **진단 제출**: 판정은 그대로 두고 모든 양성 마스크를 정답과 겹치지 않는 56화소 더미로 바꾸면 형상 점수가 0이 되어 점수 = 0.5 × macro-F1 이 됩니다. 10자리 점수에서 (정탐, 오탐)이 하나로 정해집니다. 최종 직전 구성(v22pens)은 정탐 124·오탐 22·미탐 0 이었고, 만점까지의 손실 0.171 은 오탐 0.067 + 마스크 0.104 로 나뉘었습니다.
6. **그 결과**: 남은 오탐 22개는 판정 모델·DINOv2·DINOv3 가 모두 쓰레기로 확신하는 패치였습니다. 로컬에서 오탐을 잘 거른 규칙들은 Public 에서 정탐도 같이 잃어 손익분기를 넘지 못했습니다. 마지막 이득은 마스크 쪽에서 성향이 다른 모델(`v26mk2`)을 섞은 데서 나왔습니다.

## 결과

Public 주요 지점입니다(Public 194장 기준, 괄호는 그 시점 순위).

| 날짜 | 버전 | 바꾼 것 | Public |
|---|---|---|---|
| 9/28 | v1 | 합성 데이터만으로 학습한 첫 모델 (U-Net ResNet-34) | 0.6511 (20위/90팀) |
| 9/29 | v4 | 색 지름길 끊기 + 위성 룩 + 실배경 오탐 채굴 + 지표 맞춤 손실 | 0.7732 (16위/92팀) |
| 9/30 | v13a | 판정은 v4 그대로, 마스크만 큰 백본 3개 평균 | 0.7861 (12위) |
| 10/2 | v16a | DINOv2 2차 분류기 추가 (오탐 1개 제거) | 0.7898 |
| 10/3 | v18t90 | 존재 임계 0.4 → 0.9 | 0.8126 (11위) |
| 10/4 | v19t90 | 마스크를 얼린 DINOv3 위성판 모델로 교체 | 0.8161 (10위) |
| 10/4 | v21r70 | 구제 규칙 추가 | 0.8172 (10위) |
| 10/5 | v22pens | 768 px·Rein 위치별 항·AnyUp + 정밀도 중심 손실, 마스크 2개 평균 | 0.8287 (9위) |
| 10/6 | **v22ens3 (최종)** | 흩어진 조각을 칠하지 않는 마스크 `v26mk2` 추가, 마스크 3개 평균 | **0.8321 (9위)** |

최종 제출은 Private 에서 0.8586 으로 138팀 중 6위였습니다.

효과가 없었던 시도도 기록해 둡니다.

| 시도 | Public | 해석 |
|---|---|---|
| 배경 46 → 3,679장면 (v2) | 0.6380 (v1 −0.013) | 배경 다양성은 병목이 아니었음 |
| 존재 임계 0.4 → 0.3 (v4b) | 0.7501 (v4 −0.023) | 0.3~0.4 구간은 대부분 음성 |
| AI-Hub 한국 실측 조각·배경 추가 (v6) | 0.7555 (v4 −0.018) | 작고 무채색인 실측 쪽으로 맞춘 것이 역효과. 평가셋 쓰레기는 여러 색이 섞인 큰 집적대에 가까운 것으로 해석 |
| 큰 백본 EfficientNetV2-XL (v10) / 3개 앙상블 (v11) | 0.7659 / 0.7631 | 큰 모델이 합성 분포에 더 맞춰져 실제 영상 일반화가 나빠진 것으로 해석 |
| 마스크 전체를 1화소 평행이동 (v18s) | 형상 −0.016 | 정답과의 어긋남은 패치마다 방향이 다름 |
| 흩어진 조각 음성으로 판정 모델 재학습 (v26b) | 0.8284 (v22pens −0.0003) | 정탐 1개당 오탐 약 2개 제거, 손익분기 |
| 일반화 판정 거부 규칙 (v28a) | 0.8223 (v22pens −0.0064) | 로컬 선별력이 Public 으로 옮겨가지 않음 |

## 실행 방법

GPU 1장(대회 때 RTX A6000 48 GB) 기준입니다. 대회 때는 Docker 컨테이너의 conda 환경(Python 3.11, PyTorch 2.5.1 + CUDA 12.1)에서 돌렸습니다.

```bash
pip install -r requirements.txt          # 학습·로컬 추론
pip install -r requirements-data.txt     # 데이터 수집 (따로 설치해도 됨)
```

**1) 배경 수집** — 결과는 `data/backgrounds/`

```bash
python data/scripts/fetch_bg_v2.py points --shp <coastlines-split-4326 의 lines.shp>
for s in gsi naip linz ign eri; do python data/scripts/fetch_bg_v2.py fetch --source $s; done
python data/scripts/make_splits_v2.py    # splits/v2_train.txt, v2_test.txt
```

**2) 모델 학습** — `src/pretrained/anyup_paper.pth` (AnyUp 저장소의 가중치)가 필요합니다. DINOv3·ResNet-34 사전학습 가중치는 timm·smp 가 내려받습니다.

```bash
bash scripts/train_final.sh v2        # 채굴용 초기 모델
bash scripts/train_final.sh mine_v2   # 실배경 오탐 채굴 → sheet.jpg 를 눈으로 검수
bash scripts/train_final.sh v4        # 1차 판정 (A6000 로그 기준 약 118분)
bash scripts/train_final.sh v19_d3    # DINOv3 1단계, 구제 규칙용 (약 123분)
bash scripts/train_final.sh v22p      # 마스크 1 (약 143분)
bash scripts/train_final.sh v22p_s2   # 마스크 2 (약 56분)
bash scripts/train_final.sh v26mk2    # 마스크 3 (약 55분)
```

**3) 2차 분류기** — AI-Hub 71363·506 데이터는 직접 내려받아 `data/README.md` 의 위치에 둡니다.

```bash
bash scripts/train_final.sh mine_v4
python scripts/verifier/fetch_pdok.py
python data/scripts/build_aihub_debris.py --split train --src <TS_01.Drone.zip>
python data/scripts/extract_kr_bg.py <TS_01.Drone.zip> data/aihub_71363/derived/kr_bg/train
python scripts/verifier/extract_506_sprites.py
python scripts/verifier/export_dinov2.py
python scripts/verifier/build_verifier_data.py          # → feats.npz
python scripts/verifier/add_506.py                      # → feats_506.npz
python scripts/verifier/add_506.py merge                # → feats_all.npz
python scripts/verifier/train_verifier.py --feats feats_all.npz   # → verifier_head.pt, verifier.json
```

**4) 제출 패키지** — 가중치를 `submission/assets/model/` 에 넣고 config.json 의 파일 이름을 맞춥니다.

```bash
python scripts/make_package.py \
  --presence src/runs/v4_r34/weights_fp16.pt \
  --mask src/runs/v22p/weights_step4000.pt src/runs/v22p_s2/weights_step1500.pt src/runs/v26mk2/weights_step1500.pt \
  --rescue src/runs/v19_d3/weights_step6000.pt \
  --verifier_backbone scripts/verifier/dinov2_vitb14_reg4_448_fp16.pt \
  --verifier_head scripts/verifier/verifier_head.pt \
  --tau <verifier.json 의 tau_conservative>
```

**5) 로컬 실행과 채점** — 채점 서버처럼 노트북을 실행하고 대회 지표로 채점합니다.

```bash
cd src
python e2e_check.py make e2e/input ../data/backgrounds/splits/v2_test.txt 300   # 합성 평가셋 (정답 별도 보관)
python run_local.py ../submission e2e/input e2e/pred.csv
python e2e_check.py score e2e/input e2e/pred.csv
```

대회 때 제출은 `submission/` 에서 `AIF_API_KEY=<참여키> python -m aifactory submit --notebook predict.ipynb --model-name <이름>` 으로 했습니다. 참여키는 환경변수로만 넘겼습니다.

## 폴더 구조

```
coastal-debris-segmentation/
├── README.md, README_EN.md
├── requirements.txt              학습·로컬 추론
├── requirements-data.txt         데이터 수집
├── docs/method.md                최종 파이프라인 상세
├── src/                          합성·학습·지표 (대회 때 newbase/ 폴더)
│   ├── synth.py                  합성기 (최종은 레시피 v4·v26 사용)
│   ├── train.py                  학습 (U-Net / 얼린 기반 모델)
│   ├── valmix.py                 강건성 검증셋 (6개 외관 묶음)
│   ├── vfm_net.py                DINOv3 + Rein 어댑터 + 고해상 디코더
│   ├── anyup/                    AnyUp 일부 (CC BY 4.0)
│   ├── metric.py                 대회 지표 재구현
│   ├── postproc.py, chroma.py    결정 규칙, 색 정규화
│   ├── mine_hardneg.py           실배경 오탐 채굴
│   └── e2e_check.py, run_local.py   로컬 평가셋 생성·채점, 노트북 로컬 실행
├── scripts/
│   ├── train_final.sh            최종 모델 학습 명령
│   ├── make_package.py           가중치로 제출 패키지 채우기
│   └── verifier/                 2차 분류기 데이터 구축·학습
├── data/
│   ├── README.md                 데이터 배치, 출처·라이선스
│   └── scripts/                  배경·AI-Hub 수집·변환
└── submission/                   최종 제출 패키지 (가중치 제외)
    ├── predict.ipynb, requirements.txt
    └── assets/                   config.json, 라이선스 고지, code/anyup, model/README.md
```

- `submission/predict.ipynb` 와 `assets/config.json` 은 실제로 낸 파일 그대로입니다.
- `src/synth.py`, `src/train.py` 에는 최종에 쓰지 않은 실험 레시피(v5·v6·v17·v27·v28)도 남아 있습니다. 최종 모델을 학습한 코드를 그대로 두려고 지우지 않았습니다.
- 코드 주석에 나오는 `docs/…`, `research/…` 경로는 대회 때 작업 폴더의 설계 노트이고, 이 저장소에는 없습니다.
- `scripts/train_final.sh`, `scripts/make_package.py`, `scripts/verifier/export_dinov2.py` 는 이 저장소를 정리하면서 새로 썼습니다. 앞의 둘은 대회 때 흩어져 있던 실행 명령과 패키지 조립 과정을 한곳에 모은 것이고, 마지막 것은 남아 있지 않던 백본 변환 코드를 다시 쓴 것입니다(실제 제출 파일과 텐서 값이 같음을 확인).

## 데이터·가중치 안내

이 저장소에는 영상, 데이터셋, 모델 가중치가 없습니다.

- **배경 영상**: 공개 정사영상 5종(NAIP·ERI·LINZ·국토지리원·IGN)입니다. 소스별 장면 수와 라이선스는 [data/README.md](data/README.md)에 정리했습니다. 모두 상업 이용과 2차 저작물 작성이 허용되며, 의무는 출처 표시 정도입니다.
- **AI-Hub 데이터** (71363, 506): 2차 분류기 학습에만 썼습니다. 원본 재배포가 금지라 넣지 않았습니다. 본 결과물(2차 분류기)은 한국지능정보사회진흥원의 인공지능 학습용 데이터 구축사업으로 구축된 AI-Hub 데이터(「해안 오염물질 데이터」, 「AI기반 국립공원 변화탐지 모니터링 플랫폼 구축」)를 활용했습니다.
- **대회 제공물**: 주최 측 베이스라인 패키지와 대회 페이지 원문은 넣지 않았습니다.
- **가중치**: 최종 패키지의 가중치 7개(약 2.6 GB)는 넣지 않았습니다. 파일 목록과 크기는 [submission/assets/model/README.md](submission/assets/model/README.md)에 있습니다.
- **사전학습 가중치**: torchvision ResNet-34 ImageNet (BSD-3-Clause), DINOv3 ViT-L/16 SAT-493M (DINOv3 License), DINOv2 ViT-B/14 reg4 (Apache-2.0), AnyUp (CC BY 4.0). Built with DINOv3.

## 라이선스

아직 이 저장소의 라이선스 파일을 두지 않았습니다. 포함된 제3자 코드와 고지는 각자의 라이선스를 따릅니다: `src/anyup/`, `submission/assets/code/anyup/` (AnyUp, CC BY 4.0), `submission/assets/LICENSE_DINOv3.txt` (DINOv3 License 사본).
