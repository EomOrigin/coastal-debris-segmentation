#!/bin/bash
# train_final.sh — 최종 제출(newbase-v22ens3)에 들어간 모델을 다시 학습하는 명령 모음.
#
# 인자는 대회 중 실제로 쓴 실행 스크립트에서 그대로 옮겼고, 단계마다 원래 위치를 적었다.
# 각 결과 폴더의 config.json(시드·스텝·초기화·손실 옵션·합성 설정)과 대조해 확인했다.
# 원래 스크립트에 있던 GPU 점유 확인, 같은 대기열의 다른 실험, 로그 필터는 뺐다.
#
# 실행: 저장소 루트에서  bash scripts/train_final.sh <단계> [--resume]
#   단계 순서: v2 → mine_v2 → v4 → v19_d3 → v22p → v22p_s2 → v26mk2   (+ 2차 분류기용 mine_v4)
#   끊겼으면 같은 단계를 --resume 을 붙여 다시 실행하면 runs/<이름>/last.pt 부터 이어진다.
# 필요한 입력: data/backgrounds/(배경 장면, data/README.md), src/pretrained/anyup_paper.pth(AnyUp 가중치)
set -eo pipefail
cd "$(dirname "$0")/../src"

BG=../data/backgrounds
TRAIN_LIST=$BG/splits/v2_train.txt
MINED=$BG/mined_negatives_v2/crops
EXTRA=("${@:2}")

run() {   # $1 = 출력 폴더, 나머지 = train.py 인자
  local out=$1; shift
  mkdir -p "$out"
  python train.py "$@" --out "$out" "${EXTRA[@]}" 2>&1 | tee -a "$out/train.log"
}

VFM22=(--arch VFM --vfm_tokens 48 --vfm_rein_pos 64 --vfm_anyup pretrained/anyup_paper.pth
       --encoder vit_large_patch16_dinov3.sat493m --ema 0.999)

case "$1" in
  v2)
    # 채굴용 초기 모델(레시피 v1, ResNet-34). 최종 제출에는 들어가지 않고, 아래 mine_v2 의 헛경보 채굴에만 쓴다.
    # runs/v2_r34/config.json 에 남은 설정은 20,000 스텝·시드 0·배경 v2_train 뿐이라 나머지는 기본값이다.
    run runs/v2_r34 --bg_dir "$TRAIN_LIST" --encoder resnet34 --steps 20000
    ;;
  mine_v2)
    # 실배경 헛경보 채굴(원래: src/mine_hardneg.py 설명, data/README.md). 대회 때 결과는 4,667장.
    # 결과 시트(sheet.jpg)를 눈으로 보고 쓰레기로 보이는 크롭이 없는지 확인한 뒤 썼다.
    python mine_hardneg.py --run runs/v2_r34 --bg "$TRAIN_LIST" --out $BG/mined_negatives_v2
    ;;
  v4)
    # 존재 판정 모델 v4_r34 (원래: 대회 폴더 README.md 의 학습 명령). 검증 최고 step 17,000 → weights_fp16.pt
    run runs/v4_r34 --recipe v4 --bg_dir "$TRAIN_LIST" --mined "$MINED" \
      --encoder resnet34 --steps 20000 --batch 24 --workers 48 --val_n 1000
    ;;
  v19_d3)
    # 얼린 DINOv3 위성판 마스크 모델 1단계. 최종에서는 구제 규칙 판단과 아래 모델들의 초기값으로 쓴다
    # (원래: research/v19/run_v19_train.sh d3). 제출에 쓴 파일: weights_step6000.pt
    run runs/v19_d3 --recipe v4 --arch VFM --ema 0.999 --lr 3e-4 --bg_dir "$TRAIN_LIST" --mined "$MINED" \
      --steps 6000 --batch 24 --workers 48 --val_n 1000 --eval_every 1000 --save_every_eval \
      --encoder vit_large_patch16_dinov3.sat493m
    ;;
  v22p)
    # 마스크 멤버 1: 토큰 격자 48(768 px) + Rein 위치별 항 + AnyUp, 정밀도 중심 손실, 시드 1
    # (원래: research/v21/run_v22b.sh 의 두 번째 학습). 제출에 쓴 파일: weights_step4000.pt
    run runs/v22p --recipe v4 "${VFM22[@]}" --lr 1e-4 --seed 1 --pos_weight 1 --tolf_beta 0.8 \
      --init runs/v19_d3/weights_step6000.pt --init_partial \
      --bg_dir "$TRAIN_LIST" --mined "$MINED" \
      --steps 4000 --batch 12 --workers 48 --val_n 1000 --eval_every 1000 --save_every_eval
    ;;
  v22p_s2)
    # 마스크 멤버 2: v22p 와 같은 구성, 시드 2, 1,500 스텝 (원래: research/v23/run_queue.sh 의 E1)
    # 제출에 쓴 파일: weights_step1500.pt
    run runs/v22p_s2 --recipe v4 "${VFM22[@]}" --bg_dir "$TRAIN_LIST" --mined "$MINED" \
      --batch 12 --workers 48 --val_n 1000 --eval_every 500 --save_every_eval \
      --steps 1500 --lr 1e-4 --seed 2 --pos_weight 1 --tolf_beta 0.8 \
      --init runs/v19_d3/weights_step6000.pt --init_partial
    ;;
  v26mk2)
    # 마스크 멤버 3: v22p_s2 에서 레시피 v26(흩어진 조각은 정답 없음)으로 1,500 스텝 미세조정, 시드 12
    # (원래: research/v26/run_v26mask.sh 의 두 번째 학습). 제출에 쓴 파일: weights_step1500.pt
    run runs/v26mk2 --recipe v26 "${VFM22[@]}" --bg_dir "$TRAIN_LIST" --mined "$MINED" \
      --batch 12 --workers 48 --val_n 1000 --eval_every 500 --save_every_eval --pos_weight 1 --tolf_beta 0.8 \
      --steps 1500 --lr 1e-4 --seed 12 --init runs/v22p_s2/weights_step1500.pt
    ;;
  mine_v4)
    # 2차 분류기 음성용 2차 채굴: v4 로 원본·위성 룩·크롭 스트레치 판을 채굴 (원래: src/mine_hardneg.py 설명).
    # 대회 때 결과는 11,722장.
    python mine_hardneg.py --run runs/v4_r34 --bg "$TRAIN_LIST" --variants raw,sat,cstretch --out $BG/mined_negatives_v4
    ;;
  *)
    echo "사용법: bash scripts/train_final.sh {v2|mine_v2|v4|v19_d3|v22p|v22p_s2|v26mk2|mine_v4} [--resume]"; exit 1 ;;
esac
