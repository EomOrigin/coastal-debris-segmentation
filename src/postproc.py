"""postproc.py — 지표 구조를 반영한 결정 규칙(순수 numpy). train.py 의 임계 탐색과 predict.ipynb 가 '같은 코드'를 쓴다
(make_submission.py 가 이 파일 본문을 노트북 셀에 그대로 심는다 → 학습 때 고른 임계가 추론 때와 어긋나지 않는다).

  * 존재 판정: FN 은 F1 과 그 패치의 형상 점수를 모두 잃고, FP 는 F1 만 잃는다 → presence_thr 는 검증셋에서 지표로 직접 고른다
  * 양성으로 판정했는데 마스크가 50 화소 미만이면 버리지 않고 50 화소 이상으로 부풀린다(버리면 그 패치는 0점 확정).
    가장 가까운 배경 화소부터 채우는데, GT 바깥 1 화소 고리는 정밀도 계산에서 공짜다(P 는 dilate1(gt) 안의 예측을 센다)
  * 음성으로 판정하면 완전한 빈 마스크(1~49 화소 찌꺼기는 어차피 빈 예측 취급)
"""
import numpy as np

MIN_POS_PX = 50


def dilate1(m):
    h, w = m.shape
    p = np.pad(m, 1)
    out = np.zeros_like(m)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out |= p[dy:dy + h, dx:dx + w]
    return out


def inflate_to_min_area(mask, prob, target=56):
    m = mask.copy()
    if not m.any():                                    # 씨앗이 없으면 확률 최댓값 화소에서 시작
        m[np.unravel_index(int(np.argmax(prob)), prob.shape)] = True
    while int(m.sum()) < target:
        ring = dilate1(m) & ~m
        need = target - int(m.sum())
        if int(ring.sum()) <= need:
            m |= ring
        else:                                          # 마지막 고리는 확률이 높은 화소부터 필요한 만큼만
            ys, xs = np.nonzero(ring)
            keep = np.argsort(-prob[ys, xs], kind="stable")[:need]
            m[ys[keep], xs[keep]] = True
    return m


def decide(prob, pres, cfg):
    """prob: (H,W) 화소 확률, pres: 패치 존재 확률 → 제출용 bool 마스크."""
    mask = prob >= float(cfg["mask_thr"])
    area = int(mask.sum())
    positive = float(pres) >= float(cfg["presence_thr"]) or area >= int(cfg.get("area_strong", 10 ** 9))
    if not positive:
        return np.zeros_like(mask)
    if area < MIN_POS_PX:
        mask = inflate_to_min_area(mask, prob, int(cfg.get("inflate_to", 56)))
    return mask
