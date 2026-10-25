"""chroma.py — 입력 색 정규화 (v4 M6, 선택 기능. config 의 input_chroma_sigma > 0 일 때만 동작)

학습과 추론이 '같은 코드'로 입력 영상의 색(Cb/Cr)만 σ 만큼 번지게 한다. 휘도는 건드리지 않는다.
평가 영상의 색 해상도(팬샤픈 1.2 m 다중분광 색, JPEG 4:2:0 등)가 무엇이든 모델이 늘 같은 정도로 번진 색을 보게 하려는 장치.
순수 numpy — 채점 서버에는 opencv·scipy 가 없다. make_submission.py 가 이 파일 본문을 노트북 셀에 그대로 심는다.
"""
import numpy as np

_Y = np.array([0.299, 0.587, 0.114], np.float32)                  # BT.601 전범위(JPEG) 휘도


def _gauss_kernel(sigma):
    r = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum(), r


def _blur_hw(a, sigma):
    """a: (H,W,C) float32 → 가우시안 블러(분리형, 반사 패딩)."""
    k, r = _gauss_kernel(sigma)
    h, w = a.shape[:2]
    p = np.pad(a, ((r, r), (0, 0), (0, 0)), mode="reflect")
    a = sum(k[i] * p[i:i + h] for i in range(2 * r + 1))
    p = np.pad(a, ((0, 0), (r, r), (0, 0)), mode="reflect")
    return sum(k[i] * p[:, i:i + w] for i in range(2 * r + 1))


def canon_chroma(img, sigma):
    """img: (H,W,3) uint8 RGB → 색(Cb/Cr)만 σ 화소 번진 uint8 RGB. σ<=0 이면 그대로."""
    if not sigma or float(sigma) <= 0:
        return img
    f = img.astype(np.float32)
    y = f @ _Y
    c = np.stack([(f[..., 2] - y) * 0.564, (f[..., 0] - y) * 0.713], -1)   # Cb, Cr
    c = _blur_hw(c, float(sigma))
    r = y + 1.403 * c[..., 1]
    b = y + 1.773 * c[..., 0]
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return np.clip(np.stack([r, g, b], -1) + 0.5, 0, 255).astype(np.uint8)
