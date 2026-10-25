# -*- coding: utf-8 -*-
"""
metric.py - exact local re-implementation of the AIFactory #9307 metric
("2026 국립공원 위성 모니터링 AI 챌린지 - 주제 4: 해안 쓰레기 탐지 및 규모 추정").

    final = 0.5 * F1 + 0.5 * Shape

    F1    : macro-F1 of patch-level presence.
            GT positive   <=> GT mask has >= 1 px
            pred positive <=> submitted mask has >= 50 px (MIN_POS_PX)
            F1_c = 2TP_c / (2TP_c + FP_c + FN_c),  F1 = (F1_pos + F1_neg) / 2
    Shape : mean over GT-positive patches of 2PR/(P+R) with
            P = |pred & dilate1(gt)| / |pred|,  R = |gt & dilate1(pred)| / |gt|
            dilate1 = 3x3 (8-neighbourhood) binary dilation, no wrap-around.
            A patch whose prediction is not positive (< 50 px) scores 0.
    Common: a submitted mask with < 50 px is an EMPTY prediction in both layers.

Source of truth: the competition page's rules/data/evaluation sections
(scraped 2026-09-21; the scrapes are not included in this repository).

Pure numpy (+ optional scipy.ndimage only in the self-test). Python 3.8 - 3.12.

Spec ambiguities (organizer code is not public) and the choice made here
  A1. F1_c when 2TP+FP+FN == 0 (class absent in both GT and prediction): 0/0.
      -> `zero_division` argument, default 0.0 (sklearn default, pessimistic).
  A2. Shape when N_pos == 0: 0/0 -> `zero_division` as well.
  A3. Overlapping / unsorted RLE runs are not listed as invalid -> accepted and
      OR-ed together (pass strict_order=True to reject them). rle_encode() never
      produces such runs, so this never matters for our own submissions.
  A4. P+R == 0 -> patch F = 0.

Public API
  rle_encode(mask) -> str                  rle_decode(rle, shape=(256,256)) -> bool array
  dilate1(mask) -> bool array              (works on (..., H, W) stacks)
  patch_shape_f(pred, gt) -> float         patch_prf(pred, gt) -> (P, R, F)
  presence_macro_f1(gt_pos, pred_pos) -> dict
  score_arrays(pred_stack, gt_stack) -> dict
  score_dicts(pred_dict, gt_dict) -> dict
  read_submission_csv(path, ids) -> {id: mask}     (organizer validation rules)
  score_csv(pred_csv, gt_dict) -> dict
  python metric.py                         runs the self-test
"""
from __future__ import annotations

import csv
import io
import re
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

H, W = 256, 256
N_PIX = H * W            # 65536
MIN_POS_PX = 50          # a submitted mask with fewer 1-pixels is an empty prediction
_INT_RE = re.compile(r"^[0-9]+$")

__all__ = [
    "H", "W", "N_PIX", "MIN_POS_PX", "InvalidSubmission",
    "rle_encode", "rle_decode", "dilate1", "patch_prf", "patch_shape_f",
    "presence_macro_f1", "score_arrays", "score_dicts",
    "read_submission_csv", "score_csv",
]


class InvalidSubmission(ValueError):
    """The organizer would fail the WHOLE submission ("제출 전체가 채점 실패")."""


# --------------------------------------------------------------------------- RLE
def rle_encode(mask: np.ndarray) -> str:
    """(H,W) binary mask -> row-major, 0-based 'start length' pairs. Empty mask -> ''."""
    m = np.asarray(mask).astype(bool).reshape(-1)
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return " ".join("%d %d" % (a, b - a) for a, b in zip(st.tolist(), en.tolist()))


def rle_decode(rle: Optional[str], shape: Tuple[int, int] = (H, W),
               strict_order: bool = False) -> np.ndarray:
    """RLE string -> (H,W) bool mask. Raises InvalidSubmission exactly where the
    organizer says the submission fails: non-integer token, odd token count,
    run length < 1, run outside 0..H*W-1."""
    n = int(shape[0]) * int(shape[1])
    out = np.zeros(n, dtype=bool)
    if rle is None:
        return out.reshape(shape)
    toks = str(rle).split()
    if not toks:
        return out.reshape(shape)
    if len(toks) % 2:
        raise InvalidSubmission("RLE has an odd number of values (%d)" % len(toks))
    for t in toks:
        if not _INT_RE.match(t):
            raise InvalidSubmission("RLE value is not a non-negative integer: %r" % t)
    vals = [int(t) for t in toks]
    prev_end = 0
    for s, l in zip(vals[0::2], vals[1::2]):
        if l < 1:
            raise InvalidSubmission("RLE run length < 1: (%d, %d)" % (s, l))
        if s < 0 or s + l - 1 > n - 1:
            raise InvalidSubmission("RLE run outside 0..%d: (%d, %d)" % (n - 1, s, l))
        if strict_order and s < prev_end:
            raise InvalidSubmission("RLE runs overlap / are unsorted at start %d" % s)
        prev_end = s + l
        out[s:s + l] = True
    return out.reshape(shape)


# ---------------------------------------------------------------------- geometry
def dilate1(mask: np.ndarray) -> np.ndarray:
    """3x3 (8-neighbourhood) binary dilation on the last two axes, zero padding
    (no wrap-around). Identical to scipy.ndimage.binary_dilation(m, np.ones((3,3)))."""
    m = np.asarray(mask).astype(bool)
    h, w = m.shape[-2], m.shape[-1]
    pad = [(0, 0)] * (m.ndim - 2) + [(1, 1), (1, 1)]
    p = np.pad(m, pad, mode="constant", constant_values=False)
    out = np.zeros_like(m)
    for dy in range(3):
        for dx in range(3):
            out |= p[..., dy:dy + h, dx:dx + w]
    return out


def patch_prf(pred: np.ndarray, gt: np.ndarray, min_pos_px: int = MIN_POS_PX
              ) -> Tuple[float, float, float]:
    """(P, R, F) of one GT-positive patch. Returns (0,0,0) if the prediction is not
    positive (< min_pos_px) or the GT is empty (such a patch is not in the mean)."""
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    n_pred, n_gt = int(pred.sum()), int(gt.sum())
    if n_pred < min_pos_px or n_gt == 0:
        return 0.0, 0.0, 0.0
    p = float((pred & dilate1(gt)).sum()) / n_pred
    r = float((gt & dilate1(pred)).sum()) / n_gt
    f = 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)
    return p, r, f


def patch_shape_f(pred: np.ndarray, gt: np.ndarray, min_pos_px: int = MIN_POS_PX) -> float:
    return patch_prf(pred, gt, min_pos_px)[2]


# ---------------------------------------------------------------------- presence
def presence_macro_f1(gt_pos: Sequence[bool], pred_pos: Sequence[bool],
                      zero_division: float = 0.0) -> Dict[str, float]:
    """Macro-F1 over the two classes (positive = debris present, negative = absent)."""
    g = np.asarray(gt_pos).astype(bool).reshape(-1)
    p = np.asarray(pred_pos).astype(bool).reshape(-1)
    if g.shape != p.shape:
        raise ValueError("gt_pos / pred_pos length mismatch")
    tp = int((g & p).sum())
    fp = int((~g & p).sum())
    fn = int((g & ~p).sum())
    tn = int((~g & ~p).sum())

    def _f1(t, a, b):
        den = 2 * t + a + b
        return float(zero_division) if den == 0 else 2.0 * t / den

    f1_pos = _f1(tp, fp, fn)          # positive class: TP=tp, FP=fp, FN=fn
    f1_neg = _f1(tn, fn, fp)          # negative class: TP=tn, FP=fn, FN=fp
    return {"f1": 0.5 * (f1_pos + f1_neg), "f1_pos": f1_pos, "f1_neg": f1_neg,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ------------------------------------------------------------------------- score
def score_arrays(pred: np.ndarray, gt: np.ndarray, zero_division: float = 0.0,
                 min_pos_px: int = MIN_POS_PX, return_per_patch: bool = False) -> Dict:
    """pred, gt: (N,H,W) binary stacks in the same order."""
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError("pred/gt must be (N,H,W) with equal shapes")
    n = pred.shape[0]
    n_pred = pred.reshape(n, -1).sum(1)
    n_gt = gt.reshape(n, -1).sum(1)
    pred_pos = n_pred >= min_pos_px
    gt_pos = n_gt >= 1
    f = np.zeros(n, dtype=np.float64)
    for i in np.flatnonzero(gt_pos & pred_pos):
        f[i] = patch_prf(pred[i], gt[i], min_pos_px)[2]
    res = presence_macro_f1(gt_pos, pred_pos, zero_division)
    n_pos = int(gt_pos.sum())
    shape = float(zero_division) if n_pos == 0 else float(f[gt_pos].sum()) / n_pos
    n_hit = int((gt_pos & pred_pos).sum())
    res.update({
        "shape": shape,
        "score": 0.5 * res["f1"] + 0.5 * shape,
        "n": int(n), "n_pos": n_pos, "n_neg": int(n - n_pos),
        # mean F over detected GT-positive patches (= shape / recall): segmentation quality
        "shape_detected_mean": (float(f[gt_pos & pred_pos].mean()) if n_hit else 0.0),
    })
    if return_per_patch:
        res["per_patch_f"] = f
        res["gt_pos"] = gt_pos
        res["pred_pos"] = pred_pos
    return res


def score_dicts(pred: Mapping[str, np.ndarray], gt: Mapping[str, np.ndarray],
                zero_division: float = 0.0, min_pos_px: int = MIN_POS_PX,
                return_per_patch: bool = False) -> Dict:
    """pred/gt: {id: (H,W) mask}. ids are taken from gt. A missing pred id is an
    empty prediction; an id unknown to gt raises InvalidSubmission."""
    unknown = [k for k in pred if k not in gt]
    if unknown:
        raise InvalidSubmission("ids not in the patch list: %s" % unknown[:5])
    ids = list(gt.keys())
    if not ids:
        raise ValueError("empty gt")
    shp = np.asarray(gt[ids[0]]).shape
    g = np.stack([np.asarray(gt[k]).astype(bool) for k in ids])
    p = np.stack([np.asarray(pred[k]).astype(bool) if k in pred else np.zeros(shp, bool)
                  for k in ids])
    res = score_arrays(p, g, zero_division, min_pos_px, return_per_patch)
    if return_per_patch:
        res["ids"] = ids
    return res


# --------------------------------------------------------------------------- CSV
def read_submission_csv(path_or_text, ids: Iterable[str], shape: Tuple[int, int] = (H, W),
                        is_text: bool = False, strict_order: bool = False
                        ) -> Dict[str, np.ndarray]:
    """Parse an `id,rle` CSV with the organizer's rules:
       - id missing from the file        -> empty prediction
       - id not in `ids` / duplicated id -> InvalidSubmission
       - unparsable RLE                  -> InvalidSubmission
    Returns {id: bool mask} for every id in `ids` (in that order)."""
    ids = list(ids)
    idset = set(ids)
    if is_text:
        fh = io.StringIO(path_or_text.lstrip("﻿"))
    else:
        fh = open(path_or_text, "r", encoding="utf-8-sig", newline="")
    try:
        rd = csv.DictReader(fh)
        cols = [c.strip() for c in (rd.fieldnames or [])]
        if "id" not in cols or "rle" not in cols:
            raise InvalidSubmission("CSV header must contain 'id' and 'rle', got %r" % cols)
        rd.fieldnames = cols
        seen: Dict[str, np.ndarray] = {}
        for row in rd:
            k = (row.get("id") or "").strip()
            if k == "" and not (row.get("rle") or "").strip():
                continue                                  # blank line
            if k not in idset:
                raise InvalidSubmission("id not in the patch list: %r" % k)
            if k in seen:
                raise InvalidSubmission("duplicated id: %r" % k)
            seen[k] = rle_decode(row.get("rle"), shape, strict_order)
    finally:
        fh.close()
    empty = np.zeros(shape, dtype=bool)
    return {k: seen.get(k, empty) for k in ids}


def score_csv(pred_csv, gt_dict: Mapping[str, np.ndarray], zero_division: float = 0.0,
              is_text: bool = False, return_per_patch: bool = False) -> Dict:
    """Score a submission CSV against {id: gt mask}. Raises InvalidSubmission when
    the organizer would reject the file."""
    ids = list(gt_dict.keys())
    shp = np.asarray(gt_dict[ids[0]]).shape
    pred = read_submission_csv(pred_csv, ids, shape=shp, is_text=is_text)
    return score_dicts(pred, gt_dict, zero_division, return_per_patch=return_per_patch)


# --------------------------------------------------------------------- self-test
def _selftest() -> None:
    rng = np.random.RandomState(0)

    # --- RLE examples copied from the data page
    m1 = np.zeros((H, W), bool); m1[10:13, 10:15] = True
    assert rle_encode(m1) == "2570 5 2826 5 3082 5", rle_encode(m1)
    assert np.array_equal(rle_decode("2570 5 2826 5 3082 5"), m1)
    m2 = np.zeros((H, W), bool); m2[255, 255] = True
    assert rle_encode(m2) == "65535 1" and np.array_equal(rle_decode("65535 1"), m2)
    assert rle_encode(np.zeros((H, W))) == "" and rle_decode("").sum() == 0
    assert rle_decode(None).sum() == 0 and rle_decode("   ").sum() == 0
    full = np.ones((H, W), bool)
    assert rle_encode(full) == "0 65536" and rle_decode("0 65536").all()
    # a run may cross a row boundary (row-major flattening)
    assert rle_decode("254 4")[0, 254:].all() and rle_decode("254 4")[1, :2].all()
    for _ in range(200):
        m = rng.rand(H, W) < rng.choice([0.01, 0.2, 0.5, 0.9])
        assert np.array_equal(rle_decode(rle_encode(m)), m)

    # --- invalid RLE -> whole submission fails
    for bad in ["1", "1 2 3", "1.0 2", "a 2", "1 0", "1 -1", "-1 2", "65535 2",
                "65536 1", "0 65537", "1e3 2", "5,6"]:
        try:
            rle_decode(bad)
        except InvalidSubmission:
            pass
        else:
            raise AssertionError("should be invalid: %r" % bad)
    assert rle_decode("10 5 12 5").sum() == 7                      # overlap: accepted (A3)
    try:
        rle_decode("10 5 12 5", strict_order=True); raise AssertionError
    except InvalidSubmission:
        pass

    # --- dilate1 == scipy 3x3 dilation (incl. borders / corners, stacks)
    try:
        from scipy import ndimage as ndi
        for _ in range(50):
            m = rng.rand(H, W) < rng.choice([0.001, 0.05, 0.5])
            m[0, 0] = m[-1, -1] = True
            assert np.array_equal(dilate1(m), ndi.binary_dilation(m, structure=np.ones((3, 3), bool)))
        st = rng.rand(4, 32, 32) < 0.05
        assert np.array_equal(dilate1(st), np.stack([ndi.binary_dilation(s, structure=np.ones((3, 3), bool)) for s in st]))
    except ImportError:                                             # pragma: no cover
        print("scipy missing: skipped the dilation cross-check")
    one = np.zeros((H, W), bool); one[0, 0] = True
    assert dilate1(one).sum() == 4                                  # corner: no wrap-around

    # --- patch F: exact, 1-px tolerance, 49 vs 50 px
    gt = np.zeros((H, W), bool); gt[100:110, 100:110] = True        # 100 px
    assert patch_prf(gt, gt) == (1.0, 1.0, 1.0)
    assert patch_shape_f(dilate1(gt), gt) == 1.0                    # 1 px fatter: still perfect
    assert patch_shape_f(np.roll(gt, 1, axis=1), gt) == 1.0         # 1 px shift: still perfect
    p2 = dilate1(dilate1(gt))                                       # 2 px fatter: P = 144/196
    P, R, F = patch_prf(p2, gt)
    assert abs(P - 144.0 / 196.0) < 1e-12 and R == 1.0
    far = np.zeros((H, W), bool); far[0:10, 0:10] = True
    assert patch_shape_f(far, gt) == 0.0
    p49 = np.zeros((H, W), bool); p49[100:107, 100:107] = True      # 49 px inside gt
    p50 = p49.copy(); p50[107, 100] = True                          # 50 px
    assert patch_shape_f(p49, gt) == 0.0                            # 49 px -> empty prediction
    P, R, F = patch_prf(p50, gt)                                    # 50 px -> scored: P=1, R=66/100
    assert P == 1.0 and abs(R - 0.66) < 1e-12 and abs(F - 2 * 0.66 / 1.66) < 1e-12

    # --- presence macro-F1
    r = presence_macro_f1([1, 1, 0, 0], [1, 0, 0, 1])
    assert abs(r["f1_pos"] - 0.5) < 1e-12 and abs(r["f1_neg"] - 0.5) < 1e-12
    r = presence_macro_f1([1, 1, 0, 0], [1, 1, 0, 0]); assert r["f1"] == 1.0
    r = presence_macro_f1([0, 0, 0], [0, 0, 0])                     # N_pos = 0
    assert r["f1_neg"] == 1.0 and r["f1_pos"] == 0.0 and r["f1"] == 0.5
    r = presence_macro_f1([0, 0, 0], [0, 0, 0], zero_division=1.0); assert r["f1"] == 1.0
    r = presence_macro_f1([1, 1], [1, 1])                           # no negatives
    assert r["f1_pos"] == 1.0 and r["f1_neg"] == 0.0 and r["f1"] == 0.5

    # --- full score: 4 patches (pos perfect, pos missed with 49 px, neg clean, neg false alarm)
    z = np.zeros((H, W), bool)
    gts = {"m0001": gt, "m0002": gt, "m0003": z, "m0004": z}
    prs = {"m0001": gt, "m0002": p49, "m0003": z, "m0004": p50}
    s = score_dicts(prs, gts)
    assert (s["tp"], s["fn"], s["tn"], s["fp"]) == (1, 1, 1, 1)
    assert abs(s["f1"] - 0.5) < 1e-12 and abs(s["shape"] - 0.5) < 1e-12 and abs(s["score"] - 0.5) < 1e-12
    # a 49-px prediction on a NEGATIVE patch is an empty prediction -> true negative
    s = score_dicts({"a": p49}, {"a": z}); assert s["tn"] == 1 and s["fp"] == 0
    # N_pos = 0: shape term = zero_division
    s = score_dicts({"a": z}, {"a": z}); assert s["shape"] == 0.0 and s["score"] == 0.25
    # tiny GT (1 px) is still GT-positive
    s = score_dicts({"a": z}, {"a": one}); assert s["fn"] == 1 and s["n_pos"] == 1
    # array API == dict API
    s1 = score_arrays(np.stack([prs[k] for k in gts]), np.stack([gts[k] for k in gts]))
    assert abs(s1["score"] - 0.5) < 1e-12

    # --- CSV rules
    txt = "id,rle\nm0001,%s\nm0004,%s\n" % (rle_encode(gt), rle_encode(p50))   # m0002/m0003 missing -> empty
    s = score_csv(txt, gts, is_text=True)
    assert (s["tp"], s["fn"], s["tn"], s["fp"]) == (1, 1, 1, 1)
    for bad_txt in ["id,rle\nm0001,\nm0001,\n",          # duplicate id
                    "id,rle\nm9999,\n",                  # unknown id
                    "id,rle\nm0001,1 2 3\n",             # odd count
                    "id,rle\nm0001,1.5 2\n",             # non-integer
                    "id,rle\nm0001,5 0\n",               # length < 1
                    "id,rle\nm0001,65535 2\n",           # out of range
                    "idx,rle\nm0001,\n"]:                # bad header
        try:
            score_csv(bad_txt, gts, is_text=True)
        except InvalidSubmission:
            pass
        else:
            raise AssertionError("should be invalid: %r" % bad_txt)
    s = score_csv("﻿id,rle\r\nm0001,\r\n", gts, is_text=True)   # BOM + CRLF are fine
    assert s["tp"] == 0 and s["fn"] == 2
    print("metric.py self-test: ALL OK")


if __name__ == "__main__":
    _selftest()
