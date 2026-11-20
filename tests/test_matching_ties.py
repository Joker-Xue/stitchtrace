"""Regression tests for one-to-one matching at exact IoU ties (1.0.1).

In 1.0.0 every pair with IoU >= 0.5 was accepted, so a candidate formed by exactly two equal reference
objects (IoU 0.5 with each) was counted twice: tp=2, fp=-1, F1=1.33.
"""
import numpy as np
import pytest
from stitchtrace import compare, attribute
from stitchtrace.compare import match, overlap_table


def one_candidate_two_references():
    ref = np.zeros((4, 8), np.int32); ref[:, :4] = 1; ref[:, 4:] = 2
    cand = np.zeros((4, 8), np.int32); cand[:, :] = 1
    return ref, cand


def test_one_candidate_covering_two_equal_references():
    ref, cand = one_candidate_two_references()
    c = compare(ref, cand)
    assert (c['tp'], c['fp'], c['fn']) == (1, 0, 1)
    assert c['precision'] == 1.0 and c['recall'] == 0.5 and c['f1'] == pytest.approx(2 / 3)
    assert c['mean_matched_iou'] == 0.5
    assert c['unmatched_reference_ids'] == [2] and c['unmatched_candidate_ids'] == []
    assert c['merge'] == 1


def test_two_candidates_halving_one_reference():
    cand, ref = one_candidate_two_references()
    c = compare(ref, cand)
    assert (c['tp'], c['fp'], c['fn']) == (1, 1, 0)
    assert c['mean_matched_iou'] == 0.5                      # 1.0.0 averaged over both tied pairs
    assert c['unmatched_candidate_ids'] == [2] and c['split'] == 1


def test_tie_rule_is_deterministic_and_relabel_invariant_in_counts():
    ref, cand = one_candidate_two_references()
    swapped = np.where(ref == 1, 2, np.where(ref == 2, 1, 0))
    a, b = compare(ref, cand), compare(swapped, cand)
    assert {k: a[k] for k in ('tp', 'fp', 'fn', 'f1')} == {k: b[k] for k in ('tp', 'fp', 'fn', 'f1')}
    assert a['unmatched_reference_ids'] == b['unmatched_reference_ids'] == [2]


def test_attribution_extras_account_for_every_false_positive():
    cand, ref = one_candidate_two_references()
    att = attribute(ref, cand, seams=[[], [4]])
    s = att['summary']
    assert s['seam_extra'] + s['interior_extra'] == att['comparison']['fp'] == 1


@pytest.mark.parametrize('seed', range(40))
def test_random_fields_one_to_one(seed):
    rng = np.random.default_rng(seed)
    shape = (24, 24)
    def field():
        f = np.zeros(shape, np.int32)
        for k in range(1, rng.integers(2, 9)):
            y, x = rng.integers(0, 20, 2); h, w = rng.integers(1, 7, 2)
            f[y:y + h, x:x + w] = k
        return f
    ref, cand = field(), field()
    if seed % 4 == 0:                      # force exact ties: split a candidate object in two equal halves
        ref = cand.copy(); ys, xs = np.nonzero(cand == cand.max())
        if len(ys) >= 2 and len(ys) % 2 == 0:
            ref[ys[len(ys) // 2:], xs[len(xs) // 2:]] = cand.max() + 1
    c = compare(ref, cand)
    ri, ci, n, ra, ca = overlap_table(ref, cand)
    union = np.array([ra[a] + ca[b] for a, b in zip(ri.tolist(), ci.tolist())]) - n
    keep = match(ri, ci, n / np.maximum(union, 1), 0.5)
    assert len(set(ri[keep].tolist())) == len(set(ci[keep].tolist())) == keep.sum() == c['tp']
    assert 0 <= c['tp'] <= min(c['reference_objects'], c['candidate_objects'])
    assert c['fp'] >= 0 and c['fn'] >= 0 and 0 <= c['f1'] <= 1 and c['precision'] <= 1
    assert len(c['unmatched_reference_ids']) == c['fn'] and len(c['unmatched_candidate_ids']) == c['fp']
    # without ties the result equals the plain IoU >= 0.5 rule of 1.0.0
    iou = n / np.maximum(union, 1)
    if not np.any(iou == 0.5):
        assert c['tp'] == int(np.count_nonzero(iou >= 0.5))
