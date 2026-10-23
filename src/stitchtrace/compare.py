"""Object-level comparison of two label fields and seam attribution of their differences (NumPy only)."""
import numpy as np
from .bundle import labels
from .errors import require


def overlap_table(reference, candidate):
    """Sparse contingency: (ref_ids, cand_ids, pixel counts) over positive pairs, plus object areas."""
    r, c = labels(reference).ravel(), labels(candidate).ravel()
    require(r.shape == c.shape, 'INVALID_SHAPE', 'label fields differ in shape')
    pairs, counts = np.unique(np.stack([r, c]).astype(np.int64), axis=1, return_counts=True)
    ra = dict(zip(*[x.tolist() for x in np.unique(r, return_counts=True)]))
    ca = dict(zip(*[x.tolist() for x in np.unique(c, return_counts=True)]))
    ra.pop(0, None)
    ca.pop(0, None)
    keep = (pairs[0] > 0) & (pairs[1] > 0)
    return pairs[0][keep], pairs[1][keep], counts[keep], ra, ca


def compare(reference, candidate, iou=0.5, cover=0.2):
    """Matching at IoU >= ``iou`` (one-to-one for iou >= 0.5) and split/merge counts.

    A reference object is *split* when at least two candidate objects each cover >= ``cover`` of it;
    a candidate object is a *merge* when it covers >= ``cover`` of each of at least two reference objects.
    """
    require(iou >= 0.5, 'INVALID_PARAMETER', 'iou >= 0.5 keeps matching one-to-one')
    ri, ci, n, ra, ca = overlap_table(reference, candidate)
    union = np.array([ra[a] + ca[b] for a, b in zip(ri.tolist(), ci.tolist())]) - n
    hit = n / np.maximum(union, 1) >= iou if len(n) else np.zeros(0, bool)
    matched = {int(a): int(b) for a, b in zip(ri[hit], ci[hit])}
    ious = (n / np.maximum(union, 1))[hit] if len(n) else np.zeros(0)
    split, merge = [], []
    for r_id, area in ra.items():
        sel = ri == r_id
        if np.count_nonzero(n[sel] >= cover * area) >= 2:
            split.append(r_id)
    ref_area = np.array([ra[a] for a in ri.tolist()]) if len(ri) else np.zeros(0)
    for c_id in ca:
        sel = ci == c_id
        if np.count_nonzero(n[sel] >= cover * ref_area[sel]) >= 2:
            merge.append(c_id)
    tp = len(matched)
    fp, fn = len(ca) - tp, len(ra) - tp
    return {'reference_objects': len(ra), 'candidate_objects': len(ca), 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / max(len(ca), 1), 'recall': tp / max(len(ra), 1),
            'f1': 2 * tp / max(len(ra) + len(ca), 1), 'mean_matched_iou': float(ious.mean()) if tp else 0.0,
            'split': len(split), 'merge': len(merge), 'split_ids': sorted(split), 'merge_ids': sorted(merge),
            'unmatched_reference_ids': sorted(set(ra) - set(matched)),
            'unmatched_candidate_ids': sorted(set(ca) - set(matched.values()))}


def seam_mask(shape, seams, width=1):
    """Pixels within ``width`` of a block seam (seam = first index of the next block on an axis)."""
    mask = np.zeros(shape, bool)
    for ax, cuts in enumerate(seams):
        for cut in cuts:
            index = [slice(None)] * len(shape)
            index[ax] = slice(max(cut - width, 0), min(cut + width, shape[ax]))
            mask[tuple(index)] = True
    return mask


def attribute(reference, candidate, seams, iou=0.5, width=1):
    """Locate every discrepancy between two label fields relative to the block seams.

    Reference side — one record per discrepant reference object (unmatched or split) with
    ``crosses_seam`` (its support meets a seam band of half-width ``width``), the number of
    candidate pieces covering it and ``category``: ``seam_split`` (crosses a seam and is covered by
    >= 2 candidate pieces), ``seam_other`` (crosses a seam, otherwise different) or ``interior``.

    Candidate side (``candidates``) — one record per unmatched candidate object (an extra object or fragment):
    ``seam_extra`` if it meets a seam band, otherwise ``interior_extra``.
    """
    ref = labels(reference)
    cand = labels(candidate)
    result = compare(ref, cand, iou)
    band = seam_mask(ref.shape, seams, width)
    on_seam = set(np.unique(ref[band]).tolist()) - {0}
    ri, ci, n, ra, _ = overlap_table(ref, cand)
    records = []
    discrepant = sorted(set(result['unmatched_reference_ids']) | set(result['split_ids']))
    for r_id in discrepant:
        sel = ri == r_id
        pieces = int(np.count_nonzero(n[sel] >= 0.2 * ra[r_id]))
        crosses = r_id in on_seam
        category = 'interior' if not crosses else ('seam_split' if pieces >= 2 else 'seam_other')
        records.append({'reference_id': r_id, 'pixels': ra[r_id], 'matched': r_id not in result['unmatched_reference_ids'],
                        'crosses_seam': crosses,
                        'pieces': pieces, 'category': category})
    on_seam_cand = set(np.unique(cand[band]).tolist()) - {0}
    _, _, _, _, ca = overlap_table(ref, cand)
    extras = []
    for c_id in result['unmatched_candidate_ids']:
        extras.append({'candidate_id': c_id, 'pixels': ca[c_id], 'crosses_seam': c_id in on_seam_cand,
                        'category': 'seam_extra' if c_id in on_seam_cand else 'interior_extra'})
    summary = {k: sum(r['category'] == k for r in records + extras)
               for k in ('seam_split', 'seam_other', 'interior', 'seam_extra', 'interior_extra')}
    summary['seam_objects'] = len(on_seam)
    return {'comparison': result, 'summary': summary, 'objects': records, 'candidates': extras}
