"""Probe B: replay generic stitching policies on recorded StarDist tile predictions.

One capture per image/configuration; every policy is evaluated from the recorded raw tile
labels without inference and compared to an untiled prediction of the same model.
"""
import json, sys, time, glob
from pathlib import Path
import numpy as np, tifffile
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from stardist.matching import matching
from stitchtrace.backends.stardist import capture_stardist
from stitchtrace.replay import replay

R = Path(__file__).resolve().parents[1]
OUT = R / 'runs' / 'policy_probe'


def tiles_of(b):
    out = []
    for t in b.meta['tiles']:
        with np.load(b.path / t['cache'], allow_pickle=False) as d:
            raw = d['labels'].copy()
        out.append(dict(read=t['read'], write=t['write'], raw=raw))
    return out


def ownership_boxes(tiles, shape):
    """Disjoint per-tile boxes: split each write-overlap at its midpoint, per axis."""
    boxes = []
    for t in tiles:
        box = []
        for ax, (lo, hi) in enumerate(t['write']):
            others = sorted({tuple(u['write'][ax]) for u in tiles})
            a, b = lo, hi
            for (olo, ohi) in others:
                if (olo, ohi) == (lo, hi):
                    continue
                if olo < lo < ohi <= hi:      # predecessor overlaps our start
                    a = max(a, (lo + ohi) // 2)
                if lo <= olo < hi < ohi:      # successor overlaps our end
                    b = min(b, (olo + hi) // 2)
            box.append((a, b))
        boxes.append(box)
    return boxes


def to_global(t, local_box):
    return tuple(slice(lo - r[0], hi - r[0]) for (lo, hi), r in zip(local_box, t['read']))


def naive_crop(tiles, shape, merge_iou=None):
    """Paste each tile's raw labels inside its disjoint box (objects cut at seams).
    With merge_iou, link pieces whose raw objects overlap with IoU>=t inside shared write regions."""
    out = np.zeros(shape, np.int64)
    boxes = ownership_boxes(tiles, shape)
    offset = 0
    for i, (t, box) in enumerate(zip(tiles, boxes)):
        piece = t['raw'][to_global(t, box)].astype(np.int64)
        piece[piece > 0] += offset
        out[tuple(slice(*b) for b in box)] = piece
        t['offset'] = offset
        offset += int(t['raw'].max())
    if merge_iou is None:
        return out
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            A, B = tiles[i], tiles[j]
            inter = [(max(a[0], b[0]), min(a[1], b[1])) for a, b in zip(A['write'], B['write'])]
            if any(lo >= hi for lo, hi in inter):
                continue
            la, lb = A['raw'][to_global(A, inter)], B['raw'][to_global(B, inter)]
            pairs, counts = np.unique(np.stack([la.ravel(), lb.ravel()]), axis=1, return_counts=True)
            area_a = dict(zip(*np.unique(la, return_counts=True)))
            area_b = dict(zip(*np.unique(lb, return_counts=True)))
            for (x, y), c in zip(pairs.T, counts):
                if x and y and c / (area_a[x] + area_b[y] - c) >= merge_iou:
                    ra, rb = find(int(x) + A['offset']), find(int(y) + B['offset'])
                    parent[max(ra, rb)] = min(ra, rb)
    if parent:
        ids = np.unique(out)
        lut = {int(v): find(int(v)) for v in ids if v}
        flat = out.ravel()
        out = np.vectorize(lambda v: lut.get(int(v), 0), otypes=[np.int64])(flat).reshape(shape) if len(lut) < 5000 else out
    return out


def center_owner(tiles, shape):
    """Keep a raw object only in the tile whose disjoint box contains its centroid; paste it whole."""
    out = np.zeros(shape, np.int64)
    boxes = ownership_boxes(tiles, shape)
    nxt = 1
    for t, box in zip(tiles, boxes):
        raw = t['raw']
        for v in np.unique(raw):
            if not v:
                continue
            pts = np.argwhere(raw == v)
            c = pts.mean(0) + [r[0] for r in t['read']]
            if all(lo <= ci < hi for ci, (lo, hi) in zip(c, box)):
                g = pts + [r[0] for r in t['read']]
                out[tuple(g.T)] = nxt
                nxt += 1
    return out


def errors(ref, pred):
    m = matching(ref, pred, thresh=0.5)
    # splits: ref objects covered >=20% by >=2 pred objects; merges: pred objects covering >=2 ref objects
    split = merge = 0
    for v in np.unique(ref)[1:]:
        p, c = np.unique(pred[ref == v], return_counts=True)
        split += int(np.sum((p > 0) & (c >= 0.2 * c.sum())) >= 2)
    for v in np.unique(pred)[1:]:
        r, c = np.unique(ref[pred == v], return_counts=True)
        merge += int(np.sum((r > 0) & (c >= 0.2 * c.sum())) >= 2)
    return dict(n=int(len(np.unique(pred)) - 1), tp=int(m.tp), fp=int(m.fp), fn=int(m.fn),
                f1=round(float(m.f1), 4), split=split, merge=merge)


def main(n_images=10, configs=((320, 48), (352, 32), (352, 64), (384, 48))):
    model = StarDist2D(None, name='2D_versatile_fluo', basedir=str(R / 'models'))
    files = sorted(Path(R / 'data/bbbc039/metadata/test.txt').read_text().split())[:n_images]
    rows = []
    for f in files:
        img = normalize(tifffile.imread(R / 'data/bbbc039/images' / f.replace('.png', '.tif')), 1, 99.8, axis=(0, 1))
        t0 = time.perf_counter(); ref, _ = model.predict_instances(img); t_untiled = time.perf_counter() - t0
        for b, o in configs:
            dest = OUT / f'{Path(f).stem}-b{b}-o{o}'
            t0 = time.perf_counter()
            try:
                bundle = capture_stardist(model, img, dest, model_id='2D_versatile_fluo-local', block_size=b, min_overlap=o, context=96)
            except Exception as e:
                rows.append(dict(image=f, block=b, overlap=o, status=type(e).__name__)); continue
            t_capture = time.perf_counter() - t0
            tiles = tiles_of(bundle)
            res = dict(image=f, block=b, overlap=o, status='ok', tiles=len(tiles), t_untiled=t_untiled, t_capture=t_capture, policies={})
            for name, fn in [('native', lambda: replay(bundle)[0]), ('retain_all', lambda: replay(bundle, 'retain_all')[0]),
                             ('naive_crop', lambda: naive_crop(tiles_of(bundle), img.shape)),
                             ('iou_merge_0.5', lambda: naive_crop(tiles_of(bundle), img.shape, 0.5)),
                             ('center', lambda: center_owner(tiles_of(bundle), img.shape))]:
                t0 = time.perf_counter(); lab = fn(); dt = time.perf_counter() - t0
                res['policies'][name] = dict(seconds=round(dt, 3), **errors(ref, lab))
            rows.append(res)
            print(json.dumps(res), flush=True)
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
