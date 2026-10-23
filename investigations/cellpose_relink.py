"""Probe D: re-link recorded Cellpose packed tiles with a dimension-correct face shrink, no inference.

The recorded packed core and block size are enough to rebuild every block face; the corrected
shrink applies the 1D gradient per pixel (as proposed upstream in MouseLand/cellpose#1480).
"""
import json, sys
from pathlib import Path
import numpy as np, scipy.ndimage as ndi, tifffile
from stitchtrace.bundle import Bundle, canonical
sys.path.insert(0, str(Path(__file__).parent))
from policy_probe import errors

R = Path(__file__).resolve().parents[1]


def shrink(line, threshold=1.0):
    line = line.copy()
    g = np.abs(np.gradient(line.astype(float)))
    line[g > 0] = 0
    line[ndi.distance_transform_edt(line) <= threshold] = 0
    return line


def relink(packed, block):
    edges = set()
    for ax, b in enumerate(block):
        for cut in range(b, packed.shape[ax], b):
            left = np.take(packed, cut - 1, axis=ax); right = np.take(packed, cut, axis=ax)
            a, c = shrink(left), shrink(right)
            for x, y in zip(a, c):
                if x and y:
                    edges.add((int(x), int(y)))
    parent = {}
    def find(i):
        parent.setdefault(i, i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for x, y in edges:
        rx, ry = find(x), find(y); parent[max(rx, ry)] = min(rx, ry)
    ids = np.unique(packed)
    lut = np.array([find(int(v)) if v else 0 for v in ids])
    return canonical(lut[np.searchsorted(ids, packed)]), len(edges)


if __name__ == '__main__':
    from cellpose import models
    model = models.CellposeModel(gpu=True)
    rows = []
    for d in sorted((R / 'runs/cellpose_probe').glob('IXM*-b*')):
        if not d.is_dir():
            continue
        b = Bundle.open(d)
        stem = d.name.rsplit('-b', 1)[0]
        img = tifffile.imread(R / 'data/bbbc039/images' / (stem + '.tif')).astype(np.float32)
        ref = model.eval(img, diameter=None, normalize={'percentile': [1, 99.8]})[0]
        fixed, n_edges = relink(b.array('packed.npy'), b.meta['parameters']['block_size'])
        native = b.array('expected.npy')
        row = dict(bundle=d.name, recorded_edges=len(b.meta['edges']), relinked_edges=n_edges,
                   native=errors(ref, native), relinked=errors(ref, fixed))
        rows.append(row); print(json.dumps(row), flush=True)
    (R / 'runs/cellpose_probe/relink.json').write_text(json.dumps(rows, indent=1))
