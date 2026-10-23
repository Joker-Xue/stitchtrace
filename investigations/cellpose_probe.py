"""Probe C: real Cellpose-SAM inference through the native distributed driver vs untiled inference."""
import json, sys, time
from pathlib import Path
import numpy as np, tifffile
from cellpose import models
from stardist.matching import matching
from stitchtrace.backends.cellpose import capture_cellpose
from stitchtrace.replay import replay
sys.path.insert(0, str(Path(__file__).parent))
from policy_probe import errors

R = Path(__file__).resolve().parents[1]
OUT = R / 'runs' / 'cellpose_probe'
model = models.CellposeModel(gpu=True)


def predict(img):
    masks, *_ = model.eval(img, diameter=None, normalize={'percentile': [1, 99.8]})
    return masks.astype(np.uint32)


def main(n_images=2, blocks=((256, 256), (128, 128)), diameter=30):
    files = sorted((R / 'data/bbbc039/metadata/test.txt').read_text().split())[:n_images]
    rows = []
    for f in files:
        img = tifffile.imread(R / 'data/bbbc039/images' / f.replace('.png', '.tif')).astype(np.float32)
        t0 = time.perf_counter(); ref = predict(img); t_untiled = time.perf_counter() - t0
        for b in blocks:
            dest = OUT / f'{Path(f).stem}-b{b[0]}'
            calls = []
            def tile(crop_img, crop):
                calls.append(1); return predict(crop_img)
            t0 = time.perf_counter()
            bundle = capture_cellpose(img, tile, dest, predictor_id='cpsam_v2-percentile-1-99.8', block_size=list(b), diameter=diameter)
            t_capture = time.perf_counter() - t0
            res = dict(image=f, block=b, tiles=len(calls), t_untiled=round(t_untiled, 2), t_capture=round(t_capture, 2), policies={})
            for pol in ['native', 'disable_edges']:
                t0 = time.perf_counter(); lab = replay(bundle, pol)[0]; dt = time.perf_counter() - t0
                res['policies'][pol] = dict(seconds=round(dt, 3), **errors(ref, lab))
            res['edges'] = len(bundle.meta['edges'])
            rows.append(res); print(json.dumps(res), flush=True)
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
