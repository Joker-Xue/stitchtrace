"""Addendum A1 (exploratory): Cellpose with one global normalisation instead of per-crop normalisation."""
import sys, time
import numpy as np
from cellpose import models
from stitchtrace.backends.cellpose import capture_cellpose
from stitchtrace.compare import compare
from common import RUNS, test_images, load, evaluate, bundle_bytes, slim, write

BLOCK, DIAMETER = 256, 30
PREDICTOR_ID = 'cpsam_v2;global-percentile-1-99.8;normalize-False;diameter-None'


def global_normalize(raw):
    lo, hi = np.percentile(raw, [1, 99.8])
    return ((raw.astype(np.float32) - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def main(limit=None):
    model = models.CellposeModel(gpu=True)

    def predict(img):
        return model.eval(np.array(img), diameter=None, normalize=False)[0].astype(np.uint32)

    out_dir = RUNS / 'cellpose_globalnorm'
    for stem in test_images()[:limit]:
        result_path = out_dir / 'results' / f'{stem}.json'
        if result_path.exists():
            continue
        raw, truth = load(stem)
        image = global_normalize(raw)
        t0 = time.perf_counter(); untiled = predict(image); t_untiled = time.perf_counter() - t0
        (out_dir / 'untiled').mkdir(parents=True, exist_ok=True)
        np.save(out_dir / 'untiled' / f'{stem}.npy', untiled)
        dest = out_dir / 'bundles' / f'{stem}-b{BLOCK}'
        t0 = time.perf_counter()
        bundle = capture_cellpose(image, lambda c, s: predict(c), dest, predictor_id=PREDICTOR_ID,
                                  block_size=[BLOCK, BLOCK], diameter=DIAMETER)
        record = {'image': stem, 'untiled_seconds': t_untiled, 'untiled_vs_truth': slim(compare(truth, untiled)),
                  'blocks': [{'block_size': BLOCK, 'capture_seconds': time.perf_counter() - t0,
                              'blocks': int(np.prod([-(-n // BLOCK) for n in image.shape])),
                              'recorded_edges': len(bundle.meta['edges']), 'bundle_bytes': bundle_bytes(dest),
                              'policies': evaluate(bundle, untiled, truth)}]}
        write(result_path, record)
        p = record['blocks'][0]['policies']
        print(stem, p['native']['attribution_vs_untiled'], p['relink']['attribution_vs_untiled'], flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
