"""StarDist arm of the v1 protocol: untiled reference, captures per configuration, policy replays."""
import sys, time
import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from stitchtrace.backends.stardist import capture_stardist
from stitchtrace.compare import compare
from common import ROOT, RUNS, test_images, load, evaluate, bundle_bytes, slim, write

CONFIGS = [(320, 48), (352, 32), (352, 48), (352, 64), (384, 48), (384, 64)]
CONTEXT = 96
MODEL_ID = '2D_versatile_fluo-local;normalize-1-99.8'


def main(limit=None):
    model = StarDist2D(None, name='2D_versatile_fluo', basedir=str(ROOT / 'models'))
    out_dir = RUNS / 'stardist'
    for stem in test_images()[:limit]:
        result_path = out_dir / 'results' / f'{stem}.json'
        if result_path.exists():
            continue
        raw, truth = load(stem)
        image = normalize(raw, 1, 99.8, axis=(0, 1))
        t0 = time.perf_counter(); untiled, _ = model.predict_instances(image); t_untiled = time.perf_counter() - t0
        (out_dir / 'untiled').mkdir(parents=True, exist_ok=True)
        np.save(out_dir / 'untiled' / f'{stem}.npy', untiled)
        record = {'image': stem, 'untiled_seconds': t_untiled, 'untiled_vs_truth': slim(compare(truth, untiled)), 'configs': []}
        for block, overlap in CONFIGS:
            entry = {'block_size': block, 'min_overlap': overlap, 'context': CONTEXT}
            dest = out_dir / 'bundles' / f'{stem}-b{block}-o{overlap}'
            try:
                t0 = time.perf_counter()
                bundle = capture_stardist(model, image, dest, model_id=MODEL_ID, block_size=block,
                                          min_overlap=overlap, context=CONTEXT)
                entry['capture_seconds'] = time.perf_counter() - t0
            except Exception as e:  # StarDist refusals are outcomes, recorded verbatim
                cause = e.__cause__ or e.__context__ or e
                entry.update(status='refused', error=type(e).__name__, cause=f'{type(cause).__name__}: {str(cause)[:200]}')
                record['configs'].append(entry)
                continue
            t0 = time.perf_counter()
            plain, _ = model.predict_instances_big(image, axes='YX', block_size=block, min_overlap=overlap,
                                                   context=CONTEXT, show_progress=False)
            entry['plain_seconds'] = time.perf_counter() - t0
            entry['instrumentation_pixel_equal'] = bool(np.array_equal(plain, bundle.array('expected.npy')))
            entry.update(status='ok', tiles=len(bundle.meta['tiles']), bundle_bytes=bundle_bytes(dest),
                         policies=evaluate(bundle, untiled, truth))
            record['configs'].append(entry)
        write(result_path, record)
        print(stem, [(c['block_size'], c['min_overlap'], c['status']) for c in record['configs']], flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
