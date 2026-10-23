"""Cellpose arm of the v1 protocol: untiled reference, distributed captures per block size, replays."""
import sys, time
import numpy as np
from cellpose import models
from stitchtrace.backends.cellpose import capture_cellpose
from stitchtrace.compare import compare
from common import RUNS, test_images, load, evaluate, bundle_bytes, slim, write

BLOCKS = [128, 192, 256]
DIAMETER = 30                      # sets the distributed overlap only (2 x 30 px)
PREDICTOR_ID = 'cpsam_v2;normalize-percentile-1-99.8;diameter-None'
FIDELITY_IMAGES = 10               # plain distributed_eval comparison at block 256
REPEAT_IMAGES = 10                 # untiled nondeterminism check


def main(limit=None):
    model = models.CellposeModel(gpu=True)

    def predict(img):
        # Cellpose normalises float32 input in place; never hand it a view of shared data
        return model.eval(np.array(img), diameter=None, normalize={'percentile': [1, 99.8]})[0].astype(np.uint32)

    out_dir = RUNS / 'cellpose'
    (out_dir / 'untiled').mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(test_images()[:limit]):
        result_path = out_dir / 'results' / f'{stem}.json'
        if result_path.exists():
            continue
        raw, truth = load(stem)
        image = raw.astype(np.float32)
        t0 = time.perf_counter(); untiled = predict(image); t_untiled = time.perf_counter() - t0
        np.save(out_dir / 'untiled' / f'{stem}.npy', untiled)
        record = {'image': stem, 'untiled_seconds': t_untiled, 'untiled_vs_truth': slim(compare(truth, untiled)), 'blocks': []}
        if index < REPEAT_IMAGES:
            t0 = time.perf_counter(); again = predict(image)
            record['untiled_repeat'] = {'seconds': time.perf_counter() - t0, 'pixel_equal': bool(np.array_equal(again, untiled)),
                                        'vs_first': slim(compare(untiled, again))}
        for block in BLOCKS:
            dest = out_dir / 'bundles' / f'{stem}-b{block}'
            calls = []
            def tile(crop_image, crop):
                calls.append(1)
                return predict(crop_image)
            t0 = time.perf_counter()
            bundle = capture_cellpose(image, tile, dest, predictor_id=PREDICTOR_ID, block_size=[block, block], diameter=DIAMETER)
            entry = {'block_size': block, 'capture_seconds': time.perf_counter() - t0, 'predictor_calls': len(calls),
                     'blocks': int(np.prod([-(-n // block) for n in image.shape])), 'recorded_edges': len(bundle.meta['edges']),
                     'bundle_bytes': bundle_bytes(dest)}
            if block == 256 and index < FIDELITY_IMAGES:
                entry['instrumentation_pixel_equal'] = plain_distributed_equal(image, predict, block, bundle)
            entry['policies'] = evaluate(bundle, untiled, truth)
            record['blocks'].append(entry)
        write(result_path, record)
        print(stem, [(b['block_size'], b['policies']['native']['vs_untiled']['f1'], b['policies']['relink']['vs_untiled']['f1'])
                     for b in record['blocks']], flush=True)


def plain_distributed_equal(image, predict, block, bundle):
    """Run the unmodified distributed driver with the same predictor and compare to the capture."""
    import tempfile, zarr, distributed
    from contextlib import chdir
    from unittest.mock import patch
    from cellpose.contrib import distributed_segmentation as cp
    with tempfile.TemporaryDirectory() as tmp:
        z = zarr.open(f'{tmp}/in.zarr', mode='w', shape=image.shape, chunks=(block, block), dtype=image.dtype)
        z[:] = image
        def provider(input_zarr, crop, *args, **kwargs):
            return predict(image[crop])
        with chdir(tmp), patch.object(cp, 'read_preprocess_and_segment', provider):
            with distributed.LocalCluster(n_workers=2, threads_per_worker=1, processes=False, host='127.0.0.1', protocol='tcp',
                                          dashboard_address=None, memory_limit=0, silence_logs=40) as cluster:
                with distributed.Client(cluster) as client:
                    cluster.client = client
                    out, _ = cp.distributed_eval(z, (block, block), f'{tmp}/out.zarr', cluster=cluster,
                                                 eval_kwargs={'diameter': DIAMETER}, temporary_directory=tmp)
                    out = np.asarray(out)
    return bool(np.array_equal(out, bundle.array('expected.npy')))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
