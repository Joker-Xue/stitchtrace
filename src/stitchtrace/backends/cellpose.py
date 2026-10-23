"""Native distributed driver; explicit user predictor avoids implicit model downloads."""
from pathlib import Path
import shutil
from contextlib import chdir
from unittest.mock import patch
import numpy as np
from .common import HOOK_LOCK, compatible, contract, bounds
from ..storage import transaction, cache_array, atomic_array, commit
from ..bundle import Bundle, labels
from ..errors import require


def capture_cellpose(image, predict_tile, destination, *, predictor_id, block_size, diameter=2, spacing=None,
                     allow_unverified=False):
    """predict_tile(crop_array, crop_slices) returns local labels. No implicit model creation.

    Each call receives a private copy of the crop: some predictors normalise their input in place
    (Cellpose 4.2.1.1 does so for float32 arrays), which would otherwise rewrite the caller's image
    and the overlapping crops of neighbouring blocks.
    """
    with HOOK_LOCK:
        source = compatible('cellpose', allow_unverified)
        from cellpose.contrib import distributed_segmentation as cp
        import distributed
        import zarr
        source['global_segment_ids'] = __import__('inspect').signature(cp.global_segment_ids).__str__()
        spacing = list(spacing or [1.] * image.ndim)
        require(len(block_size) == image.ndim and all(type(n) is int and n > 0 for n in block_size), 'INVALID_BLOCK', str(block_size))
        params = {'block_size': list(block_size), 'diameter': diameter, 'spacing': spacing}
        identity = contract(image, params, source, predictor_id)
        with transaction(destination, identity) as root:
            if root is None:
                return Bundle.open(destination)
            scratch = root / 'scratch'
            scratch.mkdir(exist_ok=True)
            z = zarr.open(str(scratch / 'input.zarr'), mode='w', shape=image.shape, chunks=block_size, dtype=image.dtype)
            z[:] = image
            native_graph, native_merge, native_ids = cp.block_face_adjacency_graph, cp.determine_merge_relabeling, cp.global_segment_ids
            state = {}
            def provider(input_zarr, crop, *args, **kwargs):
                result, hit = cache_array(root, {'run': identity, 'crop': bounds(crop)},
                                          lambda: labels(predict_tile(np.array(image[crop]), crop)))
                return result.copy()
            def graph(faces, nlabels):
                result = native_graph(faces, nlabels)
                coo = result.tocoo()
                state['edges'] = list(map(list, zip(coo.row.tolist(), coo.col.tolist())))
                return result
            def ids(local, block_index, nblocks):
                result = native_ids(local, block_index, nblocks)
                name = '-'.join(map(str, block_index))
                atomic_array(root / ('tiles/' + name + '-local.npy'), local)
                atomic_array(root / ('tiles/' + name + '-packed.npy'), result[0])
                return result
            def merge(*args, **kwargs):
                temp = next(scratch.glob('*_distributed_cellpose_tempdir/segmentation_unstitched.zarr'))
                packed = np.asarray(zarr.open(str(temp), mode='r'))
                atomic_array(root / 'packed.npy', packed)
                result = native_merge(*args, **kwargs)
                atomic_array(root / 'native-mapping.npy', result.reshape(1, -1))
                state['nodes'] = sorted(int(i) for i in np.unique(packed) if i)
                return result
            with chdir(scratch), patch.object(cp, 'read_preprocess_and_segment', provider), patch.object(cp, 'block_face_adjacency_graph', graph), patch.object(cp, 'determine_merge_relabeling', merge), patch.object(cp, 'global_segment_ids', ids):
                with distributed.LocalCluster(n_workers=2, threads_per_worker=1, processes=False, host='127.0.0.1', protocol='tcp', dashboard_address=None, memory_limit=0, silence_logs=40) as cluster:
                    with distributed.Client(cluster) as client:
                        cluster.client = client
                        out, boxes = cp.distributed_eval(z, block_size, str(scratch / 'output.zarr'), cluster=cluster,
                            eval_kwargs={'diameter': diameter}, temporary_directory=str(scratch))
                        atomic_array(root / 'expected.npy', np.asarray(out))
            shutil.rmtree(scratch)
            commit(root, {'schema': 1, 'backend': 'graph', 'contract': identity, 'shape': list(image.shape),
                          'axes': 'YX' if image.ndim == 2 else 'ZYX', 'spacing': spacing, 'source': source,
                          'predictor_id': predictor_id, 'parameters': params, **state})
        return Bundle.open(destination)
