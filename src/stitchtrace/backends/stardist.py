"""Actual StarDist method, serialized internal hooks, resumable raw tile predictions."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .common import HOOK_LOCK, compatible, contract, bounds
from ..storage import transaction, atomic_array, atomic_bytes, encoded, read_json, digest, commit, ARRAY_PARSE_ERRORS
from ..bundle import Bundle, labels
from ..errors import require, StitchTraceError


def capture_stardist(model, image, destination, *, model_id, block_size=320, min_overlap=64,
                     context=96, spacing=None, allow_unverified=False):
    """model_id must identify actual weights/config; caller owns preprocessing."""
    with HOOK_LOCK:
        source = compatible('stardist', allow_unverified)
        from stardist.big import BlockND
        axes = 'YX' if image.ndim == 2 else 'ZYX'
        spacing = list(spacing or [1.] * image.ndim)
        recommended = model._axes_tile_overlap(axes)
        require(all(context >= v for v in recommended), 'CONTEXT_TOO_SMALL', str(recommended))
        params = {'block_size': block_size, 'min_overlap': min_overlap, 'context': context,
                  'axes': axes, 'spacing': spacing, 'grid': list(map(int, model._axes_div_by(axes)))}
        # Caller label alone cannot safely invalidate a cache when weights change.
        import json
        weights = [{'shape': list(w.shape), 'dtype': w.dtype.str, 'sha256': digest(w.tobytes())}
                   for w in model.keras_model.get_weights()]
        config = json.loads(json.dumps(vars(model.config), default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist()))
        model_state = {'caller_id': model_id, 'weights': weights, 'config': config,
                       'thresholds': dict(model.thresholds._asdict())}
        identity = contract(image, params, source, model_state)
        with transaction(destination, identity) as root:
            if root is None:
                return Bundle.open(destination)
            tiles, active = [], {}
            read, filt, write, predict = BlockND.read, BlockND.filter_objects, BlockND.write, model.predict_instances
            def wrapped_read(block, value, axes=None):
                result = read(block, value, axes)
                active['block'] = block
                return result
            def wrapped_predict(*args, **kwargs):
                import io
                block = active['block']
                key = digest(encoded({'run': identity, 'read': bounds(block.slice_read(axes))}))
                path = root / 'cache' / (key + '.npz')
                if path.exists():
                    try:
                        with np.load(path, allow_pickle=False) as data:
                            arrays = {k: data[k].copy() for k in data.files if not k.startswith('hash_')}
                            for k, value in arrays.items():
                                require(digest(value.tobytes()) == str(data['hash_' + k].item()), 'CACHE_CORRUPT', key)
                        result = (arrays.pop('labels'), {k[5:]: v for k, v in arrays.items()})
                    except StitchTraceError:
                        raise
                    except ARRAY_PARSE_ERRORS as e:
                        raise StitchTraceError('CACHE_CORRUPT', key) from e
                else:
                    result = predict(*args, **kwargs)
                    arrays = {'labels': result[0], **{'poly_' + k: v for k, v in result[1].items()}}
                    require(all(not np.asarray(v).dtype.hasobject for v in arrays.values()), 'INVALID_ARRAY', 'polygons')
                    stream = io.BytesIO()
                    np.savez_compressed(stream, **arrays, **{'hash_' + k: np.array(digest(np.asarray(v).tobytes())) for k, v in arrays.items()})
                    atomic_bytes(path, stream.getvalue())
                active['cache'] = str(path.relative_to(root))
                return result
            def wrapped_filter(block, local, polys, axes=None):
                before = labels(local).copy()
                result = filt(block, local, polys, axes)
                tile = {'id': int(block.id), 'read': bounds(block.slice_read(axes)), 'write': bounds(block.slice_write(axes)),
                        'cache': active['cache'], 'local': f'tiles/{block.id}-local.npy',
                        'selected': f'tiles/{block.id}-selected.npy', 'written': f'tiles/{block.id}-written.npy'}
                tile['kept'] = sorted(int(i) for i in np.unique(result[0]) if i)
                tile['rejected'] = sorted(set(int(i) for i in np.unique(before) if i) - set(tile['kept']))
                atomic_array(root / tile['local'], before)
                atomic_array(root / tile['selected'], result[0])
                tiles.append(tile)
                return result
            def wrapped_write(block, target, local, axes=None):
                atomic_array(root / tiles[-1]['written'], local)
                return write(block, target, local, axes)
            with patch.object(BlockND, 'read', wrapped_read), patch.object(BlockND, 'filter_objects', wrapped_filter), patch.object(BlockND, 'write', wrapped_write), patch.object(model, 'predict_instances', wrapped_predict):
                output, polygons = model.predict_instances_big(image, axes=axes, block_size=block_size,
                    min_overlap=min_overlap, context=context, show_progress=False)
            atomic_array(root / 'expected.npy', output)
            commit(root, {'schema': 1, 'backend': 'ownership', 'contract': identity, 'shape': list(image.shape),
                          'axes': axes, 'spacing': spacing, 'parameters': params, 'source': source,
                          'predictor_id': model_id, 'model_state': model_state, 'tiles': tiles, 'order': [t['id'] for t in tiles]})
        return Bundle.open(destination)
