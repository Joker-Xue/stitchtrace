"""Versioned bundle reader, semantic validation and backend-neutral measures."""
from pathlib import Path
import numpy as np
from .storage import read_json, safe_path, file_digest, inventory, ARRAY_PARSE_ERRORS
from .errors import require, StitchTraceError


def labels(value):
    a = np.asarray(value)
    require(a.ndim in (2, 3) and a.dtype.kind in 'iu' and np.all(a >= 0), 'INVALID_LABELS', 'nonnegative integer 2D/3D required')
    require(a.size <= 64*1024**2 and (not a.size or int(a.max()) <= np.iinfo(np.int64).max), 'RESOURCE_LIMIT', 'label size/range')
    return a


def measurements(value, spacing=None):
    a = labels(value)
    spacing = np.asarray(spacing if spacing is not None else np.ones(a.ndim), float)
    require(spacing.shape == (a.ndim,) and np.all(np.isfinite(spacing)) and np.all(spacing > 0), 'INVALID_SPACING', str(spacing))
    ids, inverse = np.unique(a.ravel(), return_inverse=True)
    counts = np.bincount(inverse, minlength=len(ids))
    coords = np.indices(a.shape).reshape(a.ndim, -1)
    sums = np.stack([np.bincount(inverse, weights=c, minlength=len(ids)) for c in coords], 1).round().astype(np.int64)
    result = {}
    for k, i in enumerate(ids):
        if i:
            n = int(counts[k])
            result[str(int(i))] = {'pixels': n, 'physical_size': float(n * spacing.prod()),
                                  'centroid': (sums[k] / n * spacing).tolist(), 'coordinate_sum': sums[k].tolist()}
    return result


def canonical(value):
    """Relabel by first appearance in C order; equal partitions give equal arrays."""
    a = labels(value)
    ids, first, inverse = np.unique(a.ravel(), return_index=True, return_inverse=True)
    rank = np.zeros(len(ids), np.int64)
    nonzero = ids != 0
    rank[np.flatnonzero(nonzero)[np.argsort(first[nonzero], kind='stable')]] = np.arange(1, nonzero.sum() + 1)
    return rank[inverse].reshape(a.shape)


class Bundle:
    """Verified immutable schema1 bundle. Loading never imports a model backend."""
    def __init__(self, path, meta, manifest):
        self.path, self.meta, self.manifest = Path(path), meta, manifest

    @classmethod
    def open(cls, path):
        try:
            return cls._open(path)
        except StitchTraceError:
            raise
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as e:
            raise StitchTraceError('INVALID_BUNDLE', str(e)) from e

    @classmethod
    def _open(cls, path):
        root = Path(path).absolute()
        safe_path(root.parent, root.name)
        manifest_path = safe_path(root, 'manifest.json')
        actual = set(inventory(root))
        require(manifest_path.is_file(), 'INVALID_BUNDLE', 'manifest absent or nonregular')
        manifest = read_json(manifest_path)
        require(isinstance(manifest, dict), 'INVALID_MANIFEST', 'object required')
        require(manifest.get('schema') == 1, 'SCHEMA_UNSUPPORTED', str(manifest.get('schema')))
        require(isinstance(manifest.get('files'), dict), 'INVALID_MANIFEST', 'files')
        require('bundle.json' in manifest['files'], 'MISSING_DEPENDENCY', 'bundle.json')
        for name, spec in manifest['files'].items():
            require(isinstance(spec, dict) and type(spec.get('bytes')) is int and isinstance(spec.get('sha256'), str), 'INVALID_MANIFEST', name)
            p = safe_path(root, name)
            require(p.is_file(), 'MISSING_DEPENDENCY', name)
            require(p.stat().st_size <= 1024**3 and p.stat().st_size == spec['bytes'], 'SIZE_MISMATCH', name)
            require(file_digest(p) == spec['sha256'], 'HASH_MISMATCH', name)
        require(actual == set(manifest['files']), 'UNDECLARED_FILE', str(actual.symmetric_difference(manifest['files'])))
        meta = read_json(root / 'bundle.json')
        require(isinstance(meta, dict), 'INVALID_METADATA', 'object required')
        require(meta.get('schema') == 1 and meta.get('backend') in ('ownership', 'graph'), 'SCHEMA_UNSUPPORTED', 'backend/schema')
        obj = cls(root, meta, manifest)
        obj.validate()
        return obj

    def array(self, name):
        require(name in self.manifest['files'], 'MISSING_DEPENDENCY', name)
        try:
            return labels(np.load(safe_path(self.path, name), allow_pickle=False, mmap_mode='r'))
        except ARRAY_PARSE_ERRORS as e:
            raise StitchTraceError('INVALID_ARRAY', name) from e

    def cache_labels(self, name):
        """Raw tile prediction labels from a recorded cache (NPZ with ``labels`` or plain NPY)."""
        require(name in self.manifest['files'], 'MISSING_DEPENDENCY', str(name))
        try:
            cached = np.load(safe_path(self.path, name), allow_pickle=False)
            if isinstance(cached, np.lib.npyio.NpzFile):
                with cached:
                    return labels(cached['labels'] if 'labels' in cached.files else cached['value'])
            return labels(cached)
        except ARRAY_PARSE_ERRORS as e:
            raise StitchTraceError('INVALID_CACHE', str(name)) from e

    def validate(self):
        m = self.meta
        require(isinstance(m.get('contract'), str) and bool(m['contract']), 'INVALID_CONTRACT', 'nonempty run identity required')
        shape = m.get('shape', [])
        require(isinstance(shape, list) and len(shape) in (2, 3) and all(type(n) is int and 0 < n <= 100000 for n in shape), 'INVALID_SHAPE', str(shape))
        require(int(np.prod(shape, dtype=np.int64)) <= 64*1024**2, 'RESOURCE_LIMIT', 'shape exceeds 64M pixels')
        require(isinstance(m.get('axes'), str) and len(m.get('axes', '')) == len(shape) and len(set(m['axes'])) == len(shape), 'INVALID_AXES', 'axes')
        require(isinstance(m.get('spacing'), list) and len(m['spacing']) == len(shape), 'INVALID_SPACING', 'spacing required')
        measurements(np.zeros((1,) * len(shape), np.uint8), m['spacing'])
        expected = self.array('expected.npy')
        require(tuple(shape) == expected.shape, 'INVALID_SHAPE', 'expected')
        if m['backend'] == 'graph':
            packed = self.array('packed.npy')
            require(packed.shape == expected.shape, 'INVALID_SHAPE', 'packed')
            nodes = m.get('nodes', [])
            require(isinstance(nodes, list) and all(type(i) is int and i > 0 for i in nodes), 'GRAPH_CLOSURE', 'positive integer node IDs required')
            require(set(map(int, np.unique(packed))) - {0} == set(nodes) and len(set(nodes)) == len(nodes), 'GRAPH_CLOSURE', 'nodes differ from full packed label set')
            require(isinstance(m.get('edges'), list) and all(isinstance(e, list) and len(e) == 2 and all(v in nodes for v in e) for e in m.get('edges', [])), 'GRAPH_CLOSURE', 'edge endpoint missing')
        else:
            tiles = m.get('tiles', [])
            require(isinstance(tiles, list) and all(isinstance(t, dict) and type(t.get('id')) is int for t in tiles) and isinstance(m.get('order'), list) and all(type(i) is int for i in m['order']) and len(tiles) > 0 and [t['id'] for t in tiles] == m.get('order') and len(set(m['order'])) == len(tiles), 'INCOMPLETE_COVER', 'tile order')
            cover = np.zeros(shape, bool)
            for t in tiles:
                require(t.get('cache') in self.manifest['files'], 'MISSING_DEPENDENCY', str(t.get('cache')))
                bounds, read = t.get('write'), t.get('read')
                for domain in (bounds, read):
                    require(isinstance(domain, list) and len(domain) == len(shape) and all(isinstance(b, list) and len(b) == 2 and type(b[0]) is int and type(b[1]) is int and 0 <= b[0] < b[1] <= n for b, n in zip(domain, shape)), 'INVALID_BOUNDS', str(domain))
                require(all(r[0] <= w[0] < w[1] <= r[1] for r,w in zip(read,bounds)), 'INVALID_BOUNDS', 'write outside read')
                raw = self.cache_labels(t['cache'])
                require(raw.shape == tuple(b[1]-b[0] for b in read), 'INVALID_SHAPE', 'raw read extent')
                local, selected, written = (self.array(t[k]) for k in ('local', 'selected', 'written'))
                require(local.shape == selected.shape == written.shape == tuple(b[1]-b[0] for b in bounds), 'INVALID_SHAPE', 'tile')
                crop = tuple(slice(w[0]-r[0], w[1]-r[0]) for r,w in zip(read,bounds))
                require(np.array_equal(raw[crop], local), 'INVALID_DECISION', 'raw crop differs from local')
                require(np.all((selected == 0) | (selected == local)), 'INVALID_DECISION', 'selected is not a local-label subset')
                require(np.array_equal(selected > 0, written > 0), 'INVALID_DECISION', 'write foreground differs')
                kept = sorted(int(i) for i in np.unique(selected) if i)
                rejected = sorted(set(int(i) for i in np.unique(local) if i) - set(kept))
                require(kept == t['kept'] and rejected == t['rejected'], 'INVALID_DECISION', 'IDs')
                require(np.array_equal(selected, np.where(np.isin(local, kept), local, 0)), 'INVALID_DECISION', 'partial object selection')
                for i in kept:
                    require(len(np.unique(written[selected == i])) == 1, 'INVALID_DECISION', 'local ID split during write')
                cover[tuple(slice(*b) for b in bounds)] = True
            require(np.all(cover), 'INCOMPLETE_COVER', 'write domains do not cover shape')
