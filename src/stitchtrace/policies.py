"""Assembly policies replayed from recorded tile predictions; none imports an inference backend.

A policy is ``fn(bundle, **params) -> label array`` registered for one backend. Built-in policies:

ownership (StarDist block records)
  native      recorded native writes in original order
  retain_all  every context-cropped candidate, original last-write order (diagnostic baseline)
  crop        raw tile labels cut at the midline of each write overlap (generic crop-and-paste)
  iou_merge   ``crop`` followed by linking pieces whose raw objects overlap with IoU >= ``threshold``
  center      each raw object pasted whole by the single tile whose midline box holds its centroid

graph (Cellpose distributed records)
  native        recorded face-adjacency edges
  disable_edges no cross-block linking
  relink        edges recomputed from the packed core with a dimension-correct face shrink
"""
import numpy as np
from .bundle import canonical
from .errors import require

_REGISTRY = {'ownership': {}, 'graph': {}}


def register_policy(backend, name, fn=None):
    """Register ``fn`` (or use as a decorator). Re-registering a name replaces it."""
    require(backend in _REGISTRY, 'POLICY_UNSUPPORTED', str(backend))
    def add(f):
        _REGISTRY[backend][name] = f
        return f
    return add(fn) if fn is not None else add


def available_policies(backend=None):
    if backend is None:
        return {k: sorted(v) for k, v in _REGISTRY.items()}
    return sorted(_REGISTRY[backend])


def get_policy(backend, name):
    require(backend in _REGISTRY and name in _REGISTRY[backend], 'POLICY_UNSUPPORTED', f'{backend}:{name}')
    return _REGISTRY[backend][name]


def _relabel(values, lut_keys, lut_values):
    """Map integer array through sorted ``lut_keys`` -> ``lut_values``; unknown values become 0."""
    flat = values.ravel()
    if not len(lut_keys):
        return np.zeros(values.shape, np.int64)
    idx = np.searchsorted(lut_keys, flat)
    idx = np.clip(idx, 0, len(lut_keys) - 1)
    hit = lut_keys[idx] == flat
    out = np.where(hit, lut_values[idx], 0)
    return out.reshape(values.shape)


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, i):
        self.parent.setdefault(i, i)
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _apply_union(out, uf):
    ids = np.unique(out)
    ids = ids[ids > 0]
    if not len(ids):
        return out
    return _relabel(out, ids, np.array([uf.find(int(i)) for i in ids], np.int64))


# ---------------------------------------------------------------- ownership

def _tiles(bundle):
    """Raw tile predictions (read region, including context) with their bounds."""
    out = []
    for t in bundle.meta['tiles']:
        raw = bundle.cache_labels(t['cache'])
        out.append({'id': t['id'], 'read': t['read'], 'write': t['write'], 'raw': raw})
    return out


def ownership_boxes(tiles):
    """Disjoint boxes partitioning the image: each write overlap is split at its midline."""
    boxes = []
    for t in tiles:
        box = []
        for ax, (lo, hi) in enumerate(t['write']):
            a, b = lo, hi
            for olo, ohi in {tuple(u['write'][ax]) for u in tiles}:
                if olo < lo < ohi <= hi:
                    a = max(a, (lo + ohi) // 2)
                if lo <= olo < hi < ohi:
                    b = min(b, (olo + hi) // 2)
            box.append((a, b))
        boxes.append(box)
    return boxes


def _local(t, box):
    return tuple(slice(lo - r[0], hi - r[0]) for (lo, hi), r in zip(box, t['read']))


def _global(box):
    return tuple(slice(lo, hi) for lo, hi in box)


@register_policy('ownership', 'native')
def _ownership_native(bundle):
    output = np.zeros(bundle.meta['shape'], np.int64)
    for tile in bundle.meta['tiles']:
        local = bundle.array(tile['written'])
        view = output[_global(tile['write'])]
        view[local > 0] = local[local > 0]
    return output


@register_policy('ownership', 'retain_all')
def _ownership_retain_all(bundle):
    output = np.zeros(bundle.meta['shape'], np.int64)
    offset = 0
    for tile in bundle.meta['tiles']:
        original = bundle.array(tile['local']).astype(np.int64)
        ids = np.unique(original)
        ids = ids[ids > 0]
        local = _relabel(original, ids, np.arange(offset + 1, offset + 1 + len(ids)))
        offset += len(ids)
        view = output[_global(tile['write'])]
        view[local > 0] = local[local > 0]
    return output


def _crop_pieces(bundle):
    tiles = _tiles(bundle)
    boxes = ownership_boxes(tiles)
    output = np.zeros(bundle.meta['shape'], np.int64)
    offset = 0
    for t, box in zip(tiles, boxes):
        piece = t['raw'][_local(t, box)].astype(np.int64)
        output[_global(box)] = np.where(piece > 0, piece + offset, 0)
        t['offset'] = offset
        offset += int(t['raw'].max(initial=0))
    return tiles, output


@register_policy('ownership', 'crop')
def _ownership_crop(bundle):
    return _crop_pieces(bundle)[1]


@register_policy('ownership', 'iou_merge')
def _ownership_iou_merge(bundle, threshold=0.5):
    require(0 < float(threshold) <= 1, 'INVALID_PARAMETER', 'threshold must be in (0, 1]')
    tiles, output = _crop_pieces(bundle)
    uf = _UnionFind()
    for i, A in enumerate(tiles):
        for B in tiles[i + 1:]:
            inter = [(max(a[0], b[0]), min(a[1], b[1])) for a, b in zip(A['write'], B['write'])]
            if any(lo >= hi for lo, hi in inter):
                continue
            la, lb = A['raw'][_local(A, inter)].ravel(), B['raw'][_local(B, inter)].ravel()
            ids_a, area_a = np.unique(la, return_counts=True)
            ids_b, area_b = np.unique(lb, return_counts=True)
            area_a, area_b = dict(zip(ids_a.tolist(), area_a.tolist())), dict(zip(ids_b.tolist(), area_b.tolist()))
            pairs, counts = np.unique(np.stack([la, lb]), axis=1, return_counts=True)
            for (x, y), c in zip(pairs.T.tolist(), counts.tolist()):
                if x and y and c / (area_a[x] + area_b[y] - c) >= float(threshold):
                    uf.union(x + A['offset'], y + B['offset'])
    return _apply_union(output, uf)


@register_policy('ownership', 'center')
def _ownership_center(bundle):
    tiles = _tiles(bundle)
    output = np.zeros(bundle.meta['shape'], np.int64)
    nxt = 1
    for t, box in zip(tiles, ownership_boxes(tiles)):
        raw = t['raw']
        origin = np.array([r[0] for r in t['read']])
        ids, inverse = np.unique(raw.ravel(), return_inverse=True)
        coords = np.indices(raw.shape).reshape(raw.ndim, -1)
        counts = np.bincount(inverse)
        centroids = np.stack([np.bincount(inverse, weights=c) for c in coords], 1) / counts[:, None] + origin
        keep = np.array([i != 0 and all(lo <= c < hi for c, (lo, hi) in zip(centroids[k], box))
                         for k, i in enumerate(ids)])
        if not keep.any():
            continue
        new = np.zeros(len(ids), np.int64)
        new[keep] = np.arange(nxt, nxt + keep.sum())
        nxt += int(keep.sum())
        local = new[inverse].reshape(raw.shape)
        view = output[tuple(slice(r[0], r[1]) for r in t['read'])]
        view[local > 0] = local[local > 0]
    return output


# ---------------------------------------------------------------- graph

def _components(nodes, edges, packed):
    uf = _UnionFind()
    for n in nodes:
        uf.find(int(n))
    for a, b in edges:
        uf.union(int(a), int(b))
    return _apply_union(packed.astype(np.int64), uf)


@register_policy('graph', 'native')
def _graph_native(bundle):
    return _components(bundle.meta['nodes'], bundle.meta['edges'], bundle.array('packed.npy'))


@register_policy('graph', 'disable_edges')
def _graph_disable(bundle):
    return _components(bundle.meta['nodes'], [], bundle.array('packed.npy'))


def _near_zero(a, threshold):
    """True where a zero pixel lies within Euclidean ``threshold`` (NumPy-only, exact)."""
    r = int(np.floor(threshold))
    zero = a == 0
    hit = zero.copy()
    for offset in np.ndindex(*([2 * r + 1] * a.ndim)):
        d = np.array(offset) - r
        if not d.any() or (d * d).sum() > threshold * threshold:
            continue
        src = tuple(slice(max(0, -k), n - max(0, k)) for k, n in zip(d, a.shape))
        dst = tuple(slice(max(0, k), n - max(0, -k)) for k, n in zip(d, a.shape))
        hit[dst] |= zero[src]
    return hit


def shrink_face(face, threshold=1.0):
    """Zero label boundaries within a face, then pixels within ``threshold`` of background.

    The gradient magnitude is computed per pixel for faces of any dimensionality, so 1-pixel
    faces of 2D images keep their interior (cf. MouseLand/cellpose#1364, #1480).
    """
    plane = np.asarray(face)
    squeezed = plane.reshape([n for n in plane.shape if n != 1] or [1])
    grads = np.gradient(squeezed.astype(float)) if min(squeezed.shape) > 1 else [np.zeros(squeezed.shape)]
    grads = grads if isinstance(grads, (list, tuple)) else [grads]
    magnitude = np.sqrt(sum(g * g for g in grads))
    shrunk = np.where(magnitude > 0, 0, squeezed)
    if (shrunk == 0).any():
        shrunk[_near_zero(shrunk, float(threshold))] = 0
    return shrunk.reshape(plane.shape)


def block_seams(bundle):
    """Per axis, the coordinates where one block's core ends and the next begins."""
    if bundle.meta['backend'] == 'graph':
        block = bundle.meta['parameters']['block_size']
        return [list(range(b, n, b)) for b, n in zip(block, bundle.meta['shape'])]
    tiles = _tiles(bundle)
    seams = [set() for _ in bundle.meta['shape']]
    for box in ownership_boxes(tiles):
        for ax, (lo, hi) in enumerate(box):
            if lo > 0:
                seams[ax].add(lo)
    return [sorted(s) for s in seams]


def face_edges(packed, seams, threshold=1.0):
    """Pairs of labels touching across each seam after shrinking every block face separately.

    As in the native driver, each face is one block's patch of the seam plane, so the shrink sees
    patch borders exactly as the per-block faces of the driver do.
    """
    edges = set()
    bounds = [[0, *cuts, n] for cuts, n in zip(seams, packed.shape)]
    for ax, cuts in enumerate(seams):
        others = [list(zip(b[:-1], b[1:])) if i != ax else [None] for i, b in enumerate(bounds)]
        for cut in cuts:
            for patch in np.ndindex(*[len(o) for o in others]):
                region = [slice(*others[i][k]) if i != ax else None for i, k in enumerate(patch)]
                left = tuple(slice(cut - 1, cut) if r is None else r for r in region)
                right = tuple(slice(cut, cut + 1) if r is None else r for r in region)
                a = shrink_face(packed[left], threshold).ravel()
                b = shrink_face(packed[right], threshold).ravel()
                both = (a > 0) & (b > 0)
                edges.update(zip(a[both].tolist(), b[both].tolist()))
    return sorted(edges)


@register_policy('graph', 'relink')
def _graph_relink(bundle, threshold=1.0):
    packed = bundle.array('packed.npy')
    edges = face_edges(packed, block_seams(bundle), float(threshold))
    return _components(bundle.meta['nodes'], edges, packed)


@register_policy('graph', 'relink_absorb')
def _graph_relink_absorb(bundle, threshold=1.0, min_size=15):
    """``relink``, then absorb seam fragments: every linked piece smaller than ``min_size`` voxels
    that touches a seam joins the piece across the seam it touches most (raw contact, no shrink), or
    is removed if it touches none. ``min_size`` defaults to Cellpose's own minimum object size."""
    require(float(min_size) >= 0, 'INVALID_PARAMETER', 'min_size must be non-negative')
    packed = bundle.array('packed.npy')
    seams = block_seams(bundle)
    linked = _components(bundle.meta['nodes'], face_edges(packed, seams, float(threshold)), packed)
    ids, sizes = np.unique(linked, return_counts=True)
    size = dict(zip(ids.tolist(), sizes.tolist()))
    contact = {}
    for ax, cuts in enumerate(seams):
        for cut in cuts:
            a = np.take(linked, cut - 1, axis=ax).ravel()
            b = np.take(linked, cut, axis=ax).ravel()
            both = (a > 0) & (b > 0) & (a != b)
            pairs, counts = np.unique(np.stack([a[both], b[both]]), axis=1, return_counts=True)
            for (x, y), c in zip(pairs.T.tolist(), counts.tolist()):
                contact.setdefault(x, {}).setdefault(y, 0)
                contact[x][y] += c
                contact.setdefault(y, {}).setdefault(x, 0)
                contact[y][x] += c
    from .compare import seam_mask
    on_seam = set(np.unique(linked[seam_mask(linked.shape, seams, 1)]).tolist()) - {0}
    uf, removed = _UnionFind(), set()
    for piece in sorted((i for i in on_seam if size[i] < float(min_size)), key=lambda i: size[i]):
        partners = {k: v for k, v in contact.get(piece, {}).items() if k not in removed}
        if partners:
            uf.union(piece, max(partners, key=lambda k: (partners[k], size[k])))
        else:
            removed.add(piece)
    out = _apply_union(np.where(np.isin(linked, list(removed)), 0, linked), uf) if (uf.parent or removed) else linked
    return out


def run_policy(bundle, policy='native', **params):
    """Apply a registered policy name or a callable ``fn(bundle, **params)``; canonical labels."""
    fn = policy if callable(policy) else get_policy(bundle.meta['backend'], policy)
    return canonical(fn(bundle, **params))
