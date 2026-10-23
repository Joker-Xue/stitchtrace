import numpy as np
import pytest

pytest.importorskip('cellpose.contrib.distributed_segmentation')
pytest.importorskip('distributed')
from stitchtrace import replay
from stitchtrace.backends.cellpose import capture_cellpose


def test_in_place_predictor_cannot_rewrite_image_or_neighbours(tmp_path):
    """A predictor that normalises its input in place (as Cellpose does for float32) must not
    change the caller's image, the overlapping crops of other blocks, or the run contract."""
    image = np.zeros((48, 48), np.float32)
    image[10:30, 14:34] = 100.
    original = image.copy()
    seen = []
    def predict(crop, slices):
        seen.append(float(crop.max()))
        crop /= max(float(crop.max()), 1.)          # in-place normalisation
        return (crop > 0.5).astype(np.uint32)
    b = capture_cellpose(image, predict, tmp_path / 'b', predictor_id='inplace', block_size=[24, 24], diameter=4)
    assert np.array_equal(image, original)
    assert all(v in (0., 100.) for v in seen)
    assert len(np.unique(replay(b, 'relink')[0])) - 1 == 1
    again = capture_cellpose(image, predict, tmp_path / 'b', predictor_id='inplace', block_size=[24, 24], diameter=4)
    assert again.meta['contract'] == b.meta['contract']


def test_relink_matches_upstream_linking_on_3d_grids():
    """On 3D block grids relink must merge exactly the pieces the native face graph merges."""
    import scipy.sparse
    from cellpose.contrib import distributed_segmentation as cp
    from stitchtrace.policies import face_edges, _components
    from stitchtrace import canonical
    rng = np.random.default_rng(5)
    for trial in range(12):
        block = (8, 10, 12)
        shape = tuple(b * n for b, n in zip(block, (2, 2, 3)))
        truth = np.zeros(shape, np.uint32)
        for k in range(1, 40):
            c = [rng.integers(0, n) for n in shape]
            r = rng.integers(2, 5)
            sl = tuple(slice(max(0, x - r), x + r) for x in c)
            truth[sl] = np.where(truth[sl] == 0, k, truth[sl])
        nblocks = tuple(n // b for n, b in zip(shape, block))
        packed = np.zeros(shape, np.uint32)
        indices, faces = [], []
        for index in np.ndindex(*nblocks):
            region = tuple(slice(i * b, (i + 1) * b) for i, b in zip(index, block))
            seg, _ = cp.global_segment_ids(truth[region], index, nblocks)
            packed[region] = seg
            indices.append(index)
            faces.append(cp.block_faces(seg))
        pairs = cp.adjacent_faces(indices, faces)
        graph = cp.block_face_adjacency_graph(pairs, int(packed.max()))
        _, comp = scipy.sparse.csgraph.connected_components(graph, directed=False)
        upstream = comp[packed.astype(np.int64)] * (packed > 0)
        seams = [list(range(b, n, b)) for b, n in zip(block, shape)]
        ours = _components(np.unique(packed[packed > 0]).tolist(), face_edges(packed, seams), packed)
        assert np.array_equal(canonical(upstream), canonical(ours)), trial
