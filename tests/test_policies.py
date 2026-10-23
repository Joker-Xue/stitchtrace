import io
import json
import numpy as np
import pytest
from stitchtrace import (Bundle, replay, register_policy, available_policies, run_policy, compare, attribute,
                         canonical, StitchTraceError)
from stitchtrace.cli import main
from stitchtrace.policies import shrink_face, face_edges, block_seams, ownership_boxes, _tiles
from stitchtrace.storage import transaction, atomic_array, atomic_bytes, commit


def ownership(path):
    """Two tiles along x; object 1 spans the write overlap [8, 12), object 2 lies in tile 1 only."""
    shape = (8, 20)
    raw0 = np.zeros((8, 14), np.int32); raw0[2:6, 6:12] = 1
    raw1 = np.zeros((8, 14), np.int32); raw1[2:6, 0:6] = 1; raw1[2:6, 9:12] = 2
    expected = np.zeros(shape, np.int32); expected[2:6, 6:12] = 1; expected[2:6, 15:18] = 2
    tiles = []
    with transaction(path, 'own') as root:
        for tid, raw, read, write, kept in ((0, raw0, [[0, 8], [0, 14]], [[0, 8], [0, 12]], [1]),
                                            (1, raw1, [[0, 8], [6, 20]], [[0, 8], [8, 20]], [2])):
            stream = io.BytesIO(); np.savez(stream, labels=raw); atomic_bytes(root / f'cache/{tid}.npz', stream.getvalue())
            crop = tuple(slice(w[0] - r[0], w[1] - r[0]) for r, w in zip(read, write))
            local = raw[crop]
            selected = np.where(np.isin(local, kept), local, 0)
            written = np.where(selected > 0, tid + 1, 0).astype(np.int32)
            for k, a in (('local', local), ('selected', selected), ('written', written)):
                atomic_array(root / f'tiles/{tid}-{k}.npy', a)
            rejected = sorted(set(np.unique(local).tolist()) - {0} - set(kept))
            tiles.append({'id': tid, 'read': read, 'write': write, 'cache': f'cache/{tid}.npz', 'local': f'tiles/{tid}-local.npy',
                          'selected': f'tiles/{tid}-selected.npy', 'written': f'tiles/{tid}-written.npy',
                          'kept': kept, 'rejected': rejected})
        atomic_array(root / 'expected.npy', expected)
        commit(root, {'schema': 1, 'backend': 'ownership', 'contract': 'own', 'shape': list(shape), 'axes': 'YX',
                      'spacing': [1, 1], 'tiles': tiles, 'order': [0, 1]})
    return Bundle.open(path)


def graph2d(path, edges=()):
    """2D distributed record whose only object is cut at the x=4 block seam; edges as recorded."""
    packed = np.zeros((8, 8), np.uint32)
    packed[1:7, 2:4] = 1
    packed[1:7, 4:6] = 100001
    expected = (packed > 0).astype(np.uint32) if edges else np.where(packed == 1, 1, np.where(packed > 0, 2, 0)).astype(np.uint32)
    with transaction(path, 'g') as root:
        atomic_array(root / 'packed.npy', packed)
        atomic_array(root / 'expected.npy', expected)
        commit(root, {'schema': 1, 'backend': 'graph', 'contract': 'g', 'shape': [8, 8], 'axes': 'YX', 'spacing': [1, 1],
                      'parameters': {'block_size': [4, 4]}, 'nodes': [1, 100001], 'edges': [list(e) for e in edges]})
    return Bundle.open(path)


def count(a):
    return len(np.unique(a)) - 1


def test_ownership_policies(tmp_path):
    b = ownership(tmp_path / 'b')
    native = replay(b)[0]
    assert count(native) == 2
    assert [tuple(map(tuple, box)) for box in ownership_boxes(_tiles(b))] == [((0, 8), (0, 10)), ((0, 8), (10, 20))]
    assert count(replay(b, 'crop')[0]) == 3                      # object 1 cut at the midline
    assert np.array_equal(replay(b, 'iou_merge')[0], native)     # pieces relinked through the overlap
    assert np.array_equal(replay(b, 'center')[0], native)        # centroid x=8.5 owned by tile 0
    assert count(replay(b, 'retain_all')[0]) == 3                # tile 1 overwrites part of object 1
    assert compare(native, replay(b, 'crop')[0])['split'] == 1
    with pytest.raises(StitchTraceError, match='INVALID_PARAMETER'):
        replay(b, 'iou_merge', threshold=0)


def test_custom_policy_and_registry(tmp_path):
    b = ownership(tmp_path / 'b')
    def everything_one(bundle):
        return (bundle.array('expected.npy') > 0).astype(np.int64)
    assert count(run_policy(b, everything_one)) == 1
    register_policy('ownership', 'test_union', everything_one)
    assert 'test_union' in available_policies('ownership')
    assert count(replay(b, 'test_union')[0]) == 1
    with pytest.raises(StitchTraceError, match='POLICY_UNSUPPORTED'):
        replay(b, 'disable_edges')


def test_shrink_face_keeps_interior_of_1d_faces():
    line = np.array([[0, 7, 7, 7, 7, 7, 7, 0]])
    assert shrink_face(line).tolist() == [[0, 0, 0, 7, 7, 0, 0, 0]]
    assert shrink_face(np.full((1, 5), 3)).tolist() == [[3, 3, 3, 3, 3]]
    assert shrink_face(np.zeros((1, 4), int)).tolist() == [[0, 0, 0, 0]]


def test_shrink_face_matches_upstream_on_planes():
    cp = pytest.importorskip('cellpose.contrib.distributed_segmentation')
    rng = np.random.default_rng(0)
    for _ in range(100):
        f = np.zeros((1, 16, 16), int)
        for k in range(3):
            y, x = rng.integers(0, 12, 2)
            f[0, y:y + rng.integers(2, 7), x:x + rng.integers(2, 7)] = k + 1
        for t in (1.0, 2.0):
            assert np.array_equal(shrink_face(f, t), cp.shrink_labels(f, t))


def test_relink_repairs_unlinked_2d_seam(tmp_path):
    b = graph2d(tmp_path / 'b')
    assert block_seams(b) == [[4], [4]]
    assert face_edges(b.array('packed.npy'), block_seams(b)) == [(1, 100001)]
    assert count(replay(b)[0]) == 2
    assert count(replay(b, 'disable_edges')[0]) == 2
    assert count(replay(b, 'relink')[0]) == 1
    linked = graph2d(tmp_path / 'c', edges=[(1, 100001)])
    assert np.array_equal(replay(linked)[0], replay(b, 'relink')[0])


def test_compare_and_attribution():
    ref = np.zeros((10, 10), int); ref[2:8, 2:8] = 1; ref[0:2, 0:2] = 2
    split = ref.copy(); split[2:8, 5:8] = 3; split[0:2, 0:2] = 0
    r = compare(ref, ref)
    assert (r['tp'], r['fp'], r['fn'], r['f1'], r['split'], r['merge']) == (2, 0, 0, 1.0, 0, 0)
    r = compare(ref, split)
    # the larger piece still matches object 1 at IoU 0.5, so only object 2 is unmatched
    assert r['split'] == 1 and r['fn'] == 1 and r['split_ids'] == [1] and r['unmatched_reference_ids'] == [2]
    merged = (ref > 0).astype(int); merged[0:3, 0:3] = 1; merged[2:8, 2:8] = 1
    assert compare(ref, merged)['merge'] == 1
    att = attribute(ref, split, [[], [5]])
    assert att['summary']['seam_split'] == 1 and att['summary']['interior'] == 1
    assert {o['reference_id']: o['category'] for o in att['objects']} == {1: 'seam_split', 2: 'interior'}
    with pytest.raises(StitchTraceError):
        compare(ref, split[:5])


def test_measurements_vectorized_match_direct():
    from stitchtrace import measurements
    rng = np.random.default_rng(3)
    a = rng.integers(0, 6, (5, 7, 9))
    m = measurements(a, [2, 1, .5])
    for k, v in m.items():
        pts = np.argwhere(a == int(k))
        assert v['pixels'] == len(pts) and v['coordinate_sum'] == pts.sum(0).tolist()
        np.testing.assert_allclose(v['centroid'], pts.mean(0) * [2, 1, .5])
    c = canonical(a)
    assert c.ravel()[np.flatnonzero(a.ravel())[0]] == 1 and len(np.unique(c)) == len(np.unique(a))


def test_cli_policies_compare_and_params(tmp_path, capsys):
    b = ownership(tmp_path / 'b')
    assert main(['policies']) == 0
    assert 'iou_merge' in json.loads(capsys.readouterr().out)['ownership']
    assert main(['replay', str(tmp_path / 'b'), '--policy', 'iou_merge', '--param', 'threshold=0.3',
                 '--output', str(tmp_path / 'm.npy')]) == 0
    capsys.readouterr()
    assert main(['replay', str(tmp_path / 'b'), '--output', str(tmp_path / 'n.npy')]) == 0
    assert main(['replay', str(tmp_path / 'b'), '--policy', 'crop', '--output', str(tmp_path / 'c.npy')]) == 0
    capsys.readouterr()
    assert main(['compare', str(tmp_path / 'n.npy'), str(tmp_path / 'c.npy'), '--bundle', str(tmp_path / 'b')]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out['summary']['seam_split'] == 1
    assert main(['replay', str(tmp_path / 'b'), '--param', 'bad', '--output', str(tmp_path / 'x.npy')]) == 2


def test_policies_with_an_empty_tile(tmp_path):
    """A block without any object must not break relabeling (seen on real BBBC039 images)."""
    from stitchtrace.policies import _relabel
    empty = np.zeros((3, 4), np.int64)
    assert _relabel(empty, np.array([], np.int64), np.array([], np.int64)).tolist() == empty.tolist()
    b = ownership(tmp_path / 'b')
    blank = b.array(b.meta['tiles'][0]['local']) * 0
    from stitchtrace.policies import _ownership_retain_all
    class Blank:
        meta = b.meta
        def array(self, name):
            return blank if name == b.meta['tiles'][0]['local'] else b.array(name)
    assert count(_ownership_retain_all(Blank())) == 2


def test_attribution_covers_extra_candidate_objects():
    ref = np.zeros((10, 10), int); ref[2:8, 2:8] = 1
    cand = ref.copy(); cand[4:6, 4:6] = 0; cand[0:1, 4:6] = 7; cand[9, 0] = 8   # seam fragment + interior extra
    att = attribute(ref, cand, [[1], []])
    assert att['comparison']['unmatched_candidate_ids'] == [7, 8]
    assert att['summary']['seam_extra'] == 1 and att['summary']['interior_extra'] == 1


def test_relink_absorb_joins_or_drops_small_seam_fragments(tmp_path):
    b = graph2d(tmp_path / 'b')
    assert count(replay(b, 'relink_absorb')[0]) == 1           # normal pieces still linked
    packed = np.zeros((8, 8), np.uint32)
    packed[1:7, 0:4] = 1                                        # large piece left of the x=4 seam
    packed[3, 4] = 100001                                       # 1-pixel fragment, contact too thin for the shrink
    packed[6, 6] = 100002                                       # isolated speck away from any seam
    with transaction(tmp_path / 'f', 'f') as root:
        atomic_array(root / 'packed.npy', packed)
        atomic_array(root / 'expected.npy', packed)
        commit(root, {'schema': 1, 'backend': 'graph', 'contract': 'f', 'shape': [8, 8], 'axes': 'YX', 'spacing': [1, 1],
                      'parameters': {'block_size': [8, 4]}, 'nodes': [1, 100001, 100002], 'edges': []})
    f = Bundle.open(tmp_path / 'f')
    assert count(replay(f, 'relink')[0]) == 3
    out = replay(f, 'relink_absorb')[0]
    assert count(out) == 2 and out[3, 4] == out[3, 3]           # fragment joined; interior speck untouched
    assert count(replay(f, 'relink_absorb', min_size=0)[0]) == 3
