"""Write a small synthetic graph bundle that reproduces an unlinked 2D block seam (NumPy only).

Twelve discs on a 192 x 192 field are cut by a 64-pixel block grid exactly as a distributed driver
packs per-block labels; the recorded edge list is empty, as produced by Cellpose 4.2.1.1 for 2D
images (MouseLand/cellpose#1364). Usage: python make_demo_bundle.py OUTPUT_DIR
"""
import sys
from pathlib import Path
import numpy as np
from stitchtrace import Bundle
from stitchtrace.storage import transaction, atomic_array, commit

SHAPE, BLOCK = (192, 192), 64


def discs(seed=7, n=12, radius=(9, 15)):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(SHAPE)
    truth = np.zeros(SHAPE, np.uint32)
    k = 0
    while k < n:
        r = rng.integers(*radius)
        cy, cx = rng.integers(r, SHAPE[0] - r, 2)
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        if truth[disc].any():
            continue
        k += 1
        truth[disc] = k
    return truth


def packed_blocks(truth):
    """Give every object piece inside a block a block-unique ID, as distributed drivers do."""
    packed = np.zeros(SHAPE, np.uint32)
    nblocks = [n // BLOCK for n in SHAPE]
    for index in np.ndindex(*nblocks):
        region = tuple(slice(i * BLOCK, (i + 1) * BLOCK) for i in index)
        local = truth[region]
        prefix = int(np.ravel_multi_index(index, nblocks))
        packed[region] = np.where(local > 0, prefix * 100000 + local, 0)
    return packed


def main(destination):
    truth = discs()
    packed = packed_blocks(truth)
    nodes = sorted(int(i) for i in np.unique(packed) if i)
    with transaction(destination, 'demo-unlinked-2d-seam') as root:
        if root is not None:
            atomic_array(root / 'packed.npy', packed)
            atomic_array(root / 'expected.npy', packed)          # no edges: every piece is its own object
            atomic_array(root / 'reference.npy', truth)          # the uncut objects, for comparison
            commit(root, {'schema': 1, 'backend': 'graph', 'contract': 'demo-unlinked-2d-seam',
                          'shape': list(SHAPE), 'axes': 'YX', 'spacing': [1.0, 1.0],
                          'parameters': {'block_size': [BLOCK, BLOCK]}, 'nodes': nodes, 'edges': []})
    b = Bundle.open(destination)
    print(f'{destination}: {int(truth.max())} objects cut into {len(nodes)} recorded pieces')
    return b


if __name__ == '__main__':
    main(Path(sys.argv[1] if len(sys.argv) > 1 else 'demo-bundle'))
