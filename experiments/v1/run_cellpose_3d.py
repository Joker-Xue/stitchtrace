"""3D scale arm (PROTOCOL-3D.md): C. elegans test volumes, Cellpose-SAM 3D, chain and grid blocks."""
import sys, time
from pathlib import Path
import numpy as np, tifffile
from cellpose import models
from stitchtrace.backends.cellpose import capture_cellpose
from stitchtrace import replay
from stitchtrace.compare import compare
from stitchtrace.policies import face_edges, block_seams
from common import ROOT, RUNS, evaluate, bundle_bytes, slim, write

DATA = ROOT / 'data' / 'celegans' / 'c_elegans_nuclei'
BLOCKS = {'chain': [140, 140, 128], 'grid': [70, 70, 128]}
DRIVER_DIAMETER = 20                       # distributed overlap 2 x 20 px
MODEL_DIAMETER = None                      # Cellpose-SAM default; not tuned (DEVIATIONS S1)
GRID_VOLUMES = 2                           # grid blocking on the first volumes only (DEVIATIONS S1)


def volumes(split):
    return sorted((DATA / split / 'images').glob('*.tif'))


def undirected(edges):
    return {tuple(sorted((int(a), int(b)))) for a, b in edges if a and b and a != b}


def linked_groups(edges):
    """Connected components induced by an edge list, as a set of frozensets (groups of size >= 2).

    The native driver records each face component as a chain of labels, relink as pairs across the
    seam; equal groups mean the two linkings merge exactly the same pieces."""
    parent = {}
    def find(i):
        parent.setdefault(i, i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in undirected(edges):
        parent[find(a)] = find(b)
    groups = {}
    for node in list(parent):
        groups.setdefault(find(node), set()).add(node)
    return {frozenset(g) for g in groups.values() if len(g) > 1}


def main(split='test', limit=None, diameter=MODEL_DIAMETER, out_name='cellpose3d'):
    model = models.CellposeModel(gpu=True)

    def predict(vol):
        return model.eval(np.array(vol), do_3D=True, z_axis=0, diameter=diameter,
                          normalize={'percentile': [1, 99.8]})[0].astype(np.uint32)

    out_dir = RUNS / out_name
    (out_dir / 'untiled').mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(volumes(split)[:limit]):
        stem = path.stem
        result_path = out_dir / 'results' / f'{stem}.json'
        if result_path.exists():
            continue
        image = tifffile.imread(path).astype(np.float32)
        truth = tifffile.imread(DATA / split / 'masks' / path.name)
        t0 = time.perf_counter(); untiled = predict(image); t_untiled = time.perf_counter() - t0
        np.save(out_dir / 'untiled' / f'{stem}.npy', untiled)
        record = {'volume': stem, 'shape': list(image.shape), 'model_diameter': diameter,
                  'untiled_seconds': t_untiled, 'untiled_vs_truth': slim(compare(truth, untiled)), 'blocks': {}}
        for name, block in BLOCKS.items():
            if name == 'grid' and index >= GRID_VOLUMES:
                continue
            dest = out_dir / 'bundles' / f'{stem}-{name}'
            t0 = time.perf_counter()
            bundle = capture_cellpose(image, lambda crop, s: predict(crop), dest,
                                      predictor_id=f'cpsam_v2;3D;p1-99.8;diameter-{diameter}',
                                      block_size=block, diameter=DRIVER_DIAMETER)
            capture = time.perf_counter() - t0
            native_edges = undirected(bundle.meta['edges'])
            relink_edges = undirected(face_edges(bundle.array('packed.npy'), block_seams(bundle)))
            native_out, relink_out = replay(bundle)[0], replay(bundle, 'relink')[0]
            entry = {'block_size': block, 'capture_seconds': capture,
                     'blocks': int(np.prod([-(-n // b) for n, b in zip(image.shape, block)])),
                     'bundle_bytes': bundle_bytes(dest), 'recorded_edges': len(native_edges),
                     'relink_edges': len(relink_edges), 'edge_sets_equal': native_edges == relink_edges,
                     'linked_groups_equal': linked_groups(native_edges) == linked_groups(relink_edges),
                     'native_relink_outputs_equal': bool(np.array_equal(native_out, relink_out)),
                     'policies': evaluate(bundle, untiled, truth)}
            record['blocks'][name] = entry
            print(stem, name, entry['blocks'], 'edges', entry['recorded_edges'], entry['relink_edges'], entry['linked_groups_equal'], entry['native_relink_outputs_equal'],
                  {p: round(v['vs_untiled']['f1'], 4) for p, v in entry['policies'].items()}, flush=True)
        write(result_path, record)


if __name__ == '__main__':
    main(*(sys.argv[1:2] or ['test']), int(sys.argv[2]) if len(sys.argv) > 2 else None,
         None if len(sys.argv) <= 3 or sys.argv[3] == 'None' else float(sys.argv[3]),
         sys.argv[4] if len(sys.argv) > 4 else 'cellpose3d')
