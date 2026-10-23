"""Recompute every policy evaluation of saved bundles with the current code (no inference)."""
import json, sys
import numpy as np, tifffile
from stitchtrace import Bundle
from common import RUNS, load, evaluate, write


def cellpose_2d(arm):
    for path in sorted((RUNS / arm / 'results').glob('*.json')):
        record = json.loads(path.read_text())
        stem = record['image']
        untiled = np.load(RUNS / arm / 'untiled' / f'{stem}.npy')
        _, truth = load(stem)
        for entry in record['blocks']:
            bundle = Bundle.open(RUNS / arm / 'bundles' / f"{stem}-b{entry['block_size']}")
            entry['policies'] = evaluate(bundle, untiled, truth)
        yield path, record


def stardist(arm='stardist'):
    for path in sorted((RUNS / arm / 'results').glob('*.json')):
        record = json.loads(path.read_text())
        stem = record['image']
        untiled = np.load(RUNS / arm / 'untiled' / f'{stem}.npy')
        _, truth = load(stem)
        for entry in record['configs']:
            if entry['status'] == 'ok':
                bundle = Bundle.open(RUNS / arm / 'bundles' / f"{stem}-b{entry['block_size']}-o{entry['min_overlap']}")
                entry['policies'] = evaluate(bundle, untiled, truth)
        yield path, record


def cellpose_3d(arm='cellpose3d'):
    from run_cellpose_3d import DATA
    for path in sorted((RUNS / arm / 'results').glob('*.json')):
        record = json.loads(path.read_text())
        stem = record['volume']
        untiled = np.load(RUNS / arm / 'untiled' / f'{stem}.npy')
        truth = tifffile.imread(DATA / 'test' / 'masks' / f'{stem}.tif')
        for name, entry in record['blocks'].items():
            entry['policies'] = evaluate(Bundle.open(RUNS / arm / 'bundles' / f'{stem}-{name}'), untiled, truth)
        yield path, record


ARMS = {'cellpose': lambda: cellpose_2d('cellpose'), 'cellpose_globalnorm': lambda: cellpose_2d('cellpose_globalnorm'),
        'stardist': stardist, 'cellpose3d': cellpose_3d}


if __name__ == '__main__':
    tag = sys.argv[1]
    for arm in sys.argv[2:] or list(ARMS):
        for path, record in ARMS[arm]():
            record.setdefault('reevaluated', [])
            record['reevaluated'] = (record['reevaluated'] if isinstance(record['reevaluated'], list) else [record['reevaluated']]) + [tag]
            write(path, record)
        print(arm, 'done', flush=True)
