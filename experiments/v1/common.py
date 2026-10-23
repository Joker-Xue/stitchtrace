"""Shared data access and policy evaluation for the v1 formal experiments (see PROTOCOL.md)."""
import json, statistics, time
from pathlib import Path
import numpy as np, tifffile, imageio.v3 as iio
from skimage.measure import label
from stitchtrace import replay, compare, attribute, available_policies
from stitchtrace.policies import block_seams

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'bbbc039'
RUNS = ROOT / 'runs' / 'v1'
POLICY_PARAMS = {'iou_merge': {'threshold': 0.5}, 'relink': {'threshold': 1.0}}


def test_images():
    return [n.replace('.png', '') for n in sorted((DATA / 'metadata' / 'test.txt').read_text().split())]


def load(stem):
    image = tifffile.imread(DATA / 'images' / f'{stem}.tif')
    reference = label(iio.imread(DATA / 'masks' / f'{stem}.png')[..., 0])
    return image, reference


def timed(fn, repeats=1):
    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter(); out = fn(); times.append(time.perf_counter() - t0)
    return out, statistics.median(times), times


def bundle_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def slim(c):
    return {k: c[k] for k in ('reference_objects', 'candidate_objects', 'tp', 'fp', 'fn', 'f1',
                              'mean_matched_iou', 'split', 'merge')}


def evaluate(bundle, untiled, truth, repeats=5):
    seams = block_seams(bundle)
    out = {}
    for name in available_policies(bundle.meta['backend']):
        if name.startswith('test_'):
            continue
        params = POLICY_PARAMS.get(name, {})
        (labels, _), median, times = timed(lambda: replay(bundle, name, **params), repeats)
        att = attribute(untiled, labels, seams)
        out[name] = {'params': params, 'replay_seconds_median': median, 'replay_seconds': times,
                     'vs_untiled': slim(att['comparison']), 'attribution_vs_untiled': att['summary'],
                     'vs_truth': slim(compare(truth, labels))}
    return out


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)
