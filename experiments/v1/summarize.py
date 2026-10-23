"""Aggregate per-image v1 results into summary.json and SUMMARY.md (no new computation)."""
import json, statistics
from collections import Counter, defaultdict
from common import RUNS, write


def mean_sd(values):
    values = list(values)
    if not values:
        return None
    return {'mean': statistics.fmean(values), 'sd': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values), 'max': max(values), 'n': len(values)}


def exact(v):
    return v['fp'] == 0 and v['fn'] == 0 and v['split'] == 0 and v['merge'] == 0


def policy_block(entries):
    out = {}
    for name in sorted({p for e in entries for p in e['policies']}):
        rows = [e['policies'][name] for e in entries]
        att = Counter()
        for r in rows:
            att.update(r['attribution_vs_untiled'])
        out[name] = {
            'f1_vs_untiled': mean_sd(r['vs_untiled']['f1'] for r in rows),
            'exact_vs_untiled': sum(exact(r['vs_untiled']) for r in rows),
            'split_vs_untiled_total': sum(r['vs_untiled']['split'] for r in rows),
            'merge_vs_untiled_total': sum(r['vs_untiled']['merge'] for r in rows),
            'fp_vs_untiled_total': sum(r['vs_untiled']['fp'] for r in rows),
            'fn_vs_untiled_total': sum(r['vs_untiled']['fn'] for r in rows),
            'attribution_totals': dict(att),
            'f1_vs_truth': mean_sd(r['vs_truth']['f1'] for r in rows),
            'split_vs_truth_total': sum(r['vs_truth']['split'] for r in rows),
            'replay_seconds_median': statistics.median(r['replay_seconds_median'] for r in rows),
        }
    return out


def stardist():
    records = [json.loads(p.read_text()) for p in sorted((RUNS / 'stardist' / 'results').glob('*.json'))]
    configs = defaultdict(list)
    for r in records:
        for c in r['configs']:
            configs[(c['block_size'], c['min_overlap'])].append(c)
    summary = {'images': len(records), 'untiled_seconds': mean_sd(r['untiled_seconds'] for r in records),
               'untiled_f1_vs_truth': mean_sd(r['untiled_vs_truth']['f1'] for r in records), 'configs': {}}
    for (b, o), entries in sorted(configs.items()):
        ok = [e for e in entries if e['status'] == 'ok']
        summary['configs'][f'{b}/{o}'] = {
            'ok': len(ok), 'refused': len(entries) - len(ok),
            'refusal_causes': dict(Counter(e['cause'].split(':')[0] for e in entries if e['status'] != 'ok')),
            'instrumentation_pixel_equal': sum(e['instrumentation_pixel_equal'] for e in ok),
            'tiles': mean_sd(e['tiles'] for e in ok), 'capture_seconds': mean_sd(e['capture_seconds'] for e in ok),
            'plain_seconds': mean_sd(e['plain_seconds'] for e in ok), 'bundle_mib': mean_sd(e['bundle_bytes'] / 2**20 for e in ok),
            'policies': policy_block(ok) if ok else {}}
    return summary


def cellpose():
    records = [json.loads(p.read_text()) for p in sorted((RUNS / 'cellpose' / 'results').glob('*.json'))]
    blocks = defaultdict(list)
    for r in records:
        for b in r['blocks']:
            blocks[b['block_size']].append(b)
    repeats = [r['untiled_repeat'] for r in records if 'untiled_repeat' in r]
    summary = {'images': len(records), 'untiled_seconds': mean_sd(r['untiled_seconds'] for r in records),
               'untiled_f1_vs_truth': mean_sd(r['untiled_vs_truth']['f1'] for r in records),
               'untiled_repeat_pixel_equal': f"{sum(x['pixel_equal'] for x in repeats)}/{len(repeats)}", 'blocks': {}}
    for b, entries in sorted(blocks.items()):
        fidelity = [e['instrumentation_pixel_equal'] for e in entries if 'instrumentation_pixel_equal' in e]
        summary['blocks'][str(b)] = {
            'images': len(entries), 'blocks_per_image': mean_sd(e['blocks'] for e in entries),
            'recorded_edges_total': sum(e['recorded_edges'] for e in entries),
            'instrumentation_pixel_equal': f'{sum(fidelity)}/{len(fidelity)}',
            'capture_seconds': mean_sd(e['capture_seconds'] for e in entries),
            'bundle_mib': mean_sd(e['bundle_bytes'] / 2**20 for e in entries),
            'policies': policy_block(entries)}
    return summary


def cellpose3d():
    records = [json.loads(p.read_text()) for p in sorted((RUNS / 'cellpose3d' / 'results').glob('*.json'))]
    summary = {'volumes': len(records), 'untiled_seconds': mean_sd(r['untiled_seconds'] for r in records),
               'untiled_f1_vs_truth': mean_sd(r['untiled_vs_truth']['f1'] for r in records), 'blocks': {}}
    for kind in ('chain', 'grid'):
        entries = [r['blocks'][kind] for r in records if kind in r['blocks']]
        summary['blocks'][kind] = {
            'volumes': len(entries), 'blocks_per_volume': mean_sd(e['blocks'] for e in entries),
            'linked_groups_equal': f"{sum(e['linked_groups_equal'] for e in entries)}/{len(entries)}",
            'native_relink_outputs_equal': f"{sum(e['native_relink_outputs_equal'] for e in entries)}/{len(entries)}",
            'recorded_edges_total': sum(e['recorded_edges'] for e in entries),
            'capture_seconds': mean_sd(e['capture_seconds'] for e in entries),
            'bundle_mib': mean_sd(e['bundle_bytes'] / 2**20 for e in entries),
            'policies': policy_block(entries)}
    return summary


def fmt(ms, digits=3):
    return '–' if ms is None else f"{ms['mean']:.{digits}f} ± {ms['sd']:.{digits}f}"


def markdown(s):
    lines = ['# v1 formal results (generated by summarize.py)', '']
    c = s['cellpose']
    lines += [f"## Cellpose-SAM, distributed driver ({c['images']} images)", '',
              f"Untiled inference {fmt(c['untiled_seconds'], 2)} s; untiled F1 vs annotation {fmt(c['untiled_f1_vs_truth'])}; "
              f"untiled repeat pixel-identical {c['untiled_repeat_pixel_equal']}.", '',
              '| block | blocks/img | recorded edges | capture s | policy | F1 vs untiled | exact imgs | splits | seam_split | seam_other | interior | seam_extra | interior_extra | F1 vs annotation | replay s |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for b, e in c['blocks'].items():
        for name, p in e['policies'].items():
            a = p['attribution_totals']
            lines.append(f"| {b} | {e['blocks_per_image']['mean']:.1f} | {e['recorded_edges_total']} | {fmt(e['capture_seconds'], 1)} | {name} | "
                         f"{fmt(p['f1_vs_untiled'])} | {p['exact_vs_untiled']}/{e['images']} | {p['split_vs_untiled_total']} | "
                         f"{a.get('seam_split', 0)} | {a.get('seam_other', 0)} | {a.get('interior', 0)} | {a.get('seam_extra', 0)} | {a.get('interior_extra', 0)} | {fmt(p['f1_vs_truth'])} | {p['replay_seconds_median']:.3f} |")
        lines.append(f"| {b} | | | | instrumentation pixel-equal | {e['instrumentation_pixel_equal']} | | | | | | | | | |")
    d = s['stardist']
    lines += ['', f"## StarDist ({d['images']} images)", '',
              f"Untiled inference {fmt(d['untiled_seconds'], 2)} s; untiled F1 vs annotation {fmt(d['untiled_f1_vs_truth'])}.", '',
              '| block/overlap | ok | refused (cause) | instr. equal | tiles | capture s | policy | F1 vs untiled | exact imgs | splits | merges | F1 vs annotation | replay s |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k, e in d['configs'].items():
        causes = ', '.join(f'{n} {c}' for c, n in e['refusal_causes'].items()) or '–'
        if not e['policies']:
            lines.append(f"| {k} | 0 | {e['refused']} ({causes}) | | | | | | | | | | |")
        for name, p in e['policies'].items():
            lines.append(f"| {k} | {e['ok']} | {e['refused']} ({causes}) | {e['instrumentation_pixel_equal']}/{e['ok']} | {fmt(e['tiles'], 1)} | "
                         f"{fmt(e['capture_seconds'], 2)} | {name} | {fmt(p['f1_vs_untiled'])} | {p['exact_vs_untiled']}/{e['ok']} | "
                         f"{p['split_vs_untiled_total']} | {p['merge_vs_untiled_total']} | {fmt(p['f1_vs_truth'])} | {p['replay_seconds_median']:.3f} |")
    t = s['cellpose3d']
    lines += ['', f"## Cellpose-SAM 3D, C. elegans test volumes ({t['volumes']} volumes)", '',
              f"Untiled inference {fmt(t['untiled_seconds'], 0)} s (shared load); untiled F1 vs annotation {fmt(t['untiled_f1_vs_truth'])}.", '',
              '| blocking | volumes | blocks/vol | groups equal | outputs equal | capture s | MiB | policy | F1 vs untiled | splits | merges | seam_split | seam_other | interior | seam_extra | interior_extra | F1 vs annotation | replay s |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k, e in t['blocks'].items():
        for name, p in e['policies'].items():
            a = p['attribution_totals']
            lines.append(f"| {k} | {e['volumes']} | {e['blocks_per_volume']['mean']:.1f} | {e['linked_groups_equal']} | {e['native_relink_outputs_equal']} | "
                         f"{fmt(e['capture_seconds'], 0)} | {e['bundle_mib']['mean']:.0f} | {name} | {fmt(p['f1_vs_untiled'])} | {p['split_vs_untiled_total']} | "
                         f"{p['merge_vs_untiled_total']} | {a.get('seam_split', 0)} | {a.get('seam_other', 0)} | {a.get('interior', 0)} | "
                         f"{a.get('seam_extra', 0)} | {a.get('interior_extra', 0)} | {fmt(p['f1_vs_truth'])} | {p['replay_seconds_median']:.2f} |")
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    s = {'cellpose': cellpose(), 'stardist': stardist(), 'cellpose3d': cellpose3d()}
    write(RUNS / 'summary.json', s)
    (RUNS / 'SUMMARY.md').write_text(markdown(s))
    print((RUNS / 'SUMMARY.md').read_text())
