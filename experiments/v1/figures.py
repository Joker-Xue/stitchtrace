"""Paper figures from the saved v1 results (no inference). Output: runs/v1/figures/*.png"""
import json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from stitchtrace import Bundle, replay, compare
from stitchtrace.policies import block_seams
from common import RUNS, load

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d3'
OUT = RUNS / 'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 8.5, 'axes.titleweight': 'bold', 'savefig.dpi': 300})


def results(arm):
    return [json.loads(open(f).read()) for f in sorted(glob.glob(str(RUNS / arm / 'results' / '*.json')))]


def outline(ax, labels, color, width=0.8, only=None):
    mask = labels if only is None else np.where(np.isin(labels, only), labels, 0)
    ax.contour(find_boundaries(mask, mode='inner'), [0.5], colors=color, linewidths=width)


def fig_case(stem='IXMtest_A09_s1_w1CE70AD49-290D-4312-82E6-CDC717F32637', block=128, window=(slice(100, 380), slice(200, 520))):
    raw, _ = load(stem)
    untiled = np.load(RUNS / 'cellpose' / 'untiled' / f'{stem}.npy')
    b = Bundle.open(RUNS / 'cellpose' / 'bundles' / f'{stem}-b{block}')
    native, relinked = replay(b)[0], replay(b, 'relink')[0]
    split_native = compare(untiled, native)['split_ids']
    seams = block_seams(b)
    v = np.clip(raw / np.percentile(raw, 99.8), 0, 1)[window]
    y0, x0 = window[0].start, window[1].start
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), constrained_layout=True)
    panels = [('a  Untiled (reference)', untiled, None),
              ('b  Native distributed', native, ORANGE),
              ('c  Replay: relink', relinked, BLUE)]
    for ax, (title, lab, accent) in zip(axes, panels):
        ax.imshow(v, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        outline(ax, lab[window], '#bdbcb6', 0.5)
        if accent:
            pieces = np.unique(lab[np.isin(untiled, split_native)])
            outline(ax, lab[window], accent, 1.0, only=pieces[pieces > 0])
        for s in seams[0]:
            if window[0].start < s < window[0].stop:
                ax.axhline(s - y0 - 0.5, color='#fcfcfb', lw=0.6, ls=(0, (3, 3)))
        for s in seams[1]:
            if window[1].start < s < window[1].stop:
                ax.axvline(s - x0 - 0.5, color='#fcfcfb', lw=0.6, ls=(0, (3, 3)))
        ax.set_title(title, loc='left', color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    n_split = compare(untiled, native)['split']
    for ax, text in ((axes[1], f'{n_split} objects split (whole image)'), (axes[2], f"{compare(untiled, relinked)['split']} split, no inference")):
        ax.text(0.0, -0.03, text, transform=ax.transAxes, color=MUTED, fontsize=7, va='top')
    fig.savefig(OUT / 'fig2_case.png', facecolor='#fcfcfb')
    plt.close(fig)


def fig_cellpose():
    rs = results('cellpose')
    untiled_truth = np.mean([r['untiled_vs_truth']['f1'] for r in rs])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, key, title in ((axes[0], 'vs_untiled', 'a  F1 against untiled inference'),
                           (axes[1], 'vs_truth', 'b  F1 against BBBC039 annotation')):
        for i, blk in enumerate([128, 192, 256]):
            for j, (pol, color) in enumerate((('native', ORANGE), ('relink', BLUE))):
                vals = [next(b for b in r['blocks'] if b['block_size'] == blk)['policies'][pol][key]['f1'] for r in rs]
                x = i + (j - 0.5) * 0.36
                ax.scatter(x + rng.uniform(-0.09, 0.09, len(vals)), vals, s=9, color=color, alpha=0.55, lw=0, zorder=2)
                ax.plot([x - 0.13, x + 0.13], [np.mean(vals)] * 2, color=INK, lw=1.6, zorder=3, solid_capstyle='round')
        if key == 'vs_truth':
            ax.axhline(untiled_truth, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
            ax.text(2.55, untiled_truth, 'untiled\nmean', color=MUTED, fontsize=7, va='center', ha='left')
        ax.set_xticks(range(3), ['128 px\n30 blocks', '192 px\n12 blocks', '256 px\n9 blocks'])
        ax.set_xlim(-0.6, 2.95)
        ax.set_ylim(0.6, 1.01)
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_title(title, loc='left', color=INK)
        ax.set_ylabel('F1 (IoU ≥ 0.5)')
    axes[0].text(-0.45, 0.66, 'native', color=ORANGE, fontsize=7.5, fontweight='bold')
    axes[0].text(-0.45, 0.62, 'relink', color=BLUE, fontsize=7.5, fontweight='bold')
    fig.savefig(OUT / 'fig3_cellpose.png', facecolor='#fcfcfb')
    plt.close(fig)


def fig_stardist():
    rs = results('stardist')
    configs = [(320, 48), (352, 32), (352, 48), (384, 48), (384, 64)]   # 352/64 shares 352/48's geometry
    policies = ['native', 'center', 'iou_merge', 'crop', 'retain_all']
    fig, ax = plt.subplots(figsize=(3.45, 2.9), constrained_layout=True)
    for i, pol in enumerate(policies):
        for k, (b, o) in enumerate(configs):
            vals = [c['policies'][pol]['vs_untiled']['f1'] for r in rs for c in r['configs']
                    if c['block_size'] == b and c['min_overlap'] == o and c['status'] == 'ok']
            y = i + (k - 2) * 0.15   # rows within a policy: configs in listed order, top to bottom
            ax.scatter(vals, np.full(len(vals), y), s=5, color=BLUE, alpha=0.35, lw=0)
            ax.plot([np.mean(vals)] * 2, [y - 0.05, y + 0.05], color=INK, lw=1.4)
    ax.set_xlim(0.5, 1.02)
    ax.set_yticks(range(len(policies)), policies)
    ax.invert_yaxis()
    ax.set_xlabel('F1 against untiled inference')
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title('StarDist: one capture, five rules', loc='left', color=INK)
    fig.savefig(OUT / 'fig4_stardist.png', facecolor='#fcfcfb')
    plt.close(fig)


def fig_3d():
    rs = results('cellpose3d')
    policies = [('disable_edges', 'no links', MUTED), ('native', 'native = relink', ORANGE), ('relink_absorb', 'relink_absorb', BLUE)]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), constrained_layout=True)
    rng = np.random.default_rng(1)
    for ax, key, title in ((axes[0], 'vs_untiled', 'a  F1 against untiled 3D inference'),
                           (axes[1], 'vs_truth', 'b  F1 against annotation')):
        for i, kind in enumerate(['chain', 'grid']):
            for j, (pol, label, color) in enumerate(policies):
                vals = [r['blocks'][kind]['policies'][pol][key]['f1'] for r in rs if kind in r['blocks']]
                x = i + (j - 1) * 0.27
                ax.scatter(x + rng.uniform(-0.05, 0.05, len(vals)), vals, s=14, color=color, alpha=0.75, lw=0, zorder=2)
                ax.plot([x - 0.1, x + 0.1], [np.mean(vals)] * 2, color=INK, lw=1.6, zorder=3)
        n_chain = sum('chain' in r['blocks'] for r in rs)
        n_grid = sum('grid' in r['blocks'] for r in rs)
        ax.set_xticks([0, 1], [f'chain, 8–10 blocks\n{n_chain} volumes', f'grid, 32–36 blocks\n{n_grid} volumes'])
        ax.set_xlim(-0.55, 1.55)
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_title(title, loc='left', color=INK)
        ax.set_ylabel('F1 (IoU ≥ 0.5)')
    axes[0].set_ylim(0.68, 1.005)
    axes[1].set_ylim(0.45, 0.95)
    for k, (pol, label, color) in enumerate(policies):
        axes[0].text(0.62, 0.80 - k * 0.035, label, color=color if color != MUTED else MUTED, fontsize=7.5, fontweight='bold')
    fig.savefig(OUT / 'fig5_3d.png', facecolor='#fcfcfb')
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    fig_case(); fig_cellpose(); fig_stardist(); fig_3d()
    print(sorted(p.name for p in OUT.glob('*.png')))
