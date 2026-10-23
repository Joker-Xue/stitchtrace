"""Figure 1: architecture schematic (drawn, not data)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import RUNS

INK, MUTED, LINE, SURFACE = '#0b0b0b', '#52514e', '#9a9893', '#fcfcfb'
BLUE, BLUE_BG, ORANGE_BG, GREY_BG = '#2a78d6', '#e8f1fb', '#fdeee7', '#f1f0ec'


def box(ax, x, y, w, h, title, lines, fill, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.015',
                                fc=fill, ec=edge, lw=1.0))
    ax.text(x + 0.012, y + h - 0.03, title, fontsize=7.6, fontweight='bold', color=INK, va='top')
    for i, line in enumerate(lines):
        ax.text(x + 0.012, y + h - 0.08 - i * 0.05, line, fontsize=6.4, color=INK, va='top')


def arrow(ax, x0, y0, x1, y1, text=None):
    ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', color=LINE, lw=1.1, shrinkA=0, shrinkB=0))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.022, text, fontsize=6.5, color=MUTED, ha='center')


def main():
    fig, ax = plt.subplots(figsize=(7.1, 2.9))
    ax.set_xlim(0, 1.015); ax.set_ylim(0, 1); ax.axis('off')
    fig.patch.set_facecolor(SURFACE)

    ax.text(0.01, 0.98, 'Capture  (StarDist / Cellpose installed)', fontsize=7.5, color=MUTED, va='top')
    box(ax, 0.01, 0.50, 0.28, 0.40, 'StarDist 0.9.2 adapter',
        ['wraps predict_instances_big:', 'read · predict · filter · write', 'records ownership decisions'], ORANGE_BG, '#e8a080')
    box(ax, 0.01, 0.04, 0.28, 0.40, 'Cellpose 4.2.1.1 adapter',
        ['runs native distributed_eval', 'block predictor gets copied crops', 'records face graph, relabeling'], ORANGE_BG, '#e8a080')

    box(ax, 0.355, 0.04, 0.29, 0.86, 'Bundle (schema 1)',
        ['manifest: sizes + SHA-256', 'run contract: image, geometry,', '  upstream source, predictor id',
         'raw prediction of every block', 'assembly decisions', '  ownership: crop, selection, writes',
         '  graph: packed labels, edges', 'native output', 'atomic, resumable, reusable'], GREY_BG, LINE)

    ax.text(0.705, 0.98, 'Core  (NumPy only, no model or GPU)', fontsize=7.5, color=MUTED, va='top')
    box(ax, 0.705, 0.70, 0.29, 0.20, 'verify', ['files, digests, shapes, invariants'], BLUE_BG, BLUE)
    box(ax, 0.705, 0.37, 0.29, 0.29, 'replay(policy)',
        ['native (exact, checked)', 'crop, iou_merge, center, relink', '+ user-registered policies'], BLUE_BG, BLUE)
    box(ax, 0.705, 0.04, 0.29, 0.29, 'compare · attribute',
        ['object matching, splits, merges', 'seam_split / seam_other / interior'], BLUE_BG, BLUE)

    arrow(ax, 0.298, 0.70, 0.347, 0.70, 'write')
    arrow(ax, 0.298, 0.24, 0.347, 0.24, 'write')
    for y in (0.80, 0.515, 0.185):
        arrow(ax, 0.652, y, 0.697, y)
    fig.savefig(RUNS / 'figures' / 'fig1_architecture.png', dpi=300, facecolor=SURFACE, bbox_inches='tight', pad_inches=0.03)


if __name__ == '__main__':
    main()
