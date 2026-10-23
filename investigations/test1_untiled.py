"""test-1 investigation: untiled vs tiled StarDist on the same normalized image."""
import json, sys, numpy as np, tifffile, imageio.v3 as iio
from pathlib import Path
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from skimage.measure import label
R = Path(__file__).resolve().parents[1]
model = StarDist2D(None, name='2D_versatile_fluo', basedir=str(R/'models'))
def match(gt, pr):
    # count GT objects with IoU>=0.5 match (greedy, unique since IoU>0.5)
    from stardist.matching import matching
    m = matching(gt, pr, thresh=0.5)
    return dict(tp=int(m.tp), fp=int(m.fp), fn=int(m.fn), f1=round(m.f1, 4), panoptic=round(m.panoptic_quality, 4))
out = {}
for k in ['validation-0', 'validation-1', 'test-0', 'test-1']:
    img = normalize(tifffile.imread(R/'data/selected'/k/'image.tif'), 1, 99.8, axis=(0, 1))
    gt = label(iio.imread(R/'data/selected'/k/'mask.png')[..., 0])
    full, _ = model.predict_instances(img)
    tiled, _ = model.predict_instances_big(img, axes='YX', block_size=512, min_overlap=64, context=96, show_progress=False)
    out[k] = dict(gt=int(gt.max()), untiled=int(len(np.unique(full)) - 1), tiled512=int(len(np.unique(tiled)) - 1),
                  untiled_vs_gt=match(gt, full), tiled_vs_gt=match(gt, tiled), tiled_vs_untiled=match(full, tiled),
                  img_stats=dict(min=float(img.min()), max=float(img.max())))
    np.save(R/'investigations'/f'{k}-untiled.npy', full); np.save(R/'investigations'/f'{k}-tiled512.npy', tiled)
    print(k, json.dumps(out[k]), flush=True)
(R/'investigations/test1_untiled.json').write_text(json.dumps(out, indent=2))
