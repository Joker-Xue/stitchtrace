"""Installed 3D assembly only; explicit label oracle, no neural model."""
from pathlib import Path
from types import SimpleNamespace
from collections import namedtuple
import json,numpy as np
from stardist.models.base import StarDistBase
from stitchtrace.backends.stardist import capture_stardist
from stitchtrace import canonical,replay,measurements
R=Path(__file__).resolve().parents[1];a=np.zeros((12,40,72),np.uint32);a[2:8,5:13,19:37]=1;a[2:8,23:31,35:53]=2
class Oracle:
    _axes_out='ZYX'
    predict_instances_big=StarDistBase.predict_instances_big
    def __init__(self,seed):
        self.config=SimpleNamespace(seed=seed,scope='labeloracle-no-neural-inference')
        self.keras_model=SimpleNamespace(get_weights=lambda:[])
        self.thresholds=namedtuple('Thresholds','prob nms')(.5,.5)
    def _axes_tile_overlap(self,axes):return (0,)*3
    def _axes_div_by(self,axes):return (1,)*3
    def predict_instances(self,image,**kw):
        ids=np.unique(image);ids=ids[ids>0];perm=np.random.default_rng(self.config.seed).permutation(np.arange(1,len(ids)+1));out=np.zeros_like(image)
        for i,j in zip(ids,perm):out[image==i]=j
        return out,{'prob':np.ones(len(ids)),'points':np.array([np.argwhere(out==i).mean(0) for i in range(1,len(ids)+1)]).reshape(-1,3)}
params={'block_size':(12,40,48),'min_overlap':(9,20,24),'context':0};rows=[]
for seed in [0,1,2]:
    model=Oracle(seed);native,_=model.predict_instances_big(a,axes='ZYX',show_progress=False,**params)
    b=capture_stardist(model,a,R/'results/sd3d-oracle'/str(seed),model_id='LABEL_ORACLE_ONLY',spacing=(2,.5,.5),**params)
    out,stats=replay(b);assert np.array_equal(canonical(native),canonical(out));assert np.array_equal(canonical(out),canonical(a));assert measurements(canonical(native),(2,.5,.5))==measurements(canonical(out),(2,.5,.5))
    rows.append({'seed':seed,'partition_equal':True,'measurements_equal':True,'objects':len(stats),'scope':'3D labeloracle only'})
(R/'results/sd3d-oracle-summary.json').write_text(json.dumps(rows,indent=2));print(rows)
