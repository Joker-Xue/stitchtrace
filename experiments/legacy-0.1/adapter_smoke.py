from pathlib import Path
import json, numpy as np
from stitchtrace.backends.cellpose import capture_cellpose
from stitchtrace import replay, canonical
R=Path(__file__).resolve().parents[1];a=np.zeros((6,12,24),np.uint32);a[1:5,3:10,2:22]=1
b=capture_cellpose(a,lambda x,c:x,R/'results/smoke-cp',predictor_id='oracle-v1',block_size=(6,12,8),spacing=(2,.5,.5))
out,stats=replay(b);assert np.array_equal(canonical(out),canonical(a));print('CP smoke passed',len(stats))
