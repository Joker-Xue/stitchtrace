from pathlib import Path
from unittest.mock import patch
import json,traceback,time
import numpy as np
from stitchtrace.backends.cellpose import capture_cellpose
from stitchtrace import replay,canonical,measurements
from handwritten import capture_cp
R=Path(__file__).resolve().parents[1];records=[]
for ndim in [2,3]:
 shape=(40,72) if ndim==2 else (12,40,72);spacing=[.5,.5] if ndim==2 else [2,.5,.5]
 for case in ['clean','thin','input_error','chain']:
  a=np.zeros(shape,np.uint32)
  if ndim==2:
   if case=='thin':a[18:21,8:64]=1
   elif case=='input_error':a[6:17,3:15]=1;a[20:31,3:15]=1
   else:a[6:17,8:64]=1;a[20:31,8:64]=2 if case=='clean' else 1
  else:
   if case=='thin':a[2:10,18:21,8:64]=1
   elif case=='input_error':a[2:10,6:17,3:15]=1;a[2:10,20:31,3:15]=1
   else:a[2:10,6:17,8:64]=1;a[2:10,20:31,8:64]=2 if case=='clean' else 1
  ident=f'{ndim}d-{case}';base=R/'results/oracles'/ident;base.mkdir(parents=True);block=tuple(shape[:-1])+(24,);start=time.perf_counter()
  try:
   original=capture_cp(a,base/'baseline',block)
   b=capture_cellpose(a,lambda x,c:x.copy(),base/'tool',predictor_id='labeloracle-exact-v1',block_size=block,spacing=spacing)
   out,stats=replay(b);assert np.array_equal(canonical(out),canonical(original));assert measurements(canonical(out),spacing)==measurements(canonical(original),spacing)
   # Runtime enumeration reversal: independent blocks must retain partition/measurements.
   from cellpose.contrib import distributed_segmentation as cp
   native=cp.get_block_crops
   def reverse(*args,**kwargs):
    ids,crops=native(*args,**kwargs);return ids[::-1],crops[::-1]
   with patch.object(cp,'get_block_crops',reverse):
    rev=capture_cellpose(a,lambda x,c:x.copy(),base/'tool-reverse',predictor_id='labeloracle-exact-v1',block_size=block,spacing=spacing)
   assert np.array_equal(canonical(replay(rev)[0]),canonical(out))
   records.append({'id':ident,'status':'passed','baseline_equal':True,'order_invariant':True,'input_objects':int(np.count_nonzero(np.unique(a))),'output_objects':len(stats),'spacing':spacing,'seconds':time.perf_counter()-start})
  except Exception:records.append({'id':ident,'status':'failed','traceback':traceback.format_exc()})
  (R/'results/oracle-summary.json').write_text(json.dumps(records,indent=2));print(records[-1],flush=True)
