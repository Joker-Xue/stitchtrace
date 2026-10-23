from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
import json,traceback
import numpy as np,tifffile
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from stitchtrace.backends.stardist import capture_stardist
from stitchtrace import replay,canonical,StitchTraceError
R=Path(__file__).resolve().parents[1];out=R/'results/recovery';out.mkdir();report={}
try:
 model=StarDist2D.from_pretrained('2D_versatile_fluo');raw=tifffile.imread(R/'data/development-fixture-img2d.tif');a=normalize(np.pad(raw,64,mode='reflect'),1,99.8,axis=(0,1));native=model.predict_instances;calls=[]
 def fail_second(*args,**kwargs):
  calls.append(1)
  if len(calls)==2:raise RuntimeError('INJECTED_SECOND_TILE_FAILURE')
  return native(*args,**kwargs)
 try:
  with patch.object(model,'predict_instances',fail_second):capture_stardist(model,a,out/'resumed',model_id='official-fluo')
 except RuntimeError as e:report['injected_failure']=str(e)
 calls=[]
 def count(*args,**kwargs):calls.append(1);return native(*args,**kwargs)
 with patch.object(model,'predict_instances',count):b=capture_stardist(model,a,out/'resumed',model_id='official-fluo')
 report['resume_predict_calls']=len(calls);report['tiles']=len(b.meta['tiles']);assert len(calls)==len(b.meta['tiles'])-1
 with patch.object(model,'predict_instances',side_effect=AssertionError('no inference permitted on idempotent rerun')):
  capture_stardist(model,a,out/'resumed',model_id='official-fluo')
 report['idempotent_no_inference']=True
 def capture(name):return replay(capture_stardist(model,a,out/name,model_id='official-fluo'))[0]
 with ThreadPoolExecutor(2) as pool:values=list(pool.map(capture,['thread-a','thread-b']))
 assert np.array_equal(canonical(values[0]),canonical(values[1]));report['concurrent_calls_isolated']=True
 weights=model.keras_model.get_weights();changed=[w.copy() for w in weights];changed[0].flat[0]+=1e-3;model.keras_model.set_weights(changed)
 try:capture_stardist(model,a,out/'resumed',model_id='official-fluo')
 except StitchTraceError as e:assert e.code=='CONTRACT_CONFLICT';report['weight_change_invalidated']=True
 else:raise AssertionError('weights not invalidated')
 model.keras_model.set_weights(weights)
 report['status']='passed'
except Exception:report.update(status='failed',traceback=traceback.format_exc());raise
finally:(out/'summary.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
