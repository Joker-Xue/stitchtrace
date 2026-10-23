"""New wheel-only environment under OS denial; no inference imports or weights."""
from pathlib import Path
import json,sys,socket,importlib.util,importlib.abc,time
import numpy as np
R=Path(__file__).resolve().parents[1]
assert all(importlib.util.find_spec(x) is None for x in ['tensorflow','torch','stardist','cellpose'])
class NoInference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ['tensorflow','torch','stardist','cellpose','keras']:
            raise ImportError('INFERENCE_DISABLED')
sys.meta_path.insert(0,NoInference())
checks={}
try:socket.create_connection(('1.1.1.1',443),timeout=1)
except PermissionError:checks['network_denied']=True
weight=R/'models/models/StarDist2D/2D_versatile_fluo/2D_versatile_fluo_extracted/weights_best.h5'
try:weight.read_bytes()
except PermissionError:checks['weights_denied']=True
assert checks=={'network_denied':True,'weights_denied':True}
from stitchtrace import Bundle,replay,canonical,measurements
from handwritten import replay as manual_replay
rows=[]
for group in ['engineering512','oracles']:
 for folder in sorted((R/'results'/group).iterdir()):
  if not folder.is_dir() or not (folder/'tool/manifest.json').exists():continue
  tool=folder/('tool-final' if (folder/'tool-final').exists() else 'tool')
  t=time.perf_counter();out,stats=replay(tool);seconds=time.perf_counter()-t
  hand=manual_replay(folder/'baseline');assert np.array_equal(canonical(out),canonical(hand))
  b=Bundle.open(tool);spacing=b.meta['spacing'];assert measurements(canonical(out),spacing)==measurements(canonical(hand),spacing)
  policy='retain_all' if b.meta['backend']=='ownership' else 'disable_edges'
  alt,_=replay(b,policy);manual_alt=manual_replay(folder/'baseline',True);assert np.array_equal(canonical(alt),canonical(manual_alt))
  rows.append({'case':folder.name,'group':group,'native_equal':True,'alternative_equal':True,'measurements_equal':True,'tool_replay_seconds':seconds})
assert len(rows)==12
for folder in sorted((R/'results/sd3d-oracle').iterdir()):
 if not folder.is_dir() or folder.name.startswith('.'):continue
 b=Bundle.open(folder);out,stats=replay(b);assert np.array_equal(canonical(out),canonical(b.array('expected.npy')))
 rows.append({'case':folder.name,'group':'sd3d-labeloracle','native_equal':True,'measurements_equal':True})
assert len(rows)==15
(R/'results/offline-audit.json').write_text(json.dumps({'isolation':checks,'installed_package':__import__('stitchtrace').__file__,'python':sys.executable,'rows':rows},indent=2));print('12 paired native/alternative cases plus3 StarDist3D labeloracle replays passed with wheel/NumPy only.')
