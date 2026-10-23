"""One finite real-image stage per subprocess; no image replacement or tuning."""
from pathlib import Path
import json,sys,time,resource,traceback,hashlib,os
import numpy as np
R=Path(__file__).resolve().parents[1]
mode,identifier=sys.argv[1:3];block=int(os.environ.get('STITCHTRACE_EXPERIMENT_BLOCK','320'));cohort='finite' if block==320 else 'engineering512';folder=R/'results'/cohort/identifier/mode;folder.parent.mkdir(parents=True,exist_ok=True)
report={'mode':mode,'id':identifier,'rss_scope':'whole fresh subprocess incl imports/model; macOS bytes','timing_scope':'after model load, includes first prediction compilation and stage I/O','status':'started'}
start=time.perf_counter()
try:
 import tensorflow as tf,tifffile
 from csbdeep.utils import normalize
 from stardist.models import StarDist2D
 tf.config.threading.set_inter_op_parallelism_threads(2);tf.config.threading.set_intra_op_parallelism_threads(2)
 item=next(x for x in json.loads((R/'research/selected-inputs.json').read_text()) if x['id']==identifier)
 raw=tifffile.imread(R/item['image']['path']);image=normalize(raw,1,99.8,axis=(0,1));model=StarDist2D.from_pretrained('2D_versatile_fluo')
 report['setup_seconds']=time.perf_counter()-start;start=time.perf_counter()
 if mode=='native':
  folder.mkdir();out,polys=model.predict_instances_big(image,axes='YX',block_size=block,min_overlap=64,context=96,show_progress=False);np.save(folder/'output.npy',out)
 elif mode.startswith('tool'):
  from stitchtrace.backends.stardist import capture_stardist
  bundle=capture_stardist(model,image,folder,block_size=block,model_id='weights:42202bd269c8106782316f1a2c75afb3f5ffa65e525c2e155ee0ced3a95da349;config:2D_versatile_fluo-v0.1;normalize-global-1-99.8')
  out=bundle.array('expected.npy')
 else:
  from handwritten import capture_sd
  out=capture_sd(model,image,folder,block_size=block)
 report.update(status='completed',objects=int(np.count_nonzero(np.unique(out))),shape=list(out.shape),array_sha256=hashlib.sha256(out.tobytes()).hexdigest())
except Exception as e:
 report.update(status='failed',error=repr(e),traceback=traceback.format_exc())
finally:
 report['seconds']=time.perf_counter()-start;report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;report['storage_bytes']=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()) if folder.exists() else 0
 (folder.parent/(mode+'-status.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
