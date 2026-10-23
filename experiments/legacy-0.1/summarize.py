from pathlib import Path
import json,time,hashlib
import numpy as np
from stitchtrace import Bundle,replay,canonical,measurements
from handwritten import replay as baseline_replay
R=Path(__file__).resolve().parents[1];rows=[]
for group in ['finite','engineering512']:
 for base in sorted((R/'results'/group).iterdir()):
  if not base.is_dir():continue
  statuses={mode:json.loads((base/(mode+'-status.json')).read_text()) for mode in ['native','tool','baseline']}
  row={'group':group,'id':base.name,'stages':statuses}
  if all(s['status']=='completed' for s in statuses.values()):
   expected=np.load(base/'native/output.npy');b=Bundle.open(base/'tool');start=time.perf_counter();out,stats=replay(b);row['tool_replay_seconds']=time.perf_counter()-start
   start=time.perf_counter();hand=baseline_replay(base/'baseline');row['baseline_replay_seconds']=time.perf_counter()-start
   for a in [out,hand,b.array('expected.npy')]:
    assert np.array_equal(canonical(expected),canonical(a));assert measurements(canonical(expected))==measurements(canonical(a))
   alt,alt_stats=replay(b,'retain_all');hand_alt=baseline_replay(base/'baseline',True);assert np.array_equal(canonical(alt),canonical(hand_alt))
   row.update(partition_equal=True,measurements_equal=True,alternative_equal=True,native_objects=len(stats),alternative_objects=len(alt_stats),pixels_changed=int(np.count_nonzero(canonical(expected)!=canonical(alt))))
  rows.append(row)
(R/'results/comparison.json').write_text(json.dumps(rows,indent=2));print([(r['group'],r['id'],r.get('partition_equal'),r.get('native_objects'),r.get('alternative_objects')) for r in rows])
