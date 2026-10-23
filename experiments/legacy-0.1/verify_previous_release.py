"""Read-only compatibility replay against an explicit 0.1.0 evidence directory."""
import argparse,json,importlib.util
from pathlib import Path
import numpy as np
from stitchtrace import Bundle,replay,canonical
parser=argparse.ArgumentParser()
parser.add_argument('previous_release',type=Path)
args=parser.parse_args()
assert all(importlib.util.find_spec(x) is None for x in ('tensorflow','torch','stardist','cellpose'))
root=args.previous_release
bundles=list((root/'results/engineering512').glob('*/tool-final'))+list((root/'results/oracles').glob('*/tool'))+[root/'results/sd3d-oracle'/str(i) for i in range(3)]
rows=[]
for path in sorted(bundles):
 b=Bundle.open(path);out,_=replay(b)
 assert np.array_equal(canonical(out),canonical(b.array('expected.npy')))
 rows.append(str(path.relative_to(root)))
assert len(rows)==15
print(json.dumps({'version':__import__('stitchtrace').__version__,'installed_package':__import__('stitchtrace').__file__,'passed':len(rows),'bundles':rows},indent=2))
