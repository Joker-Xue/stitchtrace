"""Assert OS denials, then replay previous evidence with the installed core wheel."""
from pathlib import Path
import json,runpy,socket,sys
previous=Path(sys.argv[1]).resolve()
weight=previous/'models/models/StarDist2D/2D_versatile_fluo/2D_versatile_fluo_extracted/weights_best.h5'
checks={}
try:
    with socket.socket() as connection:
        connection.connect(('127.0.0.1',9))
except PermissionError:
    checks['network_denied']=True
try:
    weight.read_bytes()
except PermissionError:
    checks['weights_denied']=True
assert checks=={'network_denied':True,'weights_denied':True},checks
print(json.dumps(checks),file=sys.stderr)
script=Path(__file__).with_name('verify_previous_release.py')
sys.argv=[str(script),str(previous)]
runpy.run_path(str(script),run_name='__main__')
