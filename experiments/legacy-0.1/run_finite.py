from pathlib import Path
import subprocess,json,os
R=Path(__file__).resolve().parents[1];python=R/'.capture-venv/bin/python'
items=json.loads((R/'research/selected-inputs.json').read_text())
for item in items:
 for mode in ['native','tool','baseline']:
  prefix='original' if os.environ.get('STITCHTRACE_EXPERIMENT_BLOCK','320')=='320' else 'engineering512'
  log=R/'results'/f'{prefix}-{item["id"]}-{mode}.log'
  if log.exists():raise RuntimeError('Refuse rerun/overwrite '+str(log))
  print('START',item['id'],mode,flush=True)
  with log.open('w') as stream:subprocess.run([str(python),str(R/'experiments/job.py'),mode,item['id']],stdout=stream,stderr=subprocess.STDOUT,check=True)
  print('END',item['id'],mode,flush=True)
