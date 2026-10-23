from pathlib import Path
import subprocess,json
R=Path(__file__).resolve().parents[1]
for item in json.loads((R/'research/selected-inputs.json').read_text()):
 p=R/'results'/('final-tool-'+item['id']+'.log')
 with p.open('x') as log:subprocess.run([str(R/'.capture-venv/bin/python'),str(R/'experiments/job.py'),'tool-final',item['id']],stdout=log,stderr=subprocess.STDOUT,check=True)
 print(item['id'],'final capture completed',flush=True)
