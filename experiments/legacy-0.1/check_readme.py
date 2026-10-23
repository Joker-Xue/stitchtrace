"""Execute the README quickstart verbatim using explicit local prerequisites."""
from pathlib import Path
import hashlib,json,os,subprocess
P=Path(__file__).resolve().parents[1]
readme=(P/'README.md').read_text()
script=readme.split('<!-- verified-quickstart -->',1)[1].split('~~~sh\n',1)[1].split('~~~',1)[0]
out=P/'results/revision-0.1.2'
(out/'quickstart-executed.sh').write_text(script)
with (out/'quickstart.log').open('w') as log:
 result=subprocess.run(['/bin/sh','-c',script],cwd=P,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
text=(out/'quickstart.log').read_text()
work=text.split('Quickstart outputs: ')[-1].strip() if result.returncode==0 else None
(out/'quickstart-result.json').write_text(json.dumps({'returncode':result.returncode,
 'readme_sha256':hashlib.sha256(readme.encode()).hexdigest(),'script_sha256':hashlib.sha256(script.encode()).hexdigest(),
 'work_directory':work,'explicit_cache':os.environ['STITCHTRACE_CACHE'],'previous_release':os.environ['STITCHTRACE_PREVIOUS_RELEASE'],
 'cold_network_download':False},indent=2))
print(json.dumps({'returncode':result.returncode,'work_directory':work}))
raise SystemExit(result.returncode)
