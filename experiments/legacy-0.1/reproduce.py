"""Re-run in a new project-local folder, never replace recorded evidence.
Needs uv, POSIX/macOS for OS isolation, cached PyPI wheels, and shipped local data/model.
"""
from pathlib import Path
import argparse,os,shutil,subprocess,sys,json,hashlib
P=Path(__file__).resolve().parents[1]
from prepare_dependencies import prepare
parser=argparse.ArgumentParser()
parser.add_argument('--name')
parser.add_argument('--cache-dir', required=True)
parser.add_argument('--preflight-only', action='store_true')
args=parser.parse_args()
try:
 prepare(args.cache_dir)
except (RuntimeError, OSError) as error:
 print(json.dumps({'error':'DEPENDENCY_PREFLIGHT','detail':str(error)}),file=sys.stderr)
 sys.exit(2)
if args.preflight_only:
 print(json.dumps({'status':'ready','cache_dir':str(Path(args.cache_dir).resolve())}))
 sys.exit(0)
if not args.name:parser.error('--name is required unless --preflight-only')
os.environ['UV_CACHE_DIR']=str(Path(args.cache_dir).resolve())
os.environ['UV_PYTHON_DOWNLOADS']='never'
if not args.name.replace('-','').replace('_','').isalnum():raise ValueError('simple folder name required')
if sys.version_info[:2] != (3,11):raise RuntimeError('Use Python3.11 for locked reproduction')
R=P/'reproductions'/args.name
if R.exists():raise FileExistsError('Refuse overwrite '+str(R))
R.mkdir(parents=True)
for name in ['src','tests','experiments','research','docs','models']:
 shutil.copytree(P/name,R/name,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(P/'data/selected',R/'data/selected')
shutil.copy(P/'data/development-fixture-img2d.tif',R/'data/development-fixture-img2d.tif')
for name in ['pyproject.toml','requirements-capture.lock','requirements-core.lock']:shutil.copy(P/name,R/name)
(R/'results').mkdir()
for item in json.loads((R/'research/selected-inputs.json').read_text()):
 for kind in ['image','mask']:
  assert hashlib.sha256((R/item[kind]['path']).read_bytes()).hexdigest()==item[kind]['sha256']
env=os.environ.copy();env.update(KERAS_HOME=str(R/'models'),MPLCONFIGDIR=str(R/'results/mpl-cache'),NUMBA_CACHE_DIR=str(R/'results/numba-cache'))
def run(cmd,log):
 with (R/'results'/log).open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
run(['uv','venv','--python',sys.executable,str(R/'.capture-venv')],'setup-venv.log')
py=str(R/'.capture-venv/bin/python')
run(['uv','pip','install','--offline','--python',py,'-r',str(R/'requirements-capture.lock')],'setup-capture.log')
run(['uv','pip','install','--offline','--python',py,'--no-build-isolation','-e',str(R)],'setup-package.log')
run([py,'-m','pytest',str(R/'tests'),'-q'],'tests.log')
run([py,str(R/'experiments/run_finite.py')],'finite-progress.log')
env['STITCHTRACE_EXPERIMENT_BLOCK']='512'
run([py,str(R/'experiments/run_finite.py')],'engineering512-progress.log')
for script in ['oracles','stardist_oracle','recovery_check','reference_metrics','summarize']:
 run([py,str(R/'experiments'/(script+'.py'))],script+'.log')
run(['uv','venv','--python',sys.executable,str(R/'.audit-venv')],'setup-audit-venv.log')
audit=str(R/'.audit-venv/bin/python')
run(['uv','build',str(R),'--wheel','--offline','--no-build-isolation','--python',py],'build-wheel.log')
run(['uv','pip','install','--offline','--python',audit,str(R/'dist/stitchtrace-0.1.2-py3-none-any.whl')],'setup-audit.log')
profile=(R/'experiments/offline.sb').read_text().replace(str(P),str(R));(R/'experiments/offline-local.sb').write_text(profile)
if sys.platform!='darwin':raise RuntimeError('OS isolation audit requires macOS; earlier results preserved')
run(['sandbox-exec','-f',str(R/'experiments/offline-local.sb'),audit,str(R/'experiments/offline_audit.py')],'offline-audit.log')
print('Completed isolated reproduction:',R)
