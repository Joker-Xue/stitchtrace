"""Audit M1-M4 counterexamples; authored before fixes."""
from pathlib import Path
import json,os,subprocess,sys
import numpy as np
import pytest
from stitchtrace import Bundle,StitchTraceError,replay
from stitchtrace.storage import atomic_array,atomic_bytes,commit,encoded


def valid(root):
    root.mkdir();a=np.ones((2,2),np.uint32)
    for key in ['local','selected','written','expected']:atomic_array(root/(key+'.npy'),a)
    np.savez(root/'raw.npz',labels=a)
    meta={'schema':1,'backend':'ownership','contract':'audit','shape':[2,2],'axes':'YX','spacing':[1,1],
          'order':[0],'tiles':[{'id':0,'read':[[0,2],[0,2]],'write':[[0,2],[0,2]],'cache':'raw.npz',
          'local':'local.npy','selected':'selected.npy','written':'written.npy','kept':[1],'rejected':[]}]}
    commit(root,meta);return meta


def test_m1_root_manifest_symlink(tmp_path):
    root=tmp_path/'b';valid(root);external=tmp_path/'external.json';(root/'manifest.json').rename(external)
    (root/'manifest.json').symlink_to(external.resolve())
    with pytest.raises(StitchTraceError,match='SYMLINK'):Bundle.open(root)


def test_m1_undeclared_nested_manifest(tmp_path):
    root=tmp_path/'b';valid(root);(root/'nested').mkdir();(root/'nested/manifest.json').write_text('untracked')
    with pytest.raises(StitchTraceError,match='UNDECLARED_FILE'):Bundle.open(root)


def test_m1_commit_includes_nested_manifest(tmp_path):
    root=tmp_path/'b';m=valid(root);(root/'nested').mkdir();(root/'nested/manifest.json').write_text('{}');commit(root,m)
    assert 'nested/manifest.json' in json.loads((root/'manifest.json').read_text())['files']


@pytest.mark.parametrize('kind',['read_bounds','containment','raw_shape','raw_crop','partial_object'])
def test_m2_semantics(tmp_path,kind):
    root=tmp_path/'b';m=valid(root)
    if kind=='read_bounds':m['tiles'][0]['read']=[[-999,999999],[-100,999999]]
    elif kind=='containment':m['tiles'][0]['read']=[[0,1],[0,2]]
    elif kind=='raw_shape':np.savez(root/'raw.npz',labels=np.ones((3,2),np.uint32))
    elif kind=='raw_crop':np.savez(root/'raw.npz',labels=np.full((2,2),2,np.uint32))
    else:
        a=np.ones((2,2),np.uint32);a[0,0]=0
        for key in ['selected','written','expected']:atomic_array(root/(key+'.npy'),a)
    commit(root,m)
    with pytest.raises(StitchTraceError):replay(root)


@pytest.mark.parametrize('value',[[],None,3,'text',True,{'schema':1,'files':{'bundle.json':[]}}])
def test_m3_manifest_types(tmp_path,value):
    root=tmp_path/'b';root.mkdir();(root/'manifest.json').write_text(json.dumps(value))
    with pytest.raises(StitchTraceError):Bundle.open(root)


def test_m3_cli_list_is_json_exit2(tmp_path):
    (tmp_path/'manifest.json').write_text('[]')
    r=subprocess.run([sys.executable,'-m','stitchtrace.cli','verify',str(tmp_path)],capture_output=True,text=True)
    assert r.returncode==2
    assert isinstance(json.loads(r.stderr)['error'],str)
    assert 'Traceback' not in r.stderr


def test_m4_explicit_dependency_preflight_contract():
    script=Path(__file__).resolve().parents[1]/'experiments/legacy-0.1/reproduce.py'
    r=subprocess.run([sys.executable,str(script),'--help'],capture_output=True,text=True)
    assert r.returncode==0
    assert '--cache-dir' in r.stdout and '--preflight-only' in r.stdout

def test_m1_special_entry(tmp_path):
    root=tmp_path/'b';valid(root);os.mkfifo(root/'pipe')
    with pytest.raises(StitchTraceError,match='NONREGULAR_FILE'):Bundle.open(root)


@pytest.mark.parametrize('value',[[],None,3,'text',True])
def test_m3_metadata_types(tmp_path,value):
    root=tmp_path/'b';valid(root);commit(root,value)
    with pytest.raises(StitchTraceError):Bundle.open(root)


@pytest.mark.parametrize('field,value',[('tiles',[None]),('tiles',{}),('order',{}),('shape',None),('axes',[]),('spacing',None)])
def test_m3_nested_types(tmp_path,field,value):
    root=tmp_path/'b';m=valid(root);m[field]=value;commit(root,m)
    with pytest.raises(StitchTraceError):Bundle.open(root)


def test_m4_empty_cache_is_coded_and_no_output(tmp_path):
    script=Path(__file__).resolve().parents[1]/'experiments/legacy-0.1/reproduce.py'
    env=os.environ.copy()
    env['PATH']=str(Path.home()/'.local/bin')+os.pathsep+env['PATH']
    # This explicit empty cache must override any warm inherited UV_CACHE_DIR.
    r=subprocess.run([sys.executable,str(script),'--name','must-not-exist-empty-cache',
                      '--cache-dir',str(tmp_path/'empty')],capture_output=True,text=True,env=env)
    assert r.returncode==2
    assert json.loads(r.stderr)['error']=='DEPENDENCY_PREFLIGHT'
    assert not (script.parents[1]/'reproductions/must-not-exist-empty-cache').exists()
