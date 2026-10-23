from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import numpy as np
import pytest
from stitchtrace import Bundle, canonical, measurements, replay, StitchTraceError
from stitchtrace.storage import transaction, atomic_array, atomic_bytes, encoded, commit, cache_array


def graph(path, shape=(6, 9), spacing=None, edges=None):
    a = np.zeros(shape, np.uint32)
    a[..., 1:3] = 11
    a[..., 3:6] = 22
    a[..., 6:8] = 33
    with transaction(path, 'fixed') as root:
        if root is not None:
            atomic_array(root/'packed.npy', a)
            atomic_array(root/'expected.npy', (a>0).astype(np.uint32))
            commit(root, {'schema':1,'backend':'graph','contract':'fixed','shape':list(shape),
                'axes':'YX' if len(shape)==2 else 'ZYX','spacing':spacing or [1]*len(shape),
                'nodes':[11,22,33],'edges':edges if edges is not None else [[11,22],[22,33]]})
    return Bundle.open(path)


@pytest.mark.parametrize('shape,spacing', [((6,9),[.5,2]),((3,6,9),[2,.5,.5])])
def test_closure_and_anisotropic(tmp_path,shape,spacing):
    b=graph(tmp_path/'b',shape,spacing)
    out,stats=replay(b)
    assert len(stats)==1 and next(iter(stats.values()))['physical_size']==np.count_nonzero(out)*np.prod(spacing)
    assert len(replay(b,'disable_edges')[1])==3
    np.testing.assert_allclose(next(iter(stats.values()))['centroid'],np.argwhere(out>0).mean(0)*spacing)


def test_id_and_edge_order(tmp_path):
    b=graph(tmp_path/'a');c=graph(tmp_path/'b',edges=[[33,22],[22,11]])
    assert np.array_equal(replay(b)[0],replay(c)[0])
    a=b.array('packed.npy');renamed=np.where(a==11,999,np.where(a==22,2,np.where(a==33,7,0)))
    assert np.array_equal(canonical(a),canonical(renamed))


def test_idempotency_conflict_and_threads(tmp_path):
    p=tmp_path/'b'
    with ThreadPoolExecutor(4) as pool:
        all_b=list(pool.map(lambda _:graph(p),range(8)))
    assert all(b.meta['contract']=='fixed' for b in all_b)
    with pytest.raises(StitchTraceError,match='CONTRACT_CONFLICT'):
        with transaction(p,'changed'):pass


def test_interruption_resume_cache(tmp_path):
    p=tmp_path/'b';calls=[]
    with pytest.raises(RuntimeError):
        with transaction(p,'fixed') as root:
            cache_array(root,{'run':'fixed','tile':1},lambda: calls.append(1) or np.ones((2,2),np.uint32))
            raise RuntimeError('deliberate interruption')
    with transaction(p,'fixed') as root:
        value,hit=cache_array(root,{'run':'fixed','tile':1},lambda: (_ for _ in ()).throw(AssertionError('must not recompute')))
        assert hit and len(calls)==1
        atomic_array(root/'packed.npy',value);atomic_array(root/'expected.npy',value)
        commit(root,{'schema':1,'backend':'graph','contract':'fixed','shape':[2,2],'axes':'YX','spacing':[1,1],'nodes':[1],'edges':[]})
    assert len(list((p/'failures').glob('*.json')))==1
    with transaction(tmp_path/'new','new') as root:
        _,hit=cache_array(root,{'run':'new','tile':1},lambda:np.ones((2,2),np.uint32));assert not hit
        atomic_array(root/'packed.npy',value);atomic_array(root/'expected.npy',value)
        commit(root,{'schema':1,'backend':'graph','contract':'new','shape':[2,2],'axes':'YX','spacing':[1,1],'nodes':[1],'edges':[]})


@pytest.mark.parametrize('kind',['missing','corrupt','traversal','symlink','schema','closure'])
def test_rejections(tmp_path,kind):
    b=graph(tmp_path/'b');p=b.path
    if kind=='missing':(p/'packed.npy').unlink()
    elif kind=='corrupt':(p/'packed.npy').write_bytes(b'bad')
    elif kind=='traversal':
        m=json.loads((p/'manifest.json').read_text());m['files']['../outside']={'bytes':1,'sha256':'bad'};(p/'manifest.json').write_text(json.dumps(m))
    elif kind=='symlink':
        (p/'packed.npy').rename(tmp_path/'external.npy');(p/'packed.npy').symlink_to(tmp_path/'external.npy')
    else:
        m=json.loads((p/'bundle.json').read_text())
        if kind=='schema':m['schema']=99
        else:m['nodes']=[11,22]
        commit(p,m)
    with pytest.raises(StitchTraceError):Bundle.open(p)


def test_cache_corruption_and_invalid_labels(tmp_path):
    _,hit=cache_array(tmp_path,{'k':1},lambda:np.ones((2,2),np.uint32));assert not hit
    next((tmp_path/'cache').glob('*.npz')).write_bytes(b'corrupt')
    with pytest.raises(StitchTraceError,match='CACHE_CORRUPT'):cache_array(tmp_path,{'k':1},lambda:np.ones((2,2)))
    for a in [np.zeros((2,2),float),np.full((2,2),-1),np.ones(4)]:
        with pytest.raises(StitchTraceError):measurements(a)
    with pytest.raises(StitchTraceError):measurements(np.ones((2,2),np.uint32),[1,-1])


def ownership(path):
    with transaction(path,'ownership') as root:
        a=np.array([[0,1,1,0],[0,2,2,0]],np.uint32)
        selected=a.copy();selected[selected==2]=0
        atomic_array(root/'local.npy',a);atomic_array(root/'selected.npy',selected);atomic_array(root/'written.npy',selected);atomic_array(root/'expected.npy',selected)
        commit(root,{'schema':1,'backend':'ownership','contract':'ownership','shape':[2,4],'axes':'YX','spacing':[1,1],
            'order':[0],'tiles':[{'id':0,'cache':'local.npy','read':[[0,2],[0,4]],'write':[[0,2],[0,4]],'local':'local.npy','selected':'selected.npy','written':'written.npy','kept':[1],'rejected':[2]}]})
    return Bundle.open(path)


def test_ownership_rejected_and_intervention(tmp_path):
    b=ownership(tmp_path/'b');assert len(replay(b)[1])==1;assert len(replay(b,'retain_all')[1])==2
    with pytest.raises(StitchTraceError):replay(b,'disable_edges')
    m=b.meta;m['tiles'][0]['write']=[[0,200],[0,4]];commit(b.path,m)
    with pytest.raises(StitchTraceError,match='INVALID_BOUNDS'):Bundle.open(b.path)

@pytest.mark.parametrize('change',['contract','node_type'])
def test_required_identity_and_node_types(tmp_path,change):
    b=graph(tmp_path/'b');m=b.meta
    if change=='contract':del m['contract']
    else:m['nodes']=[11.0,22,33]
    commit(b.path,m)
    with pytest.raises(StitchTraceError):Bundle.open(b.path)
