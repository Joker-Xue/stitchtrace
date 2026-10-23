import os
import json
import multiprocessing
from pathlib import Path
import numpy as np
import pytest
from stitchtrace.storage import transaction, atomic_array, commit, cache_array
from stitchtrace import Bundle, StitchTraceError


def crash(destination):
    with transaction(destination, 'crash') as root:
        cache_array(root, {'tile':0}, lambda: np.ones((2,2),np.uint32))
        os._exit(17)


def test_real_process_death_lock_released(tmp_path):
    p=tmp_path/'job';ctx=multiprocessing.get_context('spawn');worker=ctx.Process(target=crash,args=(p,));worker.start();worker.join(10)
    assert worker.exitcode==17
    with transaction(p,'crash') as root:
        a,hit=cache_array(root,{'tile':0},lambda: (_ for _ in ()).throw(RuntimeError('not reusable')));assert hit
        atomic_array(root/'expected.npy',a);atomic_array(root/'packed.npy',a)
        commit(root,{'schema':1,'backend':'graph','contract':'crash','shape':[2,2],'axes':'YX','spacing':[1,1],'nodes':[1],'edges':[]})
    assert Bundle.open(p).meta['contract']=='crash'


def test_pending_contract_change_refused(tmp_path):
    with pytest.raises(RuntimeError):
        with transaction(tmp_path/'x','old') as root:raise RuntimeError('stop')
    with pytest.raises(StitchTraceError,match='CONTRACT_CONFLICT'):
        with transaction(tmp_path/'x','new'):pass
