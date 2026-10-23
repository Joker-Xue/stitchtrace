"""Independent complete-script baseline. No stitchtrace import or bundle input.
Own capture layout, manifest, compatibility check and offline replay.
"""
from pathlib import Path
import hashlib
import json
import numpy as np


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda x:int(x)))


def check_source(package):
    import importlib, importlib.metadata
    lock = json.loads(Path(__file__).with_name('baseline-lock.json').read_text())
    assert importlib.metadata.version(package) == lock[package]['version']
    for name, expected in lock[package]['files'].items():
        module = importlib.import_module(name)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected


def seal(folder, record):
    dump(folder/'record.json',record)
    dump(folder/'hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file() and p.name!='hashes.json'})


def capture_sd(model, image, folder, block_size=320):
    check_source('stardist')
    from stardist.big import BlockND
    from unittest.mock import patch
    folder=Path(folder);folder.mkdir(parents=True)
    read,filter_fn,write,predict=BlockND.read,BlockND.filter_objects,BlockND.write,model.predict_instances
    active={};records=[]
    def rd(block,x,axes=None):
        result=read(block,x,axes);active['id']=block.id;return result
    def pr(*args,**kwargs):
        result=predict(*args,**kwargs)
        np.savez_compressed(folder/f'raw-{active["id"]}.npz',labels=result[0],**result[1]);return result
    def fl(block,labels,polys,axes=None):
        result=filter_fn(block,labels,polys,axes)
        np.save(folder/f'local-{block.id}.npy',labels);np.save(folder/f'selected-{block.id}.npy',result[0])
        records.append({'id':block.id,'write':[[s.start,s.stop] for s in block.slice_write(axes)],
            'kept':[int(i) for i in np.unique(result[0]) if i],
            'rejected':sorted((set(map(int,np.unique(labels)))-{0})-(set(map(int,np.unique(result[0])))-{0}))})
        return result
    def wr(block,x,labels,axes=None):
        np.save(folder/f'write-{block.id}.npy',labels);return write(block,x,labels,axes)
    with patch.object(BlockND,'read',rd),patch.object(BlockND,'filter_objects',fl),patch.object(BlockND,'write',wr),patch.object(model,'predict_instances',pr):
        out,_=model.predict_instances_big(image,axes='YX',block_size=block_size,min_overlap=64,context=96,show_progress=False)
    np.save(folder/'expected.npy',out)
    seal(folder,{'kind':'ownership','shape':list(image.shape),'records':records,'parameters':{'block':block_size,'overlap':64,'context':96}})
    return out


def capture_cp(image,folder,block_size):
    check_source('cellpose')
    from cellpose.contrib import distributed_segmentation as cp
    import zarr,distributed
    from unittest.mock import patch
    from contextlib import chdir
    folder=Path(folder).absolute();folder.mkdir(parents=True)
    # Native temporary artifacts are captured explicitly; no tool-prepared inputs.
    z=zarr.open(str(folder/'input.zarr'),mode='w',shape=image.shape,chunks=block_size,dtype=image.dtype);z[:]=image
    graph_fn,merge_fn=cp.block_face_adjacency_graph,cp.determine_merge_relabeling;record={}
    def predictor(input_zarr,crop,*args,**kwargs):
        local=image[crop].copy();np.save(folder/('raw_'+'_'.join(str(s.start)+'-'+str(s.stop) for s in crop)+'.npy'),local);return local
    def graph(faces,nlabels):
        result=graph_fn(faces,nlabels);c=result.tocoo();record['edges']=list(map(list,zip(c.row.tolist(),c.col.tolist())));return result
    def merge(*args,**kwargs):
        temp=next(folder.glob('*_distributed_cellpose_tempdir/segmentation_unstitched.zarr'));np.save(folder/'packed.npy',np.asarray(zarr.open(str(temp),mode='r')))
        result=merge_fn(*args,**kwargs);np.save(folder/'mapping.npy',result);return result
    with chdir(folder),patch.object(cp,'read_preprocess_and_segment',predictor),patch.object(cp,'block_face_adjacency_graph',graph),patch.object(cp,'determine_merge_relabeling',merge):
        with distributed.LocalCluster(n_workers=2,threads_per_worker=1,processes=False,host='127.0.0.1',protocol='tcp',dashboard_address=None,memory_limit=0,silence_logs=40) as cluster:
            with distributed.Client(cluster) as client:
                cluster.client=client;out,_=cp.distributed_eval(z,block_size,str(folder/'output.zarr'),cluster=cluster,eval_kwargs={'diameter':2},temporary_directory=str(folder));out=np.asarray(out)
    np.save(folder/'expected.npy',out);record.update(kind='graph',shape=list(image.shape));seal(folder,record);return out


def replay(folder,alternative=False):
    folder=Path(folder);hashes=json.loads((folder/'hashes.json').read_text())
    for name,h in hashes.items():
        assert Path(name).name==name and not (folder/name).is_symlink()
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==h
    record=json.loads((folder/'record.json').read_text());out=np.zeros(record['shape'],np.int64)
    if record['kind']=='ownership':
        offset=1
        for item in record['records']:
            i=item['id']
            for prefix in ['local','selected','write']:assert f'{prefix}-{i}.npy' in hashes
            if alternative:
                source=np.load(folder/f'local-{i}.npy');local=np.zeros_like(source)
                for j in np.unique(source):
                    if j:local[source==j]=offset;offset+=1
            else:local=np.load(folder/f'write-{i}.npy')
            view=out[tuple(slice(*b) for b in item['write'])];view[local>0]=local[local>0]
    else:
        packed=np.load(folder/'packed.npy');nodes=[int(i) for i in np.unique(packed) if i];groups={i:{i} for i in nodes}
        for a,b in [] if alternative else record['edges']:
            assert a in groups and b in groups
            union=groups[a]|groups[b]
            for i in union:groups[i]=union
        for i in nodes:out[packed==i]=min(groups[i])
    return out
