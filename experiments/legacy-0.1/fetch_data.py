"""Official BBBC039 only; bounded download and selective safe extraction."""
from pathlib import Path
import urllib.request,zipfile,json,hashlib
R=Path(__file__).resolve().parents[1];D=R/'data';D.mkdir(exist_ok=True);records=[];total=905397
for name in ['metadata','images','masks']:
 p=D/(name+'.zip');url='https://data.broadinstitute.org/bbbc/BBBC039/'+name+'.zip'
 if not p.exists():
  tmp=p.with_suffix('.part')
  with urllib.request.urlopen(url,timeout=60) as response,tmp.open('wb') as out:
   while chunk:=response.read(1024*1024):
    total+=len(chunk)
    if total>150*1024**2:raise RuntimeError('RAW_DOWNLOAD_CAP')
    out.write(chunk)
  tmp.rename(p)
 else:total+=p.stat().st_size
 records.append({'url':url,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 (D/'downloads.json').write_text(json.dumps(records,indent=2))
 if name=='metadata':
  with zipfile.ZipFile(p) as z:
   entries={n:z.read(n).decode() for n in z.namelist() if n.startswith('metadata/') and n.endswith('.txt')};(D/'metadata.json').write_text(json.dumps(entries,indent=2));print(entries,flush=True)
with zipfile.ZipFile(D/'images.zip') as a,zipfile.ZipFile(D/'masks.zip') as b:
 if sum(i.file_size for z in [a,b] for i in z.infolist())>1024**3:raise RuntimeError('DECODE_CAP')
print('Downloads verified',total,flush=True)
