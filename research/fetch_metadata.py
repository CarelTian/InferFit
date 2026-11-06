"""Refresh representative safetensors headers; never fetch tensor payloads."""
import hashlib,json,re,struct,urllib.request,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def norm(name):
 return re.sub(r'\.\d+(?=\.)', '.#', name)

def get(url, start=None, stop=None):
 headers={} if start is None else {'Range':f'bytes={start}-{stop}'}
 for attempt in range(4):
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=30) as f:
    if start is not None and f.status!=206: raise RuntimeError('Server ignored Range; refusing weight download')
    return f.read()
  except Exception:
   if attempt==3: raise
   time.sleep(1)

for mode in ('bf16','fp8'):
 info=json.loads((ROOT/f'{mode}-info.json').read_text())
 idx=json.loads((ROOT/f'{mode}-model.safetensors.index.json').read_text())
 files={}
 for name,file in idx['weight_map'].items(): files.setdefault(file,set()).add(norm(name))
 remaining=set().union(*files.values()); selected=[]
 while remaining:
  chosen=max(files,key=lambda f:len(files[f]&remaining))
  selected.append(chosen);remaining-=files[chosen]
 templates={}; sources=[]
 for file in selected:
  url=f"https://huggingface.co/{info['id']}/resolve/{info['sha']}/{file}"
  cache=ROOT/f'{mode}-{file}.header.json'
  if cache.exists(): raw=cache.read_bytes()
  else:
   size=struct.unpack('<Q',get(url,0,7))[0]
   if size>8*1024*1024: raise RuntimeError('Unexpected header size')
   raw=get(url,8,7+size); cache.write_bytes(raw)
  size=len(raw);head=json.loads(raw)
  for name,spec in head.items():
   if name=='__metadata__':continue
   key=norm(name); value={'shape':spec['shape'],'dtype':spec['dtype']}
   if key in templates and templates[key]!=value:raise RuntimeError(('Conflicting templates',key))
   templates[key]=value
  sources.append({'url':url,'header_sha256':hashlib.sha256(raw).hexdigest(),'header_bytes':size})
  print(mode,file,size,flush=True)
 counts={key:sum(norm(n)==key for n in idx['weight_map']) for key in templates}
 result={'model_id':info['id'],'revision':info['sha'],'checkpoint_total_bytes':idx['metadata']['total_size'],'templates':templates,'counts':counts,'sources':sources}
 (ROOT/f'{mode}-tensor-templates.json').write_text(json.dumps(result,indent=2)+'\n')
for name,path in [('vllm-flashmla-v019.py','vllm/v1/attention/backends/mla/flashmla_sparse.py'),('vllm-kv-utils-v019.py','vllm/v1/core/kv_cache_utils.py'),('vllm-mla-attention-v019.py','vllm/model_executor/layers/attention/mla_attention.py')]:
 url='https://raw.githubusercontent.com/vllm-project/vllm/v0.19.0/'+path
 (ROOT/name).write_bytes(get(url));print(name,flush=True)
