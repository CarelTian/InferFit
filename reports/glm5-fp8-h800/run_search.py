"""Offline reproduction after source metadata/config snapshots have been saved."""
import json,csv,math
from pathlib import Path
from dataclasses import asdict,replace
from inferfit import *
from inferfit.models import glm_profile
root=Path(__file__).resolve().parent
source=json.loads((root/'source.json').read_text())
c=json.loads((root/'config.json').read_text())
model=glm_profile(c,model_id=source['model_id'],revision=source['revision'])
full=glm_profile(c,include_mtp=True)
index=json.loads((root/'checkpoint-metadata.json').read_text())
assert logical_weight_bytes(full,'fp8')==index['metadata']['total_size']
(root/'model.profile.json').write_text(json.dumps(model_profile_to_dict(model),indent=2)+'\n')
h=load_hardware('h800-80gb');w=Workload()
rows=[];runs=[]
variants=[('vllm',RuntimeProfile(),False),('sglang',RuntimeProfile.sglang(),False),
          ('sglang-dp-attention',RuntimeProfile.sglang(sglang_dp_attention=True),False),
          ('vllm-cross-node-tp',RuntimeProfile(),True),('sglang-cross-node-tp',RuntimeProfile.sglang(),True)]
for label,rt,cross in variants:
    result=search(model,hardware=replace(h,allow_cross_node_tp=cross),runtime=rt,workload=w,
                  tp_values=[1,2,4,8,16],dp_values=range(1,17),pp_values=range(1,17),
                  max_gpus=16,limit=100000,weight_quantization='fp8',kv_quantization='bf16')
    runs.append({'name':label,'minimum_gpus':result.minimum_num_gpus,'evaluated':result.evaluated,
                 'counts':result.counts,'search_space':result.search_space})
    for item in result.candidates:
        p=item.parallelism;s=item.summary
        rows.append({'engine_mode':label,'gpus':p.num_gpus,'tp':p.tp,'dp':p.dp,'ep':p.ep,'pp':p.pp,
          'layers':','.join(map(str,s['stage_layers'])),'nodes':s['nodes_required'],
          'min_headroom_gib':s['minimum_headroom_bytes']/GiB,
          'max_kv_bytes_per_token_per_rank':s['kv_bytes_per_token_per_rank_max'],
          'cross_node_tp_allowed':cross})
    # Keep complete per-rank data for the minimum-card candidates.
    minimum=[x.to_dict() for x in result.candidates if x.parallelism.num_gpus==result.minimum_num_gpus]
    (root/f'{label}-minimum-details.json').write_text(json.dumps(minimum,indent=2)+'\n')
    print(label, result.minimum_num_gpus,result.evaluated,result.counts,flush=True)
selected=[]
for engine,rt in [('vllm',RuntimeProfile()),('sglang',RuntimeProfile.sglang())]:
    for tp,dp,ep,pp,gpus in [(8,1,8,1,8),(4,1,4,3,12),(2,1,2,6,12),(4,1,4,4,16),(8,1,8,2,16),(16,1,16,1,16),(8,2,16,1,16)]:
        if engine=='sglang' and dp>1:rt_use=RuntimeProfile.sglang(sglang_dp_attention=True)
        else:rt_use=rt
        p=Parallelism(tp,dp,ep,pp,gpus)
        for optimize in [False,True] if pp>1 else [False]:
            r=(optimize_pipeline if optimize else check)(model,hardware=replace(h,allow_cross_node_tp=tp>8),parallelism=p,runtime=rt_use,workload=w,weight_quantization='fp8')
            selected.append({'engine':engine,'optimize_pp':optimize,'result':r.to_dict()})
summary={'package_version':__import__('inferfit').__version__,'date':'2026-11-05',
 'source':source,'hardware':asdict(h),'workload':asdict(w),'weight_quantization':'fp8_block',
 'kv_quantization':'bf16','mtp':False,'cpu_offload':False,'gpu_execution_performed':False,
 'checkpoint_full_bytes':index['metadata']['total_size'],
 'base_weight_bytes':logical_weight_bytes(model,'fp8'),'kv_logical_bytes_per_token':kv_per_token(model)['total'],
 'bare_weight_card_lower_bound':math.ceil(logical_weight_bytes(model,'fp8')/h.memory_bytes),
 'runtime_profiles':{label:asdict(rt) for label,rt,_ in variants},'search_runs':runs,
 'candidate_count':len(rows),'note':'Conditional on assumed runtime reserves and specified workload. Search includes PP memory optimization, not throughput optimization.'}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(root/'selected-details.json').write_text(json.dumps(selected,indent=2)+'\n')
(root/'candidates.json').write_text(json.dumps(rows,indent=2)+'\n')
with (root/'candidates.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
for row in rows:
    if row['gpus']==12 or (row['gpus']==16 and row['tp']>=4 and row['dp']<=2 and row['ep'] in [row['tp'],row['tp']*row['dp']]):print(row,flush=True)
