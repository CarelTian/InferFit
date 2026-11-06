"""SDK usage: default is 2x6; --reproduce-780 replays the historical search space.

Install inferfit 0.5.0 first. No GPU or network is required.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from inferfit import Hardware, RuntimeProfile, Workload, GiB, load_model_profile, search

parser=argparse.ArgumentParser()
parser.add_argument('--reproduce-780',action='store_true')
parser.add_argument('--memory-bytes',type=int,default=80*GiB,
                    help='Per-GPU visible total bytes; default 80GiB is an explicit assumption')
args=parser.parse_args()
model=load_model_profile(Path(__file__).with_name('model.profile.json'))
workload=Workload(context_tokens=8192,concurrency_per_dp=1,max_batch_tokens=2048)
common=dict(weight_quantization='fp8',kv_quantization='bf16',workload=workload,
            tp_values=[1,2,4,8,16],dp_values=range(1,17),pp_values=range(1,17),
            optimize_pp=True,limit=100000)
if args.reproduce_780:
    # Historical assumptions: 80 decimal GB, 8 GPU slots/node, no inventory limit.
    hardware=Hardware(memory_bytes=80_000_000_000,gpus_per_node=8)
    modes=[('vllm',RuntimeProfile(),False),
           ('sglang',RuntimeProfile.sglang(),False),
           ('sglang-dp-attention',RuntimeProfile.sglang(sglang_dp_attention=True),False),
           ('vllm-cross-node-tp',RuntimeProfile(),True),
           ('sglang-cross-node-tp',RuntimeProfile.sglang(),True)]
    total=0
    for name,runtime,cross in modes:
        result=search(model,hardware=replace(hardware,allow_cross_node_tp=cross),
                      runtime=runtime,max_gpus=16,**common)
        total+=result.evaluated
        print(json.dumps({'mode':name,'evaluated':result.evaluated,'counts':result.counts,
                          'minimum_gpus':result.minimum_num_gpus}))
    print('Total evaluations:',total)
else:
    hardware=Hardware(memory_bytes=args.memory_bytes,gpus_per_node=6,available_nodes=2)
    result=search(model,hardware=hardware,runtime=RuntimeProfile.sglang(),
                  gpu_counts=[12],max_gpus=12,**common)
    print('Evaluated:',result.evaluated,'Statuses:',result.counts)
    print('Minimum GPUs within search space:',result.minimum_num_gpus)
    for candidate in result.candidates:
        p=candidate.parallelism;s=candidate.summary
        print(json.dumps({'tp':p.tp,'dp':p.dp,'ep':p.ep,'pp':p.pp,
                          'layers':s['stage_layers'],'nodes':s['nodes_required'],
                          'headroom_gib':round(s['minimum_headroom_bytes']/GiB,3)}))
