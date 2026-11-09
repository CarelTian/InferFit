"""Versioned, offline data contracts. Loading data never imports Python code."""
from __future__ import annotations
import copy
import json
import math
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any
from .tensor_rules import ROLES, first_stage, tensor_role
from .types import Hardware, ModelProfile, RuntimeProfile, TensorSpec

FAMILIES = {'gqa','gqa_moe','gdn_moe','gdn_qsa_moe','mla_dsa'}
PLACEMENTS = {'replicated','column','row','vocab','kv_column','expert_column','expert_row'}

def _read(value):
    if isinstance(value,(str,Path)):
        value=json.loads(Path(value).read_text())
    if not isinstance(value,dict):raise ValueError('Profile must be a JSON object')
    return copy.deepcopy(value)

def _keys(d,allowed,required=()):
    if set(d)-set(allowed):raise ValueError(f'Unknown profile fields: {sorted(set(d)-set(allowed))}')
    if set(required)-set(d):raise ValueError(f'Missing profile fields: {sorted(set(required)-set(d))}')

def _pos(v,name,zero=False):
    if type(v) is not int or v<(0 if zero else 1):raise ValueError(f'{name} must be an integer >= {0 if zero else 1}')

def validate_model_profile(model: ModelProfile) -> ModelProfile:
    if not isinstance(model,ModelProfile):raise ValueError("Model converters must return ModelProfile")
    if model.architecture not in FAMILIES:raise ValueError('Unsupported memory family; a new state adapter is required')
    if not isinstance(model.model_id,str) or not model.model_id or not isinstance(model.revision,str) or not model.revision:
        raise ValueError('model_id and revision are required strings')
    c=model.config
    for key in ('num_hidden_layers','num_attention_heads','hidden_size','max_position_embeddings'):_pos(c.get(key),key)
    if model.architecture in {'gqa','gqa_moe','gdn_moe','gdn_qsa_moe'}:
        for key in ('num_key_value_heads','head_dim'):_pos(c.get(key),key)
        if c['num_attention_heads']%c['num_key_value_heads']:raise ValueError('Q heads must divide evenly into KV groups')
        if model.architecture in {'gdn_moe', 'gdn_qsa_moe'}:
            for key in ('linear_num_key_heads', 'linear_num_value_heads', 'linear_key_head_dim',
                        'linear_value_head_dim', 'linear_conv_kernel_dim'):
                _pos(c.get(key), key)
            kinds = c.get('layer_types')
            if not isinstance(kinds, list) or len(kinds) != c['num_hidden_layers'] or set(kinds) != {'linear_attention', 'full_attention'}:
                raise ValueError('Hybrid profiles require explicit full/linear layer_types')
            if c.get('mamba_ssm_dtype', 'float32') != 'float32':
                raise ValueError('Only FP32 hybrid recurrent state is supported')
            if model.architecture == 'gdn_qsa_moe':
                for key in ('hc_count', 'indexer_kv_heads', 'indexer_head_dim', 'indexer_compress_ratio',
                            'indexer_budget', 'ngram_size', 'ple_conv_kernel_size'):
                    _pos(c.get(key), key)
                if c['indexer_kv_heads'] * c['indexer_head_dim'] * 2 % c['indexer_compress_ratio']:
                    raise ValueError('QSA compressed bytes must divide evenly per token')
                if not isinstance(c.get('ple_layer_ids'), list) or any(type(i) is not int or not 1 <= i <= c['num_hidden_layers'] for i in c['ple_layer_ids']):
                    raise ValueError('PLE layer IDs must be one-based model layer indices')
    else:
        for key in ('kv_lora_rank','qk_rope_head_dim','qk_nope_head_dim','v_head_dim','index_head_dim'):_pos(c.get(key),key)
    expert_count=None
    if model.architecture in {'gqa_moe','gdn_moe','gdn_qsa_moe','mla_dsa'}:
        expert_count=c.get('num_experts' if model.architecture in {'gqa_moe','gdn_moe','gdn_qsa_moe'} else 'n_routed_experts')
        _pos(expert_count,'expert count')
    if not model.tensors:raise ValueError('Tensor inventory cannot be empty')
    seen=set();covered=set()
    for t in model.tensors:
        if not isinstance(t.name,str) or not t.name or t.name in seen:raise ValueError('Tensor names must be nonempty and unique')
        seen.add(t.name)
        if t.role not in ROLES|{"auto"}:raise ValueError("Unknown tensor role")
        if t.pp_owner not in {None,"first","last"}:raise ValueError("Unknown PP owner")
        if t.layer is not None and t.pp_owner is not None:raise ValueError("Layer tensors cannot have an endpoint owner")
        if not t.shape:raise ValueError('Tensor shape is required')
        for dim in t.shape:_pos(dim,'tensor dimension')
        if t.dtype not in {'BF16','F16','F32','I64'}:raise ValueError('Tensor source dtype must be BF16/F16/F32/I64')
        if t.placement not in PLACEMENTS:raise ValueError('Unknown tensor placement')
        _pos(t.multiplicity,'multiplicity')
        if type(t.quantizable) is not bool:raise ValueError('quantizable must be boolean')
        if t.quantizable and len(t.shape)!=2:raise ValueError('Only matrices support fp8_block storage')
        if t.layer is not None:
            _pos(t.layer,'layer',True)
            if t.layer>=c['num_hidden_layers']:raise ValueError('Tensor layer outside base-model bounds')
            covered.add(t.layer)
        if t.placement.startswith('expert_') and t.multiplicity!=expert_count:
            raise ValueError('Expert tensor multiplicity must equal total resident experts')
        if t.placement in {'row','expert_row'} and len(t.shape)!=2:raise ValueError('Row sharding requires a matrix')
    if covered!=set(range(c['num_hidden_layers'])):raise ValueError('Tensor inventory must cover every model layer')
    return model

def model_profile_to_dict(model: ModelProfile) -> dict:
    validate_model_profile(model)
    return {'schema_version':1,'kind':'model','model_id':model.model_id,'revision':model.revision,
            'memory_family':model.architecture,'config':copy.deepcopy(model.config),
            'tensors':[{**asdict(t),'role':tensor_role(t),
                        'pp_owner':('first' if first_stage(t) else 'last') if t.layer is None else None}
                       for t in model.tensors], 'provenance':copy.deepcopy(model.provenance)}

def load_model_profile(value: dict | str | Path) -> ModelProfile:
    d=_read(value)
    _keys(d,{'schema_version','kind','model_id','revision','memory_family','config','tensors','provenance','reference_weight_bytes'},
          {'schema_version','kind','model_id','revision','memory_family','config','tensors'})
    if type(d['schema_version']) is not int or d['schema_version']!=1 or d['kind']!='model':raise ValueError('Expected model profile schema_version=1')
    if not isinstance(d['config'],dict) or not isinstance(d.get('provenance',{}),dict):raise ValueError('config/provenance must be objects')
    if not isinstance(d['tensors'],list):raise ValueError('tensors must be a list')
    tensors=[]
    for row in d['tensors']:
        if not isinstance(row,dict):raise ValueError('Tensor row must be an object')
        _keys(row,{f.name for f in fields(TensorSpec)},{'name','shape','placement'})
        if not isinstance(row['shape'],(tuple,list)):raise ValueError('shape must be an array')
        if row.get('role','auto')=='auto':raise ValueError('External tensor inventories require explicit semantic roles')
        if row.get('layer') is None and row.get('pp_owner') is None:raise ValueError('Global tensors require explicit pp_owner')
        tensors.append(TensorSpec(**{**row,'shape':tuple(row['shape'])}))
    model=validate_model_profile(ModelProfile(d['model_id'],d['revision'],d['config'],tuple(tensors),d['memory_family'],{**d.get('provenance',{}),'input_kind':'external_profile'}))
    if 'reference_weight_bytes' in d:
        from .estimator import logical_weight_bytes
        if not isinstance(d['reference_weight_bytes'],dict):raise ValueError('reference_weight_bytes must be an object')
        for fmt,size in d['reference_weight_bytes'].items():
            _pos(size,'reference weight bytes')
            if logical_weight_bytes(model,fmt)!=size:raise ValueError('Tensor inventory disagrees with reference weight bytes')
    return model

def load_hardware_profile(value: dict | str | Path) -> Hardware:
    d=_read(value)
    _keys(d,{'schema_version','kind','hardware','provenance'},{'schema_version','kind','hardware'})
    if d['schema_version']!=1 or type(d['schema_version']) is not int or d['kind']!='hardware':raise ValueError('Expected hardware profile schema_version=1')
    h=d['hardware']
    if not isinstance(h,dict):raise ValueError('hardware must be an object')
    _keys(h,{f.name for f in fields(Hardware)},{'name','memory_bytes','platform','compute_capability','gpus_per_node'})
    for key in ('memory_bytes','gpus_per_node'):_pos(h[key],key)
    if not isinstance(h['name'],str) or not h['name']:raise ValueError('Hardware name is required')
    if h['platform']=='cuda':_pos(h['compute_capability'],'compute_capability')
    elif h['compute_capability'] is not None:raise ValueError('Non-CUDA hardware must not claim CUDA compute capability')
    if h.get("available_nodes") is not None:_pos(h["available_nodes"],"available_nodes")
    obj=Hardware(**h)
    for key in ('external_used_bytes_per_gpu','safety_margin_bytes'):_pos(getattr(obj,key),key,True)
    if type(obj.utilization) not in (int,float) or not math.isfinite(obj.utilization) or not 0<obj.utilization<=1:raise ValueError('Invalid utilization')
    for key in ('inter_node_network','allow_cross_node_tp'):
        if type(getattr(obj,key)) is not bool:raise ValueError(f'{key} must be boolean')
    return obj

def load_runtime_profile(value: dict | str | Path) -> RuntimeProfile:
    d=_read(value)
    _keys(d,{'schema_version','kind','runtime','provenance'},{'schema_version','kind','runtime'})
    if d['schema_version']!=1 or type(d['schema_version']) is not int or d['kind']!='runtime':raise ValueError('Expected runtime profile schema_version=1')
    r=d['runtime']
    if not isinstance(r,dict):raise ValueError('runtime must be an object')
    _keys(r,{f.name for f in fields(RuntimeProfile)},{'engine','version'})
    # Capability and budget checks remain in check(); JSON never installs adapters.
    return RuntimeProfile.sglang(**r) if r['engine']=='sglang' else RuntimeProfile(**r)
