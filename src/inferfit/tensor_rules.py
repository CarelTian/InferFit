"""Semantic tensor roles; config-derived model builders may infer canonical names."""
ROLES={'generic','embedding','output','attention','dense_ffn','shared_expert','routed_expert','norm','index_norm','index'}

def tensor_role(t):
    if t.role!='auto':return t.role
    n=t.name
    if 'embed_tokens' in n:return 'embedding'
    if n=='lm_head.weight':return 'output'
    if '.shared_experts.' in n:return 'shared_expert'
    if '.experts.' in n:return 'routed_expert'
    if '.indexer.k_norm.' in n:return 'index_norm'
    if '.indexer.' in n:return 'index'
    if '.mlp.' in n:return 'dense_ffn'
    if '.self_attn.' in n:return 'attention'
    return 'norm' if 'norm' in n else 'generic'

def first_stage(t):
    return t.pp_owner=='first' if t.pp_owner is not None else tensor_role(t)=='embedding'
