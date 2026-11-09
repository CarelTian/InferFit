"""PD capacity preflight with explicit handoff liveness and disjoint devices.

No network simulator or connector launch guarantee is implied.
"""
from dataclasses import dataclass, asdict, field, replace
from .types import Hardware, Parallelism, RuntimeProfile, Workload, ModelProfile, CheckResult
from .estimator import check, kv_per_token

@dataclass(frozen=True)
class PDPool:
    gpu_ids: tuple[str, ...]
    hardware: Hardware = field(default_factory=Hardware)
    parallelism: Parallelism = field(default_factory=Parallelism)
    runtime: RuntimeProfile = field(default_factory=RuntimeProfile)
    weight_quantization: str = "bf16"
    kv_quantization: str = "bf16"
    max_batch_tokens: int = 2048
    # Explicit extra resident connector/staging bytes on EACH GPU, not host RAM.
    transfer_buffer_bytes_per_gpu: int = 0

@dataclass(frozen=True)
class PDDemand:
    prompt_tokens: int = 8192
    output_tokens: int = 1024
    prefill_concurrency_per_dp: int = 1
    decode_concurrency_per_dp: int = 1
    # Outstanding transfers per DP; P retains source until ACK while D reserves slots.
    inflight_handoffs_per_dp: int = 1

@dataclass
class PDResult:
    status: str
    memory_pass: bool | None
    prefill: CheckResult | None = None
    decode: CheckResult | None = None
    reasons: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    def to_dict(self):
        return asdict(self)

def check_pd(model: ModelProfile, *, prefill: PDPool, decode: PDPool,
             demand: PDDemand | None = None) -> PDResult:
    if model.architecture in {"gdn_moe", "gdn_qsa_moe"} or "shared" in model.config.get("indexer_types", []):
        return PDResult("unsupported", None, reasons=["Hybrid recurrent / shared-index state transfer is not validated; standalone pools only"])
    d = demand or PDDemand()
    for name,value in asdict(d).items():
        if type(value) is not int or value < (0 if name=="inflight_handoffs_per_dp" else 1):
            return PDResult("invalid",None,reasons=[f"Invalid PD demand: {name}"])
    for pool in (prefill,decode):
        if any(not isinstance(x,str) or not x for x in pool.gpu_ids):
            return PDResult("invalid",None,reasons=["GPU IDs must be nonempty strings"])
        if len(pool.gpu_ids)!=pool.parallelism.num_gpus or len(set(pool.gpu_ids))!=len(pool.gpu_ids):
            return PDResult("invalid",None,reasons=["Pool physical GPU IDs must be unique and match num_gpus"])
        if any(not isinstance(x,str) or not x for x in pool.gpu_ids):
            return PDResult("invalid",None,reasons=["GPU IDs must be nonempty strings"])
        if type(pool.transfer_buffer_bytes_per_gpu) is not int or pool.transfer_buffer_bytes_per_gpu<0:
            return PDResult("invalid",None,reasons=["Transfer buffer bytes must be non-negative"])
    if set(prefill.gpu_ids)&set(decode.gpu_ids):
        return PDResult("invalid",None,reasons=["Prefill and decode pools overlap physical GPUs"])
    if (prefill.runtime.engine,prefill.runtime.version)!=(decode.runtime.engine,decode.runtime.version):
        return PDResult("unsupported",None,reasons=["Mixed-engine/version KV transfer adapter is not implemented"])
    if prefill.kv_quantization!=decode.kv_quantization or prefill.weight_quantization!=decode.weight_quantization:
        return PDResult("unsupported",None,reasons=["PD precision conversion is not modeled; both pools must use identical formats"])
    p,dplan=prefill.parallelism,decode.parallelism
    if (p.tp,p.dp,p.ep,p.pp,p.stage_layers)!=(dplan.tp,dplan.dp,dplan.ep,dplan.pp,dplan.stage_layers):
        return PDResult("unsupported",None,reasons=["Only identical P/D layouts supported; KV reshards/routing are not modeled"])
    if prefill.runtime.sglang_dp_attention!=decode.runtime.sglang_dp_attention:
        return PDResult("unsupported",None,reasons=["P/D attention meshes differ"])
    results=[]
    for pool,context,concurrency in (
        (prefill,d.prompt_tokens,d.prefill_concurrency_per_dp+d.inflight_handoffs_per_dp),
        (decode,d.prompt_tokens+d.output_tokens,d.decode_concurrency_per_dp+d.inflight_handoffs_per_dp)):
        runtime=replace(pool.runtime,extra_bytes=pool.runtime.extra_bytes+pool.transfer_buffer_bytes_per_gpu)
        results.append(check(model,hardware=pool.hardware,parallelism=pool.parallelism,
            workload=Workload(context_tokens=context,concurrency_per_dp=concurrency,
                              max_batch_tokens=pool.max_batch_tokens),runtime=runtime,
            weight_quantization=pool.weight_quantization,kv_quantization=pool.kv_quantization))
    pr,dr=results
    status=next((r.status for r in results if r.status in {"invalid","unsupported"}),None)
    if status:
        return PDResult(status,None,pr,dr,[reason for r in results for reason in r.reasons])
    passed=bool(pr.memory_pass and dr.memory_pass)
    payload=kv_per_token(model,tp=1,kv_quantization=prefill.kv_quantization,runtime=prefill.runtime)["total"]*d.prompt_tokens
    return PDResult("pass" if passed else "fail",passed,pr,dr,
        [reason for r in results for reason in r.reasons],{
        "num_gpus_total":p.num_gpus+dplan.num_gpus,
        "minimum_headroom_bytes":min(pr.summary["minimum_headroom_bytes"],dr.summary["minimum_headroom_bytes"]),
        "handoff_logical_payload_bytes_per_request":payload,
        "transfer_bytes_include_tp_replicas":False,
        "payload_scope":"logical state only, excludes wire padding/metadata/replicated transport bytes",
        "source_release":"after_ack", "source_and_destination_overlap_counted":True,
        "outstanding_handoffs_per_dp":d.inflight_handoffs_per_dp,
        "deployment_verified":False,"connector_verified":False,
        "scope":"Analytical PD capacity only; connector, RDMA, latency and bandwidth require separate validation",
        "pool_engine_args_scope":"standalone pool sizing hints, not a PD launch configuration",
    })
