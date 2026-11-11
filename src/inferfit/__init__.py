"""InferFit: LLM memory preflight with explicit runtime-memory assumptions."""
from .flagships import flagship_catalog, load_model
from .states.gdn import recurrent_state_bytes
from .profiles import load_model_profile, model_profile_to_dict, load_hardware_profile, load_runtime_profile
from .hardware import load_hardware, hardware_catalog
from .analysis import analyze, search_plans
from .analysis_types import (AnalysisResult, CapacityAssessment, CompatibilityAssessment,
                             MemoryAssumptions, PlanSearchResult)
from .backends import backend_catalog
from .pd import PDPool, PDDemand, PDResult, check_pd
from .engines import SglangAdapter, VllmAdapter
from .estimator import check, kv_per_token, logical_weight_bytes
from .models import from_config, glm_profile, gqa_profile, load_glm51, load_qwen3_235b, UnsupportedModelError, register_model_adapter
from .search import search
from .partition import optimize_pipeline
from .types import (CheckResult, GiB, Hardware, ModelProfile, Parallelism, RankMemory,
                    RuntimeProfile, SearchResult, TensorSpec, Workload)

from ._version import __version__
__all__ = [
    "__version__",
    'flagship_catalog',
    'load_model',
    'recurrent_state_bytes',
    'load_model_profile',
    'model_profile_to_dict',
    'load_hardware_profile',
    'load_runtime_profile',
    'register_model_adapter',
    'load_hardware',
    'load_qwen3_235b',
    'UnsupportedModelError',
    'PDPool',
    'PDDemand',
    'PDResult',
    'check_pd',
    'SglangAdapter',
    'VllmAdapter',
    'check',
    'search',
    'optimize_pipeline',
    'kv_per_token',
    'logical_weight_bytes',
    'from_config',
    'glm_profile',
    'gqa_profile',
    'load_glm51',
    'CheckResult',
    'GiB',
    'Hardware',
    'ModelProfile',
    'Parallelism',
    'RankMemory',
    'RuntimeProfile',
    'SearchResult',
    'TensorSpec',
    'Workload',
    'analyze',
    'search_plans',
    'MemoryAssumptions',
    'AnalysisResult',
    'CapacityAssessment',
    'CompatibilityAssessment',
    'PlanSearchResult',
    'hardware_catalog',
    'backend_catalog',
]
