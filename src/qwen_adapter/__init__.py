"""
Qwen Adapter Module

Adapts Qwen2.5-3B-Instruct model to use BPHA attention.
"""

from .model_loader import load_qwen_model, get_model_config
from .bpha_attention import BPHAAttention
from .kv_cache_manager import KVCacheManager
from .replace_attention import replace_attention_with_bpha, BPHAAttentionWrapper, get_attention_info

__all__ = [
    "load_qwen_model",
    "get_model_config",
    "BPHAAttention",
    "KVCacheManager",
    "replace_attention_with_bpha",
    "BPHAAttentionWrapper",
    "get_attention_info",
]