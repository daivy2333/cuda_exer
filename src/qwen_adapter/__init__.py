"""
Qwen Adapter Module

Adapts Qwen2.5-3B-Instruct model to use BPHA attention.
"""

from .model_loader import load_qwen_model, get_model_config

__all__ = ["load_qwen_model", "get_model_config"]