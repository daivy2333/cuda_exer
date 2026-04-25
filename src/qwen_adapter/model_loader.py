"""
Model Loader: Load Qwen2.5-3B-Instruct from local cache
"""

from typing import Tuple
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_qwen_model(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Qwen model and tokenizer from local path.

    Args:
        model_path: Path to model snapshot directory
        device: Device to load model onto
        torch_dtype: Model data type

    Returns:
        model: Qwen2ForCausalLM model
        tokenizer: Qwen tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )

    return model, tokenizer


def get_model_config(model_path: str) -> dict:
    """
    Extract key config parameters for BPHA adaptation.

    Args:
        model_path: Path to model directory

    Returns:
        config dict with hidden_size, num_heads, num_kv_heads, head_dim
    """
    import json
    config_path = os.path.join(model_path, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    return {
        "hidden_size": config["hidden_size"],
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config["num_key_value_heads"],
        "head_dim": config["hidden_size"] // config["num_attention_heads"],
        "num_hidden_layers": config["num_hidden_layers"],
    }