"""Tokenizer utilities for token budget management."""
from functools import lru_cache


@lru_cache(maxsize=1)
def get_tokenizer(model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def count_tokens(text: str, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> int:
    tokenizer = get_tokenizer(model_name)
    return len(tokenizer.encode(text))


# Fixed token cost for image patches in VLM
IMAGE_TOKEN_COST = 1024  # approximate patch tokens per image crop
