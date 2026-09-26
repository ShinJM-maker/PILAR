"""Unified LLM/VLM client interfaces for vLLM and HuggingFace."""
import logging
import base64
import asyncio
from typing import List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


def _strip_think_tags(text: str) -> str:
    """Strip Qwen3 thinking tags from response."""
    if not text:
        return ""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if "<think>" in text:
        # Thinking was cut off by max_tokens — no usable output
        return ""
    return text


class VLLMClient:
    """Client for vLLM OpenAI-compatible API server."""

    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 model: str = "Qwen/Qwen3-8B",
                 temperature: float = 0.0,
                 max_tokens: int = 1024,
                 max_concurrent: int = 32):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self._client = None
        self._async_client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key="dummy")
        return self._client

    def _get_async_client(self):
        if self._async_client is None:
            from openai import AsyncOpenAI
            self._async_client = AsyncOpenAI(base_url=self.base_url, api_key="dummy")
        return self._async_client

    def generate(self, prompt: str, image_path: str = None) -> str:
        """Generate a response from the LLM/VLM."""
        client = self._get_client()

        messages = []
        if image_path:
            content = []
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return _strip_think_tags(response.choices[0].message.content)

    async def _async_generate(self, prompt: str, semaphore: asyncio.Semaphore) -> str:
        """Async generate with concurrency limit."""
        async with semaphore:
            client = self._get_async_client()
            messages = [{"role": "user", "content": prompt}]
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return _strip_think_tags(response.choices[0].message.content)
            except Exception as e:
                logger.warning(f"Async generate failed: {e}")
                return ""

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for multiple prompts concurrently."""
        if len(prompts) <= 1:
            return [self.generate(p) for p in prompts]

        async def _run_batch():
            semaphore = asyncio.Semaphore(self.max_concurrent)
            tasks = [self._async_generate(p, semaphore) for p in prompts]
            return await asyncio.gather(*tasks)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in an async context, use thread pool
                with ThreadPoolExecutor(1) as pool:
                    results = pool.submit(lambda: asyncio.run(_run_batch())).result()
                return list(results)
            else:
                return list(loop.run_until_complete(_run_batch()))
        except RuntimeError:
            return list(asyncio.run(_run_batch()))


class HFClient:
    """HuggingFace Transformers client for local inference."""

    def __init__(self, model_name: str, device: str = "cuda:0",
                 temperature: float = 0.0, max_tokens: int = 1024,
                 is_vlm: bool = False):
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.is_vlm = is_vlm
        self._model = None
        self._tokenizer = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.is_vlm:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )

    def generate(self, prompt: str, image_path: str = None) -> str:
        self._load_model()
        import torch

        if self.is_vlm and image_path:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(self._model.device)
        else:
            tokenizer = self._tokenizer or self._processor.tokenizer
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
            )

        tokenizer = self._tokenizer or self._processor.tokenizer
        input_len = inputs["input_ids"].shape[-1]
        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return _strip_think_tags(generated)

    def generate_batch(self, prompts: List[str]) -> List[str]:
        return [self.generate(p) for p in prompts]
