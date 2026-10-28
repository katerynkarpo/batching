"""Generation backends with one interface:

    backend.generate(conversations, max_new_tokens, batch_records=None) -> list[Generation]

`conversations` is a list of chat message lists (one per batched prompt). Decoding is greedy.
Backends:
  hf      - Hugging Face Transformers (needed for attention analysis, RQ2)
  vllm    - vLLM offline engine (fast sweeps, RQ1)
  openai  - any OpenAI-compatible chat endpoint (OpenAI, OpenRouter, a vLLM server)
  dummy   - oracle / copy outputs for pipeline tests, no model
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


@dataclass
class Generation:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    seconds: float | None = None
    extra: dict = field(default_factory=dict)


class Backend:
    name: str = "base"

    def generate(self, conversations: list[list[dict]], max_new_tokens: int,
                 batch_records: list[list[dict]] | None = None) -> list[Generation]:
        raise NotImplementedError


# ----------------------------------------------------------------------------- dummy
class DummyBackend(Backend):
    """mode='oracle' answers with the gold target in the expected format (pipeline must score ~1.0);
    mode='copy' echoes the input (GEC no-op baseline); mode='garbage' returns unparsable text."""
    name = "dummy"

    def __init__(self, mode: str = "oracle", task: str | None = None):
        self.mode, self.task = mode, task

    def generate(self, conversations, max_new_tokens, batch_records=None):
        from .prompts import render_demo_answer

        assert batch_records is not None, "DummyBackend needs batch_records"
        outs = []
        for recs in batch_records:
            if self.mode == "oracle":
                lines = [f"[{i + 1}] {render_demo_answer(self.task, r, False)}" for i, r in enumerate(recs)]
            elif self.mode == "copy":
                lines = [f"[{i + 1}] {r['input']}" for i, r in enumerate(recs)]
            else:
                lines = ["I refuse to follow the format."]
            text = "\n".join(lines)
            outs.append(Generation(text=text, prompt_tokens=0, completion_tokens=len(text.split()), seconds=0.0))
        return outs


# ----------------------------------------------------------------------------- hf
class HFBackend(Backend):
    name = "hf"

    def __init__(self, model: str, dtype: str = "bfloat16", device_map: str = "auto",
                 attn_implementation: str = "sdpa", micro_batch: int = 8,
                 chat_template_kwargs: dict | None = None, max_model_len: int | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=getattr(torch, dtype), device_map=device_map,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self.micro_batch = micro_batch
        self.chat_kwargs = chat_template_kwargs or {}
        self.max_model_len = max_model_len

    def render(self, conversation: list[dict]) -> str:
        return self.tok.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True,
                                            **self.chat_kwargs)

    def generate(self, conversations, max_new_tokens, batch_records=None):
        import torch

        outs: list[Generation] = []
        texts = [self.render(c) for c in conversations]
        for i in range(0, len(texts), self.micro_batch):
            chunk = texts[i:i + self.micro_batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            t0 = time.time()
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            dt = time.time() - t0
            plen = enc["input_ids"].shape[1]
            for j in range(len(chunk)):
                new = gen[j, plen:]
                n_new = int((new != self.tok.pad_token_id).sum())
                text = self.tok.decode(new, skip_special_tokens=True)
                n_prompt = int(enc["attention_mask"][j].sum())
                outs.append(Generation(text=text, prompt_tokens=n_prompt, completion_tokens=n_new,
                                       seconds=dt / len(chunk)))
        return outs


# ----------------------------------------------------------------------------- vllm
class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, model: str, dtype: str = "bfloat16", max_model_len: int = 16384,
                 gpu_memory_utilization: float = 0.90, tensor_parallel_size: int = 1,
                 chat_template_kwargs: dict | None = None, **kwargs):
        from vllm import LLM

        self.model_id = model
        self.llm = LLM(model=model, dtype=dtype, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization,
                       tensor_parallel_size=tensor_parallel_size, **kwargs)
        self.chat_kwargs = chat_template_kwargs or {}

    def generate(self, conversations, max_new_tokens, batch_records=None):
        from vllm import SamplingParams

        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        t0 = time.time()
        res = self.llm.chat(conversations, sp, use_tqdm=False, chat_template_kwargs=self.chat_kwargs)
        dt = (time.time() - t0) / max(len(conversations), 1)
        outs = []
        for r in res:
            o = r.outputs[0]
            outs.append(Generation(text=o.text, prompt_tokens=len(r.prompt_token_ids),
                                   completion_tokens=len(o.token_ids), seconds=dt,
                                   extra={"finish_reason": o.finish_reason}))
        return outs


# ----------------------------------------------------------------------------- openai-compatible
class OpenAIBackend(Backend):
    name = "openai"

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 concurrency: int = 8, extra_body: dict | None = None, **kwargs):
        from openai import OpenAI

        self.model_id = model
        self.client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.concurrency = concurrency
        self.extra_body = extra_body or {}

    def _one(self, conversation, max_new_tokens) -> Generation:
        t0 = time.time()
        for attempt in range(5):
            try:
                r = self.client.chat.completions.create(
                    model=self.model_id, messages=conversation, temperature=0.0,
                    max_tokens=max_new_tokens, extra_body=self.extra_body,
                )
                break
            except Exception as e:  # rate limits / transient errors
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        u = r.usage
        return Generation(text=r.choices[0].message.content or "", prompt_tokens=getattr(u, "prompt_tokens", None),
                          completion_tokens=getattr(u, "completion_tokens", None), seconds=time.time() - t0,
                          extra={"finish_reason": r.choices[0].finish_reason})

    def generate(self, conversations, max_new_tokens, batch_records=None):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(self.concurrency) as ex:
            return list(ex.map(lambda c: self._one(c, max_new_tokens), conversations))


BACKENDS = {"dummy": DummyBackend, "hf": HFBackend, "vllm": VLLMBackend, "openai": OpenAIBackend}


def make_backend(name: str, **kwargs) -> Backend:
    return BACKENDS[name](**kwargs)
