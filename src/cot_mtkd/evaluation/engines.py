"""Generation back ends. vLLM reproduces exp_s1k ``eval/run_eval.py`` exactly;
the HF back end is a slower fallback with the same sampling distribution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)

STOP_STRINGS = ["<|im_end|>", "<|endoftext|>", "</s>"]


@dataclass(frozen=True)
class Sample:
    text: str
    num_tokens: int
    truncated: bool


def _vllm_lora_rank(rank: int) -> int:
    for allowed in (8, 16, 32, 64, 128, 256, 320, 512):
        if rank <= allowed:
            return allowed
    raise ValueError(f"LoRA rank {rank} exceeds what vLLM supports")


class VLLMEngine:
    def __init__(
        self,
        model: dict[str, Any],
        generation: dict[str, Any],
        engine: dict[str, Any],
        adapter_dir: Optional[Path],
        lora_rank: int,
    ) -> None:
        from vllm import LLM

        self.generation = generation
        self.adapter_dir = adapter_dir
        max_tokens = int(generation["max_new_tokens"])
        self.llm = LLM(
            model=model["name_or_path"],
            revision=model.get("revision"),
            tokenizer_revision=model.get("revision"),
            dtype=model.get("dtype", "bfloat16"),
            enable_lora=adapter_dir is not None,
            max_lora_rank=_vllm_lora_rank(lora_rank),
            gpu_memory_utilization=float(engine.get("gpu_memory_utilization", 0.9)),
            max_model_len=int(engine.get("max_model_len") or max_tokens + 2048),
            tensor_parallel_size=int(engine.get("tensor_parallel_size", 1)),
            seed=int(engine.get("seed", 42)),
            enable_prefix_caching=True,
            max_num_seqs=int(engine.get("max_num_seqs", 256)),
            max_num_batched_tokens=int(engine.get("max_num_batched_tokens", 16384)),
        )

    def generate(self, prompts: list[str], seed: int) -> list[list[Sample]]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        generation = self.generation
        params = SamplingParams(
            n=int(generation["num_samples"]),
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            top_k=int(generation.get("top_k", -1)),
            repetition_penalty=float(generation["repetition_penalty"]),
            max_tokens=int(generation["max_new_tokens"]),
            seed=seed,
            stop=STOP_STRINGS,
        )
        lora = (
            LoRARequest("student", 1, str(self.adapter_dir))
            if self.adapter_dir is not None
            else None
        )
        outputs = self.llm.generate(prompts, params, lora_request=lora)
        return [
            [
                Sample(
                    text=completion.text,
                    num_tokens=len(completion.token_ids),
                    truncated=completion.finish_reason == "length",
                )
                for completion in output.outputs
            ]
            for output in outputs
        ]


class HFEngine:
    """Single-process ``transformers`` fallback for machines without vLLM.

    The model's ``generation_config.json`` sets ``top_k=20`` for Qwen2.5; it is
    overridden here so sampling matches vLLM (no top-k truncation).
    """

    def __init__(
        self,
        model: dict[str, Any],
        generation: dict[str, Any],
        adapter_dir: Optional[Path],
        tokenizer: Any,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        from ..models.multi_adapter import torch_dtype

        self.generation = generation
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base = AutoModelForCausalLM.from_pretrained(
            model["name_or_path"],
            revision=model.get("revision"),
            torch_dtype=torch_dtype(model.get("dtype", "bfloat16")),
            attn_implementation=model.get("attn_implementation", "sdpa"),
            low_cpu_mem_usage=True,
        )
        if adapter_dir is not None:
            from peft import PeftModel

            base = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        self.model = base.to(self.device).eval()
        self.stop_ids = [
            token_id
            for token_id in tokenizer.convert_tokens_to_ids(STOP_STRINGS)
            if token_id is not None and token_id != tokenizer.unk_token_id
        ]

    def generate(self, prompts: list[str], seed: int) -> list[list[Sample]]:
        import torch
        from tqdm.auto import tqdm

        from ..utils.seed import derived_seed

        generation = self.generation
        max_new_tokens = int(generation["max_new_tokens"])
        top_k = int(generation.get("top_k", -1))
        results: list[list[Sample]] = []
        for index, prompt in enumerate(tqdm(prompts, desc="HF generate")):
            encoded = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).to(self.device)
            torch.manual_seed(derived_seed(seed, "evaluation", index))
            with torch.inference_mode():
                outputs = self.model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=float(generation["temperature"]),
                    top_p=float(generation["top_p"]),
                    top_k=top_k if top_k > 0 else 0,
                    repetition_penalty=float(generation["repetition_penalty"]),
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=int(generation["num_samples"]),
                    eos_token_id=self.stop_ids,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            samples = []
            for row in outputs[:, encoded["input_ids"].shape[1] :].tolist():
                ended = [position for position, token in enumerate(row) if token in self.stop_ids]
                length = ended[0] if ended else len(row)
                samples.append(
                    Sample(
                        text=self.tokenizer.decode(row[:length], skip_special_tokens=True),
                        num_tokens=length,
                        truncated=not ended and length >= max_new_tokens,
                    )
                )
            results.append(samples)
        return results
