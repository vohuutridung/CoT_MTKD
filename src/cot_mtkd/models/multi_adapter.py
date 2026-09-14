from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from ..utils.seed import deterministic_rng

LOGGER = logging.getLogger(__name__)


def require_same_model_source(
    actual: dict[str, Any], expected: dict[str, Any], context: str
) -> None:
    keys = ("name_or_path", "revision")
    actual_source = {key: actual.get(key) for key in keys}
    expected_source = {key: expected.get(key) for key in keys}
    if actual_source != expected_source:
        raise RuntimeError(
            f"{context} model source mismatch: expected {expected_source}, got {actual_source}"
        )


def torch_dtype(name: str) -> torch.dtype:
    normalized = name.lower().replace("torch.", "")
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[normalized]


def load_tokenizer(model_config: dict[str, Any]):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"],
        revision=model_config.get("revision"),
        use_fast=True,
        trust_remote_code=False,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer with offset mappings is required")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_causal_lm(model_config: dict[str, Any], device: torch.device):
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "revision": model_config.get("revision"),
        "torch_dtype": torch_dtype(model_config.get("dtype", "bfloat16")),
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    attention = model_config.get("attn_implementation")
    if attention:
        kwargs["attn_implementation"] = attention
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_config["name_or_path"], **kwargs
        )
    except (ImportError, ValueError) as error:
        if attention != "flash_attention_2":
            raise
        LOGGER.warning("FlashAttention-2 unavailable (%s); falling back to SDPA", error)
        kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            model_config["name_or_path"], **kwargs
        )
    model.to(device)
    model.config.use_cache = bool(model_config.get("use_cache", False))
    if model_config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def lora_config(lora: dict[str, Any]):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias="none",
        inference_mode=False,
    )


def create_multi_adapter_model(
    model_config: dict[str, Any],
    lora: dict[str, Any],
    num_experts: int,
    device: torch.device,
    base_seed: int = 42,
):
    from peft import get_peft_model

    base = load_base_causal_lm(model_config, device)
    configuration = lora_config(lora)
    names = [f"expert_{index}" for index in range(num_experts)]
    # PEFT initializes A randomly and B at zero. Fork the RNG explicitly so
    # expert m follows the specified seed base_seed + m, independent of model
    # loading or the order in which other random operations were performed.
    with deterministic_rng(base_seed, device):
        model = get_peft_model(
            base,
            configuration,
            adapter_name=names[0],
            autocast_adapter_dtype=False,
        )
    for expert, name in enumerate(names[1:], start=1):
        with deterministic_rng(base_seed + expert, device):
            model.add_adapter(name, configuration)
    model.to(device)
    set_active_adapter(model, names[0])
    return model, names


def create_student_model(
    model_config: dict[str, Any],
    lora: dict[str, Any],
    device: torch.device,
    seed: int = 42,
):
    from peft import get_peft_model

    base = load_base_causal_lm(model_config, device)
    with deterministic_rng(seed, device):
        model = get_peft_model(
            base,
            lora_config(lora),
            adapter_name="student",
            autocast_adapter_dtype=False,
        )
    model.to(device)
    set_active_adapter(model, "student")
    return model


def set_active_adapter(model: torch.nn.Module, adapter_name: str) -> None:
    model.set_adapter(adapter_name)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_" in name and f".{adapter_name}." in name)


def set_all_adapters_trainable(
    model: torch.nn.Module, adapter_names: list[str]
) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            "lora_" in name
            and any(f".{adapter_name}." in name for adapter_name in adapter_names)
        )


def adapter_parameter_map(
    model: torch.nn.Module, adapter_name: str
) -> OrderedDict[str, torch.nn.Parameter]:
    values: list[tuple[str, torch.nn.Parameter]] = []
    marker = f".{adapter_name}."
    for name, parameter in model.named_parameters():
        if "lora_" in name and marker in name:
            canonical = name.replace(marker, ".{adapter}.")
            values.append((canonical, parameter))
    if not values:
        raise RuntimeError(f"No LoRA parameters found for adapter {adapter_name!r}")
    return OrderedDict(sorted(values))


def adapter_parameter_groups(
    model: torch.nn.Module, adapter_names: list[str]
) -> list[OrderedDict[str, torch.nn.Parameter]]:
    groups = [adapter_parameter_map(model, name) for name in adapter_names]
    reference = list(groups[0])
    for group in groups[1:]:
        if list(group) != reference:
            raise RuntimeError(
                "LoRA adapters do not have identical canonical parameter structures"
            )
    return groups


def extract_adapter_state(
    model: torch.nn.Module, adapter_name: str
) -> dict[str, torch.Tensor]:
    marker = f".{adapter_name}."
    return {
        name.replace(marker, ".{adapter}."): parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name and marker in name
    }


def load_adapter_state(
    model: torch.nn.Module, adapter_name: str, state: dict[str, torch.Tensor]
) -> None:
    target = adapter_parameter_map(model, adapter_name)
    missing = set(target) - set(state)
    unexpected = set(state) - set(target)
    if missing or unexpected:
        raise RuntimeError(
            f"Adapter state mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    with torch.no_grad():
        for key, parameter in target.items():
            parameter.copy_(
                state[key].to(device=parameter.device, dtype=parameter.dtype)
            )


def save_adapter_bundle(
    model: torch.nn.Module,
    adapter_names: list[str],
    output_dir: str | Path,
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    state = {name: extract_adapter_state(model, name) for name in adapter_names}
    path = destination / "adapter_states.pt"
    torch.save(state, path)
    adapter_dir = destination / "adapters"
    model.save_pretrained(
        adapter_dir,
        selected_adapters=adapter_names,
        safe_serialization=True,
    )
    return path


def load_adapter_bundle(path: str | Path) -> dict[str, dict[str, torch.Tensor]]:
    return torch.load(Path(path), map_location="cpu", weights_only=True)
