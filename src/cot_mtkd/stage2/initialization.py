from __future__ import annotations

import torch

from ..models.multi_adapter import (
    adapter_parameter_map,
    create_student_model,
    load_adapter_state,
)
from ..utils.manifest import require_file_sha256


def create_cached_student(config, distributed, cache):
    """Copy the complete best-expert LoRA; no teacher adapters are instantiated."""
    model = create_student_model(
        config["model"], config["lora"], distributed.device, int(config["seed"])
    )
    path = require_file_sha256(
        cache.directory, cache.manifest, "best_expert_file", "best_expert_file_sha256"
    )
    load_adapter_state(model, "student", torch.load(path, weights_only=True, map_location="cpu"))
    if int(config["stage2"]["max_length"]) > int(model.config.max_position_embeddings):
        raise ValueError("Configured Phase-2 context exceeds the model context limit")
    model.train()
    parameters = list(adapter_parameter_map(model, "student").values())
    return model, list(cache.manifest["adapter_names"]), parameters
