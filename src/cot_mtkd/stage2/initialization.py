from __future__ import annotations

import torch

from ..models.multi_adapter import (
    adapter_parameter_map,
    create_student_model,
    load_adapter_state,
)
from ..utils.manifest import require_file_sha256


def create_cached_student(config, distributed, cache):
    """Create the student LoRA; no teacher adapters are instantiated.

    stage2.student_init: base starts from a fresh LoRA (B = 0, i.e. the base
    model), so Phase 2 is one pass over the data. best_expert copies the
    complete lowest-SFT expert, which continues an already trained adapter.
    """
    model = create_student_model(
        config["model"], config["lora"], distributed.device, int(config["seed"])
    )
    if config["stage2"].get("student_init", "base") == "best_expert":
        path = require_file_sha256(
            cache.directory, cache.manifest, "best_expert_file", "best_expert_file_sha256"
        )
        state = torch.load(path, weights_only=True, map_location="cpu")
        load_adapter_state(model, "student", state)
    if int(config["stage2"]["max_length"]) > int(model.config.max_position_embeddings):
        raise ValueError("Configured Phase-2 context exceeds the model context limit")
    model.train()
    parameters = list(adapter_parameter_map(model, "student").values())
    return model, list(cache.manifest["adapter_names"]), parameters


def initial_adapter_name(config, council_manifest):
    if config["stage2"].get("student_init", "base") == "best_expert":
        return council_manifest["selected_expert"]
    return None
