from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..data.dataset import JsonlRecordDataset
from ..data.prepare import tokenizer_fingerprint
from ..data.schema import PreparedRecord
from ..models.chunked_head import decoder_and_lm_head, forward_hidden
from ..models.multi_adapter import (
    create_multi_adapter_model,
    load_adapter_bundle,
    load_adapter_state,
    load_tokenizer,
    require_same_model_source,
)
from ..signals.medoid import functional_medoid
from ..utils.distributed import (
    DistributedContext,
    all_reduce_tensor,
    barrier,
    shard_indices,
)
from ..utils.manifest import (
    file_sha256,
    fingerprint,
    read_json,
    require_file_sha256,
    runtime_metadata,
    write_config_snapshot,
    write_json,
)
from .teachers import ensure_stage2_teachers, require_teacher_dataset

LOGGER = logging.getLogger(__name__)


def full_vocab_medoid_scores(
    hidden_by_teacher: list[torch.Tensor],
    head: torch.nn.Module,
    chunk_tokens: int,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    """Sum KL(uniform council mean || teacher) across response tokens.

    The vocabulary is never truncated. Only the token dimension is chunked,
    allowing teacher hidden states to stay on CPU between head projections.
    Returned sums use FP64 accumulation and are detached on the head device.
    """
    if len(hidden_by_teacher) < 2:
        raise ValueError("Stage-2 medoid requires at least two teachers")
    if chunk_tokens <= 0:
        raise ValueError("lm_head_chunk_tokens must be positive")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Medoid temperature must be finite and positive")
    shape = hidden_by_teacher[0].shape
    if len(shape) != 2 or any(hidden.shape != shape for hidden in hidden_by_teacher):
        raise ValueError("Teacher hidden states must share shape [tokens, hidden_size]")
    parameter = next(head.parameters())
    scores = torch.zeros(len(hidden_by_teacher), device=parameter.device, dtype=torch.float64)
    token_count = int(shape[0])
    with torch.no_grad():
        for start in range(0, token_count, chunk_tokens):
            end = min(start + chunk_tokens, token_count)
            log_probabilities = []
            for hidden in hidden_by_teacher:
                logits = head(hidden[start:end].to(parameter.device, dtype=parameter.dtype)).float()
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Non-finite teacher logits during medoid scoring")
                log_probabilities.append(F.log_softmax(logits / temperature, dim=-1))
            mixture = torch.stack([value.exp() for value in log_probabilities]).mean(dim=0)
            log_mixture = mixture.clamp_min(1.0e-30).log()
            for teacher, log_probability in enumerate(log_probabilities):
                per_token = (mixture * (log_mixture - log_probability)).sum(dim=-1)
                scores[teacher] += per_token.clamp_min(0.0).double().sum()
    return scores, token_count


def score_record_medoid(
    model: torch.nn.Module,
    adapter_names: list[str],
    record: PreparedRecord,
    device: torch.device,
    chunk_tokens: int,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    """Teacher-force every assistant target, including control and answer tokens."""
    input_ids = torch.tensor([record.input_ids], device=device, dtype=torch.long)
    attention_mask = torch.tensor([record.attention_mask], device=device, dtype=torch.long)
    valid = torch.tensor(record.labels[1:], device=device, dtype=torch.long).ne(-100)
    hidden_by_teacher = []
    model.eval()
    with torch.no_grad():
        for name in adapter_names:
            model.set_adapter(name)
            # PEFT adapter selection can re-enable the selected adapter's grads.
            # Medoid preparation never trains any council or backbone parameter.
            model.requires_grad_(False)
            output = forward_hidden(model, input_ids, attention_mask, use_cache=False)
            hidden_by_teacher.append(output.last_hidden_state[0, :-1, :][valid].cpu())
    _, head = decoder_and_lm_head(model)
    return full_vocab_medoid_scores(hidden_by_teacher, head, chunk_tokens, temperature)


def build_stage2_medoid(
    config: dict[str, Any], distributed: DistributedContext
) -> dict[str, Any]:
    """Select one functional medoid over the complete prepared Phase-1 corpus."""
    section = (
        "aggregation"
        if config.get("method") == "disagreement_adaptive_distribution_aggregation_mtkd"
        else "geometry"
    )
    temperature = float(config[section]["temperature"])
    chunk_tokens = int(config["runtime"]["lm_head_chunk_tokens"])
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"{section}.temperature must be finite and positive")
    if chunk_tokens <= 0:
        raise ValueError("runtime.lm_head_chunk_tokens must be positive")
    prepared_dir = Path(config["paths"]["prepared"])
    stage1_dir = Path(config["paths"]["stage1"])
    output_dir = Path(config["paths"]["medoid"])
    prepared_manifest = read_json(prepared_dir / "manifest.json")
    stage1_manifest = ensure_stage2_teachers(config)
    data_path = require_file_sha256(
        prepared_dir, prepared_manifest, "data_file", "data_file_sha256"
    )
    require_file_sha256(prepared_dir, prepared_manifest, "config_file", "config_file_sha256")
    bundle_path = require_file_sha256(
        stage1_dir, stage1_manifest, "adapter_bundle", "adapter_bundle_sha256"
    )
    require_file_sha256(stage1_dir, stage1_manifest, "config_file", "config_file_sha256")
    prepared_fingerprint = fingerprint(prepared_manifest)
    require_teacher_dataset(stage1_manifest, prepared_manifest)
    stage1_config = stage1_manifest["config"]
    require_same_model_source(config["model"], stage1_config["model"], "Medoid/Stage 1")
    tokenizer = load_tokenizer(config["model"])
    token_fingerprint = tokenizer_fingerprint(tokenizer)
    if token_fingerprint != prepared_manifest["tokenizer_fingerprint"]:
        raise RuntimeError("Medoid tokenizer does not match the prepared dataset")
    if stage1_manifest.get("tokenizer_fingerprint", token_fingerprint) != token_fingerprint:
        raise RuntimeError("Medoid tokenizer does not match the Stage-1 checkpoint")
    adapter_names = list(stage1_manifest["adapter_names"])
    if len(adapter_names) < 2 or len(set(adapter_names)) != len(adapter_names):
        raise ValueError("Stage-2 medoid requires at least two distinct teacher adapters")
    dataset = JsonlRecordDataset(data_path)
    if not len(dataset) or len(dataset) != int(prepared_manifest["records"]):
        raise RuntimeError("Prepared dataset record count does not match its manifest")
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    if distributed.is_main:
        write_config_snapshot(config_path, public_config)
    barrier()
    model, created_names = create_multi_adapter_model(
        config["model"],
        stage1_config["lora"],
        len(adapter_names),
        distributed.device,
        int(stage1_config["seed"]),
    )
    if created_names != adapter_names:
        raise RuntimeError("Adapter-name mismatch during medoid preparation")
    bundle = load_adapter_bundle(bundle_path)
    for name in adapter_names:
        load_adapter_state(model, name, bundle[name])
    model.eval()
    model.requires_grad_(False)
    sums = torch.zeros(len(adapter_names), device=distributed.device, dtype=torch.float64)
    counts = torch.zeros(2, device=distributed.device, dtype=torch.int64)
    for index in tqdm(
        shard_indices(len(dataset), distributed.rank, distributed.world_size),
        desc=f"Stage-2 medoid rank {distributed.rank}",
    ):
        scores, token_count = score_record_medoid(
            model, adapter_names, dataset[index], distributed.device, chunk_tokens, temperature
        )
        sums += scores.to(distributed.device)
        counts[0] += token_count
        counts[1] += 1
    all_reduce_tensor(sums)
    all_reduce_tensor(counts)
    if int(counts[1].item()) != len(dataset):
        raise RuntimeError("Medoid scoring did not cover every prepared training example")
    if int(counts[0].item()) == 0:
        raise RuntimeError("No supervised assistant tokens available for medoid selection")
    scores = sums / counts[0]
    medoid_index = functional_medoid(scores)
    if distributed.is_main:
        manifest = {
            "schema_version": 1,
            "artifact": "stage2_medoid",
            "records": int(counts[1].item()),
            "response_token_count": int(counts[0].item()),
            "temperature": temperature,
            "selection": "full_prepared_training_set",
            "score_definition": "mean_full_vocab_KL(uniform_council_mean||teacher)",
            "token_selection": "all_non_prompt_assistant_targets",
            "adapter_names": adapter_names,
            "functional_medoid_index": medoid_index,
            "functional_medoid_adapter": adapter_names[medoid_index],
            "functional_medoid_kl": scores.detach().cpu().tolist(),
            "prepared_manifest_fingerprint": prepared_fingerprint,
            "stage1_manifest_fingerprint": fingerprint(stage1_manifest),
            "stage1_prepared_manifest_fingerprint": stage1_manifest["prepared_manifest_fingerprint"],
            "stage1_training_dataset_identity_verified": (
                stage1_manifest["prepared_manifest_fingerprint"] == prepared_fingerprint
            ),
            "prepared_data_file_sha256": prepared_manifest["data_file_sha256"],
            "prepared_config_file_sha256": prepared_manifest["config_file_sha256"],
            "stage1_adapter_bundle_sha256": stage1_manifest["adapter_bundle_sha256"],
            "stage1_config_file_sha256": stage1_manifest["config_file_sha256"],
            "tokenizer_fingerprint": token_fingerprint,
            "model_source": {
                key: config["model"].get(key) for key in ("name_or_path", "revision")
            },
            "config": public_config,
            "config_fingerprint": fingerprint(public_config),
            "config_file": config_path.name,
            "config_file_sha256": file_sha256(config_path),
            "runtime": runtime_metadata(config["_project_root"]),
        }
        write_json(output_dir / "manifest.json", manifest)
        LOGGER.info("Stage-2 functional medoid: %s", adapter_names[medoid_index])
    barrier()
    return read_json(output_dir / "manifest.json")
