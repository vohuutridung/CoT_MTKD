from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm

from ..data.dataset import JsonlRecordDataset
from ..data.prepare import tokenizer_fingerprint
from ..models.chunked_head import decoder_and_lm_head, forward_hidden, full_vocab_probe
from ..models.multi_adapter import (
    create_multi_adapter_model,
    load_adapter_bundle,
    load_adapter_state,
    load_tokenizer,
    require_same_model_source,
)
from ..utils.distributed import all_reduce_tensor, barrier, shard_indices
from ..utils.manifest import file_sha256, fingerprint, read_json, require_file_sha256, write_json
from .online import _select_adapter
from .output_space import plan_record
from .output_space_losses import (
    local_js_log_distribution,
    normalized_js_disagreement,
    power_mean_log_target,
    reduced_log_distribution,
    support_from_probes,
)
from .teachers import ensure_stage2_teachers, require_teacher_dataset

LOGGER = logging.getLogger(__name__)
CACHE_VERSION = 1
SUPPORT_SEMANTICS = "phase1-local-nontarget-kneedle-union-plus-gold-ascending-unique-v1"
MASK_SEMANTICS = "complete-retained-REASONING-content-only-token-step-sample-v1"


def validate_council_config(config: dict[str, Any]) -> None:
    a = config["aggregation"]
    for key in ("js_temperature", "kd_temperature"):
        if not math.isfinite(float(a[key])) or float(a[key]) <= 0:
            raise ValueError(f"aggregation.{key} must be finite and positive")
    if not math.isfinite(float(a["sft_weight"])) or float(a["sft_weight"]) < 0:
        raise ValueError("aggregation.sft_weight must be finite and nonnegative")
    if a["search_k"] != 512 or a["k_min"] != 8:
        raise ValueError("Phase 2 reuses Phase-1 local Kneedle: search_k=512, k_min=8, no k_max")
    if a.get("teacher_execution") != "precomputed_support_tail":
        raise ValueError("Use precomputed_support_tail teacher execution")
    if any(key in a for key in ("temperature", "medoid_temperature", "k_max")):
        raise ValueError("Obsolete aggregation configuration")
    if any(
        k in config["runtime"] for k in ("teacher_hidden_storage", "teacher_probability_cache_gib")
    ):
        raise ValueError("Obsolete online teacher runtime configuration")
    if int(config["runtime"]["lm_head_chunk_tokens"]) < 1:
        raise ValueError("lm_head_chunk_tokens must be positive")
    if config["runtime"].get("preprocessing_hidden_storage", "cpu") not in ("cpu", "device"):
        raise ValueError("preprocessing_hidden_storage must be cpu or device")


def cache_identity(config: dict[str, Any], prepared: dict, teachers: dict) -> dict:
    """Epoch, optimizer, batch size, sft_weight and resume never affect static q."""
    source = Path(__file__).resolve().parents[1]
    code_files = [
        "stage1/kneedle.py",
        "models/chunked_head.py",
        "models/multi_adapter.py",
        "stage2/council_cache.py",
        "stage2/output_space_losses.py",
        "stage2/output_space.py",
        "data/token_spans.py",
        "data/schema.py",
    ]
    return {
        "cache_version": CACHE_VERSION,
        "code_sha256": {name: file_sha256(source / name) for name in code_files},
        "prepared_manifest_fingerprint": fingerprint(prepared),
        "prepared_data_sha256": prepared["data_file_sha256"],
        "teacher_manifest_fingerprint": fingerprint(teachers),
        "teacher_bundle_sha256": teachers["adapter_bundle_sha256"],
        "teacher_checkpoint_checksums": teachers.get("hub_file_sha256", {}),
        "backbone": {
            k: config["model"].get(k)
            for k in ("name_or_path", "dtype", "attn_implementation")
        },
        "tokenizer_fingerprint": prepared["tokenizer_fingerprint"],
        "search_k": int(config["aggregation"]["search_k"]),
        "k_min": int(config["aggregation"]["k_min"]),
        "js_temperature": float(config["aggregation"]["js_temperature"]),
        "kd_temperature": float(config["aggregation"]["kd_temperature"]),
        "max_length": int(config["stage2"]["max_length"]),
        "lm_head_chunk_tokens": int(config["runtime"]["lm_head_chunk_tokens"]),
        "support_semantics": SUPPORT_SEMANTICS,
        "mask_semantics": MASK_SEMANTICS,
        "storage": "int32-ragged-support-float32-log-target-float64-js-rho-v1",
        "torch_version": str(torch.__version__),
    }


def select_best_expert(scores: torch.Tensor) -> int:
    if scores.ndim != 1 or not len(scores) or not bool(torch.isfinite(scores).all()):
        raise ValueError("Expert SFT scores must be a nonempty finite vector")
    # torch.argmin selects the first equal minimum: manifest adapter order.
    return int(scores.argmin())


def distribution_summary(values: torch.Tensor) -> dict:
    values = values.double().flatten()
    if not len(values):
        return {"count": 0}
    low, high = float(values.min()), float(values.max())
    histogram_low, histogram_high = (low - 0.5, high + 0.5) if low == high else (low, high)
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "median": float(torch.quantile(values, 0.5)),
        "p90": float(torch.quantile(values, 0.90)),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
        "min": float(values.min()),
        "histogram": torch.histc(values, bins=20, min=histogram_low, max=histogram_high)
        .long()
        .tolist(),
        "histogram_range": [histogram_low, histogram_high],
    }


@torch.no_grad()
def compile_record(
    model, names, record, config, device
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict]:
    """One decoder forward/expert/sample; SFT scoring shares the q pass.

    The first bounded head sweep reuses full_vocab_probe (gold exclusion) and
    Phase-1 Kneedle. A second head sweep computes logZ and reduced categories.
    Reduced teacher values survive only within one step until its rho is known.
    No full teacher probability vector is constructed or persisted.
    """
    if len(names) != 3:
        raise ValueError("Phase 2 requires exactly three frozen experts")
    plan = plan_record(record, None, int(config["stage2"]["max_length"]))
    positions = [p for step in plan.step_positions for p in step]
    step_offsets = [0]
    for step in plan.step_positions:
        step_offsets.append(step_offsets[-1] + len(step))
    tensors = {
        "token_positions": torch.tensor(positions, dtype=torch.int32),
        "step_offsets": torch.tensor(step_offsets, dtype=torch.int64),
        "support_offsets": torch.zeros(len(positions) + 1, dtype=torch.int64),
        "support_ids": torch.empty(0, dtype=torch.int32),
        "support_log_target": torch.empty(0, dtype=torch.float32),
        "tail_log_target": torch.empty(len(positions), dtype=torch.float32),
        "step_js": torch.empty(plan.num_steps, dtype=torch.float64),
        "step_rho": torch.empty(plan.num_steps, dtype=torch.float64),
        "raw_k": torch.empty((3, len(positions)), dtype=torch.int16),
        "selected_k": torch.empty((3, len(positions)), dtype=torch.int16),
        "union_sizes": torch.empty(len(positions), dtype=torch.int16),
        "gold_present_before_add": torch.empty(len(positions), dtype=torch.bool),
    }
    scores = torch.zeros(3, dtype=torch.float64)
    stats: dict[str, Any] = {"discarded_steps": plan.discarded_steps, "teacher_forward_count": 0}
    if not positions:
        return tensors, scores, stats
    ids = torch.tensor([plan.input_ids], device=device, dtype=torch.long)
    indices = torch.tensor(positions, device=device) - 1
    hidden = []
    for name in names:
        _select_adapter(model, name, training=False)
        output = forward_hidden(model, ids, torch.ones_like(ids), use_cache=False)
        selected = output.last_hidden_state[0].index_select(0, indices).detach()
        hidden.append(
            selected.cpu()
            if config["runtime"].get("preprocessing_hidden_storage", "cpu") == "cpu"
            else selected
        )
        del selected, output
    stats["teacher_forward_count"] = len(names)
    _, head = decoder_and_lm_head(model)
    parameter = next(head.parameters())
    chunk = int(config["runtime"]["lm_head_chunk_tokens"])
    a = config["aggregation"]
    all_ids, all_logq = [], []
    cursor = 0
    for step_index, step in enumerate(plan.step_positions):
        targets = torch.tensor([record.input_ids[p] for p in step], device=parameter.device)
        probes = [
            full_vocab_probe(
                h[cursor : cursor + len(step)],
                head,
                targets,
                int(a["search_k"]),
                chunk,
                output_device=parameter.device,
            )
            for h in hidden
        ]
        support, mask, metadata = support_from_probes(
            torch.stack([p[0] for p in probes]),
            torch.stack([p[1] for p in probes]),
            targets,
            int(a["k_min"]),
        )
        del probes
        step_reduced = []
        js_sum = 0.0
        for start in range(0, len(step), chunk):
            end = min(len(step), start + chunk)
            local_ids, local_mask = support[start:end], mask[start:end]
            reduced, js_values = [], []
            for expert, h in enumerate(hidden):
                logits = head(
                    h[cursor + start : cursor + end].to(parameter.device, parameter.dtype)
                ).float()
                js_values.append(
                    local_js_log_distribution(
                        logits, local_ids, local_mask, float(a["js_temperature"])
                    )
                )
                reduced.append(
                    reduced_log_distribution(
                        logits, local_ids, local_mask, float(a["kd_temperature"]), stats
                    ).double()
                )
                nll = torch.logsumexp(logits, -1) - logits.gather(
                    -1, targets[start:end, None]
                ).squeeze(-1)
                scores[expert] += nll.double().sum().cpu() / (len(step) * plan.num_steps)
                del logits
            js_sum += float(normalized_js_disagreement(torch.stack(js_values)).sum()) * math.log(3)
            step_reduced.append(torch.stack(reduced).cpu())
        js = js_sum / len(step)
        rho = min(1.0, max(0.0, js / math.log(3)))
        tensors["step_js"][step_index], tensors["step_rho"][step_index] = js, rho
        # Ragged persistence: concatenate only valid support categories; tail
        # remains one FP32 log probability per supervised token.
        for chunk_index, start in enumerate(range(0, len(step), chunk)):
            end = min(len(step), start + chunk)
            q = power_mean_log_target(step_reduced[chunk_index], rho)
            valid = mask[start:end].cpu()
            all_ids.append(support[start:end].cpu()[valid].int())
            all_logq.append(q[:, :-1][valid].float())
            tensors["tail_log_target"][cursor + start : cursor + end] = q[:, -1].float()
        tensors["raw_k"][:, cursor : cursor + len(step)] = metadata["raw_k"].cpu().short()
        tensors["selected_k"][:, cursor : cursor + len(step)] = metadata["selected_k"].cpu().short()
        tensors["union_sizes"][cursor : cursor + len(step)] = metadata["union_sizes"].cpu().short()
        tensors["gold_present_before_add"][cursor : cursor + len(step)] = metadata[
            "gold_present_before_add"
        ].cpu()
        sizes = mask.sum(-1).cpu()
        tensors["support_offsets"][cursor + 1 : cursor + len(step) + 1] = tensors[
            "support_offsets"
        ][cursor] + sizes.cumsum(0)
        cursor += len(step)
    tensors["support_ids"] = torch.cat(all_ids)
    tensors["support_log_target"] = torch.cat(all_logq)
    return tensors, scores, stats


class CouncilCache:
    """Version/fingerprint/content checked, lazy one-sample safetensors reader."""

    def __init__(self, directory: str | Path, expected_fingerprint: str):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / "manifest.json")
        if (
            self.manifest.get("artifact") != "stage2_council_cache"
            or self.manifest.get("cache_version") != CACHE_VERSION
        ):
            raise RuntimeError("Council cache version mismatch")
        if (
            self.manifest.get("fingerprint") != expected_fingerprint
            or fingerprint(self.manifest["identity"]) != expected_fingerprint
        ):
            raise RuntimeError("Council cache fingerprint mismatch")
        self.index = read_json(
            require_file_sha256(self.directory, self.manifest, "index_file", "index_file_sha256")
        )
        if len(self.index) != self.manifest["records"]:
            raise RuntimeError("Council cache record count mismatch")
        if "best_expert_file" in self.manifest:
            require_file_sha256(
                self.directory, self.manifest, "best_expert_file", "best_expert_file_sha256"
            )
            best = select_best_expert(
                torch.tensor(self.manifest["expert_sft_scores"], dtype=torch.float64)
            )
            if (
                self.manifest["selected_expert_index"] != best
                or self.manifest["selected_expert"] != self.manifest["adapter_names"][best]
            ):
                raise RuntimeError("Council cache initialization selection mismatch")
        self.verified: set[str] = set()

    def get(self, sample_id: str) -> dict[str, torch.Tensor]:
        entry = self.index[sample_id]
        path = self.directory / entry["file"]
        if sample_id not in self.verified:
            if file_sha256(path) != entry["sha256"]:
                raise RuntimeError(f"Council cache content checksum mismatch: {sample_id}")
            self.verified.add(sample_id)
        return load_file(str(path), device="cpu")

    def verify_all(self) -> None:
        for sample_id, entry in self.index.items():
            if file_sha256(self.directory / entry["file"]) != entry["sha256"]:
                raise RuntimeError(f"Council cache content checksum mismatch: {sample_id}")
            self.verified.add(sample_id)


def load_council_cache(config: dict, prepared: dict, teachers: dict) -> CouncilCache:
    identity = cache_identity(config, prepared, teachers)
    key = fingerprint(identity)
    root = Path(config["paths"]["teacher_cache_dir"]) / key
    if not (root / "manifest.json").exists():
        raise RuntimeError(f"Council cache miss ({key}); run stage2-cache before training")
    cache = CouncilCache(root, key)
    cache.verify_all()
    LOGGER.info(
        "Council cache hit fingerprint=%s samples=%d disk_bytes=%d",
        key,
        cache.manifest["records"],
        cache.manifest["disk_bytes"],
    )
    return cache


def build_council_cache(config: dict[str, Any], distributed) -> dict:
    validate_council_config(config)
    started = time.perf_counter()
    prepared_dir, stage1_dir = Path(config["paths"]["prepared"]), Path(config["paths"]["stage1"])
    prepared, teachers = read_json(prepared_dir / "manifest.json"), ensure_stage2_teachers(config)
    data_path = require_file_sha256(prepared_dir, prepared, "data_file", "data_file_sha256")
    require_file_sha256(prepared_dir, prepared, "config_file", "config_file_sha256")
    bundle_path = require_file_sha256(
        stage1_dir, teachers, "adapter_bundle", "adapter_bundle_sha256"
    )
    require_file_sha256(stage1_dir, teachers, "config_file", "config_file_sha256")
    require_teacher_dataset(teachers, prepared)
    require_same_model_source(
        config["model"], teachers["config"]["model"], "Council preprocessing/teachers"
    )
    for key in ("rank", "alpha", "target_modules"):
        if config["lora"][key] != teachers["config"]["lora"][key]:
            raise ValueError(f"Best-expert cloning requires matching LoRA {key}")
    identity = cache_identity(config, prepared, teachers)
    cache_key = fingerprint(identity)
    root = Path(config["paths"]["teacher_cache_dir"]) / cache_key
    if (root / "manifest.json").exists():
        cache = CouncilCache(root, cache_key)
        cache.verify_all()
        LOGGER.info("Council cache hit fingerprint=%s; no teacher model loaded", cache_key)
        return cache.manifest
    LOGGER.info("Council cache miss fingerprint=%s; preprocessing three frozen experts", cache_key)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(config["model"])
    token_fingerprint = tokenizer_fingerprint(tokenizer)
    if (
        token_fingerprint != prepared["tokenizer_fingerprint"]
        or teachers.get("tokenizer_fingerprint", token_fingerprint) != token_fingerprint
    ):
        raise RuntimeError("Council tokenizer mismatch with prepared data/teachers")
    names = list(teachers["adapter_names"])
    if len(names) != 3 or len(set(names)) != 3:
        raise ValueError("Phase 2 requires three distinct experts")
    dataset = JsonlRecordDataset(data_path)
    if not len(dataset) or len(dataset) != int(prepared["records"]):
        raise RuntimeError("Prepared dataset record count mismatch")
    model, created_names = create_multi_adapter_model(
        config["model"],
        teachers["config"]["lora"],
        len(names),
        distributed.device,
        int(teachers["config"]["seed"]),
    )
    if int(config["stage2"]["max_length"]) > int(model.config.max_position_embeddings):
        raise ValueError("Configured preprocessing context exceeds the backbone context limit")
    if created_names != names:
        raise RuntimeError("Council adapter names mismatch")
    bundle = load_adapter_bundle(bundle_path)
    for name in names:
        load_adapter_state(model, name, bundle[name])
    del bundle
    sums = torch.zeros(3, device=distributed.device, dtype=torch.float64)
    counts = torch.zeros(4, device=distributed.device, dtype=torch.int64)
    index = {}
    rank_stats = {}
    for i in tqdm(
        shard_indices(len(dataset), distributed.rank, distributed.world_size),
        desc=f"Council preprocessing rank {distributed.rank}",
    ):
        record = dataset[i]
        if record.sample_id in index:
            raise RuntimeError("Duplicate prepared sample ID")
        value, scores, stats = compile_record(model, names, record, config, distributed.device)
        filename = f"sample-{i:07d}.safetensors"
        temporary = root / (filename + ".tmp")
        save_file(
            value,
            str(temporary),
            metadata={"format": f"cot-mtkd-council-v{CACHE_VERSION}", "fingerprint": cache_key},
        )
        temporary.replace(root / filename)
        index[record.sample_id] = {"file": filename, "sha256": file_sha256(root / filename)}
        sums += scores.to(distributed.device)
        counts += counts.new_tensor(
            [
                1,
                len(value["token_positions"]),
                len(value["step_js"]),
                int(len(value["step_js"]) == 0),
            ]
        )
        for key, number in stats.items():
            rank_stats[key] = rank_stats.get(key, 0) + number
    write_json(root / f"index-rank{distributed.rank}.json", index)
    write_json(root / f"stats-rank{distributed.rank}.json", rank_stats)
    all_reduce_tensor(sums)
    all_reduce_tensor(counts)
    barrier()
    if distributed.is_main:
        combined, anomalies = {}, {}
        for rank in range(distributed.world_size):
            rank_index = read_json(root / f"index-rank{rank}.json")
            if set(combined) & set(rank_index):
                raise RuntimeError("Duplicate council samples across ranks")
            combined.update(rank_index)
            for key, number in read_json(root / f"stats-rank{rank}.json").items():
                anomalies[key] = anomalies.get(key, 0) + number
        if len(combined) != len(dataset) or not int(counts[1]):
            raise RuntimeError("Council cache incomplete or no reasoning tokens")
        write_json(root / "index.json", combined)
        summaries: dict[str, list[torch.Tensor]] = {
            key: []
            for key in (
                "support_size",
                "union_size",
                "js",
                "rho",
                "target_tail_mass",
                "raw_k",
                "selected_k",
                "gold_present",
            )
        }
        for entry in combined.values():
            v = load_file(str(root / entry["file"]))
            for key, tensor in {
                "support_size": v["support_offsets"].diff(),
                "union_size": v["union_sizes"],
                "js": v["step_js"],
                "rho": v["step_rho"],
                "target_tail_mass": v["tail_log_target"].exp(),
                "raw_k": v["raw_k"],
                "selected_k": v["selected_k"],
                "gold_present": v["gold_present_before_add"],
            }.items():
                summaries[key].append(tensor)
        diagnostics = {
            key: distribution_summary(
                torch.cat(values, dim=1 if key in ("raw_k", "selected_k") else 0)
            )
            for key, values in summaries.items()
            if key not in ("raw_k", "selected_k")
        }
        for key in ("raw_k", "selected_k"):
            per_expert = torch.cat(summaries[key], 1)
            diagnostics[key + "_histogram_by_expert"] = [
                torch.bincount(row.long(), minlength=513).tolist() for row in per_expert
            ]
        scores = sums / len(dataset)
        best = select_best_expert(scores)
        best_path = root / "best_expert.pt"
        # Persist only the winning adapter, so output-space training loads no council weights.
        best_temporary = best_path.with_suffix(".pt.tmp")
        torch.save(load_adapter_bundle(bundle_path)[names[best]], best_temporary)
        best_temporary.replace(best_path)
        disk_bytes = sum((root / entry["file"]).stat().st_size for entry in combined.values())
        manifest = {
            "artifact": "stage2_council_cache",
            "cache_version": CACHE_VERSION,
            "fingerprint": cache_key,
            "identity": identity,
            "records": len(combined),
            "tokens": int(counts[1]),
            "steps": int(counts[2]),
            "empty_samples": int(counts[3]),
            "adapter_names": names,
            "expert_sft_scores": scores.cpu().tolist(),
            "selected_expert_index": best,
            "selected_expert": names[best],
            "tie_breaking": "first minimum in adapter_names order",
            "initialization_scoring": MASK_SEMANTICS + "-temperature1",
            "index_file": "index.json",
            "index_file_sha256": file_sha256(root / "index.json"),
            "best_expert_file": best_path.name,
            "best_expert_file_sha256": file_sha256(best_path),
            "prepared_manifest_fingerprint": fingerprint(prepared),
            "stage1_manifest_fingerprint": fingerprint(teachers),
            "stage1_prepared_manifest_fingerprint": teachers["prepared_manifest_fingerprint"],
            "stage1_training_dataset_identity_verified": teachers["prepared_manifest_fingerprint"]
            == fingerprint(prepared),
            "diagnostics": diagnostics,
            "numerical_anomalies": anomalies,
            "preprocessing_wall_seconds": time.perf_counter() - started,
            "disk_bytes": disk_bytes,
            "initialization_disk_bytes": best_path.stat().st_size,
            "cache_directory": str(root.resolve()),
            "cache_status_at_creation": "miss",
        }
        write_json(root / "manifest.json", manifest)
        LOGGER.info(
            "Council cache built: expert_sft=%s selected=%s anomalies=%s wall=%.1fs bytes=%d",
            manifest["expert_sft_scores"],
            names[best],
            anomalies,
            manifest["preprocessing_wall_seconds"],
            disk_bytes,
        )
    barrier()
    return read_json(root / "manifest.json")
