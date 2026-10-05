"""Council pass for Phase 2: ``V_k``, ``M(v)``, ``q`` and ``JS`` per reasoning token.

The ``M`` frozen experts are run once per record (``no_grad``). For every
retained reasoning token the consensus ``p_bar`` is formed over the full
vocabulary, Kneedle selects ``V_k`` (Section 3 tool, optional ``K_max``), and
the restricted expert distributions give the Jensen-Shannon disagreement and
the per-category variance mask. Everything is persisted ragged per sample with
content hashes; the student initialization is the council medoid (Medoid
Cloning), saved next to the cache.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm

from ..data.dataset import JsonlRecordDataset
from ..data.prepare import tokenizer_fingerprint
from ..models.chunked_head import decoder_and_lm_head, forward_hidden
from ..models.multi_adapter import (
    create_multi_adapter_model,
    load_adapter_bundle,
    load_adapter_state,
    load_tokenizer,
    require_same_model_source,
)
from ..utils.distributed import all_reduce_tensor, barrier, shard_indices
from ..utils.manifest import file_sha256, fingerprint, read_json, require_file_sha256, write_json
from .council import (
    council_signals,
    kneedle_support,
    restricted_log_softmax,
    support_mass,
    validate_temperature_config,
)
from .medoid import MedoidResult, select_medoid, validate_medoid_config
from .online import _select_adapter
from .student import plan_record
from .teachers import ensure_stage2_teachers, require_teacher_dataset

LOGGER = logging.getLogger(__name__)
CACHE_VERSION = 2
# torch.quantile rejects inputs larger than 2^24 elements.
_TORCH_QUANTILE_LIMIT = 1 << 24
SUPPORT_SEMANTICS = "kneedle-on-sorted-consensus-pbar-full-vocab-or-kmax-no-gold-exclusion-v1"
MASK_SEMANTICS = "complete-REASONING-content-tokens-per-step-v1"
STORAGE = "int32-ragged-support-float32-variance-mask-int32-k-float64-js-q-bool-gold-v1"


def validate_council_config(config: dict[str, Any]) -> None:
    council = config["council"]
    k_max = council.get("k_max")
    if k_max is not None and (isinstance(k_max, bool) or int(k_max) != k_max or int(k_max) < 2):
        raise ValueError("council.k_max must be null (N' = N) or an integer >= 2")
    validate_temperature_config(council)
    for key in ("alpha", "beta"):
        value = float(council[key])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"council.{key} must be finite and nonnegative")
    if not (0.0 < float(council["epsilon_m"]) < 0.5):
        raise ValueError("council.epsilon_m must lie in (0, 0.5)")
    mask_epsilon = float(council.get("mask_epsilon", 1.0e-12))
    if not math.isfinite(mask_epsilon) or mask_epsilon <= 0.0:
        raise ValueError("council.mask_epsilon must be finite and positive")
    validate_medoid_config(config["medoid"])
    if any(key in config for key in ("aggregation", "geometry")):
        raise ValueError("Obsolete Phase-2 configuration sections: aggregation/geometry")
    if int(config["runtime"]["lm_head_chunk_tokens"]) < 1:
        raise ValueError("lm_head_chunk_tokens must be positive")
    if config["runtime"].get("preprocessing_hidden_storage", "cpu") not in ("cpu", "device"):
        raise ValueError("preprocessing_hidden_storage must be cpu or device")


def cache_identity(config: dict[str, Any], prepared: dict, teachers: dict) -> dict:
    """Only quantities that change ``V_k``/``M(v)``/``q``/``JS`` enter the key.

    Temperatures, loss coefficients, optimizer and batch settings are applied
    at training time and never invalidate the council pass.
    """
    source = Path(__file__).resolve().parents[1]
    code_files = [
        "models/chunked_head.py",
        "models/multi_adapter.py",
        "stage2/council.py",
        "stage2/council_cache.py",
        "stage2/medoid.py",
        "stage2/student.py",
        "data/token_spans.py",
        "data/schema.py",
    ]
    council = config["council"]
    return {
        "cache_version": CACHE_VERSION,
        "code_sha256": {name: file_sha256(source / name) for name in code_files},
        "prepared_manifest_fingerprint": fingerprint(prepared),
        "prepared_data_sha256": prepared["data_file_sha256"],
        "teacher_manifest_fingerprint": fingerprint(teachers),
        "teacher_bundle_sha256": teachers["adapter_bundle_sha256"],
        "teacher_checkpoint_checksums": teachers.get("hub_file_sha256", {}),
        "backbone": {
            k: config["model"].get(k) for k in ("name_or_path", "dtype", "attn_implementation")
        },
        "tokenizer_fingerprint": prepared["tokenizer_fingerprint"],
        "k_max": None if council.get("k_max") is None else int(council["k_max"]),
        "mask_epsilon": float(council.get("mask_epsilon", 1.0e-12)),
        "medoid": {
            "epsilon_rel": float(config["medoid"]["epsilon_rel"]),
            "epsilon_abs": float(config["medoid"]["epsilon_abs"]),
            "tau_b": float(config["medoid"].get("tau_b", 0.0)),
        },
        "max_length": int(config["stage2"]["max_length"]),
        "lm_head_chunk_tokens": int(config["runtime"]["lm_head_chunk_tokens"]),
        "support_semantics": SUPPORT_SEMANTICS,
        "mask_semantics": MASK_SEMANTICS,
        "storage": STORAGE,
        "torch_version": str(torch.__version__),
    }


def _linear_quantile(ordered: torch.Tensor, q: float) -> float:
    """Linear interpolation on an ascending 1-D tensor, matching ``torch.quantile``."""
    count = ordered.numel()
    if count == 1:
        return float(ordered[0])
    position = q * (count - 1)
    lower = math.floor(position)
    upper = min(lower + 1, count - 1)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def distribution_summary(values: torch.Tensor) -> dict:
    values = values.detach().double().flatten()
    if not len(values):
        return {"count": 0}
    low, high = float(values.min()), float(values.max())
    histogram_low, histogram_high = (low - 0.5, high + 0.5) if low == high else (low, high)
    if len(values) <= _TORCH_QUANTILE_LIMIT:
        median, p90, p95 = (float(torch.quantile(values, q)) for q in (0.5, 0.90, 0.95))
    else:
        # The council pass can exceed 2^24 reasoning tokens. Sort once; torch.quantile cannot.
        ordered = torch.sort(values).values
        median, p90, p95 = (_linear_quantile(ordered, q) for q in (0.5, 0.90, 0.95))
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "median": median,
        "p90": p90,
        "p95": p95,
        "max": high,
        "min": low,
        "histogram": torch.histc(values, bins=20, min=histogram_low, max=histogram_high)
        .long()
        .tolist(),
        "histogram_range": [histogram_low, histogram_high],
    }


@torch.no_grad()
def compile_record(
    model, names, record, config, device
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict]:
    """One decoder forward per expert, then two bounded head sweeps per chunk.

    Sweep 1 accumulates ``p_bar`` (full vocabulary, temperature 1) and runs
    Kneedle; sweep 2 gathers each expert on ``V_k`` for ``JS``, ``sigma^2`` and
    ``M(v)``. Expert NLLs are collected as diagnostics only.
    """
    if len(names) < 2:
        raise ValueError("Phase 2 requires at least two frozen experts")
    plan = plan_record(record, None, int(config["stage2"]["max_length"]))
    positions = plan.reasoning_positions
    step_offsets = [0]
    for step in plan.step_positions:
        step_offsets.append(step_offsets[-1] + len(step))
    count = len(positions)
    tensors = {
        "token_positions": torch.tensor(positions, dtype=torch.int32),
        "step_offsets": torch.tensor(step_offsets, dtype=torch.int64),
        "support_offsets": torch.zeros(count + 1, dtype=torch.int64),
        "support_ids": torch.empty(0, dtype=torch.int32),
        "support_mask": torch.empty(0, dtype=torch.float32),
        "k": torch.empty(count, dtype=torch.int32),
        "js": torch.empty(count, dtype=torch.float64),
        "q": torch.empty(count, dtype=torch.float64),
        "gold_in_support": torch.empty(count, dtype=torch.bool),
    }
    scores = torch.zeros(len(names), dtype=torch.float64)
    stats: dict[str, Any] = {"teacher_forward_count": 0}
    if not count:
        return tensors, scores, stats
    ids = torch.tensor([plan.input_ids], device=device, dtype=torch.long)
    indices = torch.tensor(positions, device=device) - 1
    store_on_cpu = config["runtime"].get("preprocessing_hidden_storage", "cpu") == "cpu"
    hidden = []
    for name in names:
        _select_adapter(model, name, training=False)
        output = forward_hidden(model, ids, torch.ones_like(ids), use_cache=False)
        selected = output.last_hidden_state[0].index_select(0, indices).detach()
        hidden.append(selected.cpu() if store_on_cpu else selected)
        del selected, output
    stats["teacher_forward_count"] = len(names)
    _, head = decoder_and_lm_head(model)
    parameter = next(head.parameters())
    chunk = int(config["runtime"]["lm_head_chunk_tokens"])
    council = config["council"]
    k_max = council.get("k_max")
    mask_epsilon = float(council.get("mask_epsilon", 1.0e-12))
    targets = torch.tensor([record.input_ids[p] for p in positions], device=parameter.device)
    step_lengths = torch.tensor(
        [len(step) for step in plan.step_positions for _ in step], device=parameter.device
    ).double()
    all_ids, all_masks = [], []
    for start in range(0, count, chunk):
        end = min(count, start + chunk)
        mean_probabilities = None
        for expert, h in enumerate(hidden):
            logits = head(h[start:end].to(parameter.device, parameter.dtype)).float()
            probabilities = F.softmax(logits, dim=-1)
            mean_probabilities = (
                probabilities if mean_probabilities is None else mean_probabilities + probabilities
            )
            nll = torch.logsumexp(logits, -1) - logits.gather(-1, targets[start:end, None]).squeeze(
                -1
            )
            scores[expert] += float(
                (nll.double() / (step_lengths[start:end] * plan.num_steps)).sum()
            )
            del logits, probabilities
        mean_probabilities = mean_probabilities / len(hidden)
        support_ids, support_mask, k = kneedle_support(mean_probabilities, k_max)
        q = support_mass(mean_probabilities, support_ids, support_mask)
        del mean_probabilities
        expert_log_pi = []
        for h in hidden:
            logits = head(h[start:end].to(parameter.device, parameter.dtype)).float()
            expert_log_pi.append(restricted_log_softmax(logits, support_ids, support_mask))
            del logits
        signals = council_signals(torch.stack(expert_log_pi), support_mask, mask_epsilon)
        del expert_log_pi
        gold_in = ((support_ids == targets[start:end, None]) & support_mask).any(-1)
        tensors["k"][start:end] = k.cpu().int()
        tensors["js"][start:end] = signals.js.cpu()
        tensors["q"][start:end] = q.cpu()
        tensors["gold_in_support"][start:end] = gold_in.cpu()
        valid = support_mask.cpu()
        all_ids.append(support_ids.cpu()[valid].int())
        all_masks.append(signals.variance_mask.cpu()[valid].float())
        tensors["support_offsets"][start + 1 : end + 1] = tensors["support_offsets"][
            start
        ] + k.cpu().long().cumsum(0)
    tensors["support_ids"] = torch.cat(all_ids)
    tensors["support_mask"] = torch.cat(all_masks)
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
        require_file_sha256(
            self.directory, self.manifest, "student_init_file", "student_init_file_sha256"
        )
        medoid = self.manifest["medoid"]
        sums = torch.tensor(medoid["sums"], dtype=torch.float64)
        best = int(torch.argmin(sums))
        if (
            medoid["medoid_index"] != best
            or self.manifest["selected_expert_index"] != best
            or self.manifest["selected_expert"] != self.manifest["adapter_names"][best]
        ):
            raise RuntimeError("Council cache medoid selection mismatch")
        self.verified: set[str] = set()

    @property
    def js_median(self) -> float:
        return float(self.manifest["diagnostics"]["js"]["median"])

    @property
    def js_p95(self) -> float:
        return float(self.manifest["diagnostics"]["js"]["p95"])

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


def _medoid_from_bundle(bundle_path: Path, names: list[str], config: dict) -> MedoidResult:
    bundle = load_adapter_bundle(bundle_path)
    missing = [name for name in names if name not in bundle]
    if missing:
        raise RuntimeError(f"Adapter bundle is missing experts {missing}")
    return select_medoid({name: bundle[name] for name in names}, config["medoid"])


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
            raise ValueError(f"Medoid cloning requires matching LoRA {key}")
    identity = cache_identity(config, prepared, teachers)
    cache_key = fingerprint(identity)
    root = Path(config["paths"]["teacher_cache_dir"]) / cache_key
    if (root / "manifest.json").exists():
        cache = CouncilCache(root, cache_key)
        cache.verify_all()
        LOGGER.info("Council cache hit fingerprint=%s; no teacher model loaded", cache_key)
        return cache.manifest
    LOGGER.info("Council cache miss fingerprint=%s; running the frozen council", cache_key)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(config["model"])
    token_fingerprint = tokenizer_fingerprint(tokenizer)
    if (
        token_fingerprint != prepared["tokenizer_fingerprint"]
        or teachers.get("tokenizer_fingerprint", token_fingerprint) != token_fingerprint
    ):
        raise RuntimeError("Council tokenizer mismatch with prepared data/teachers")
    names = list(teachers["adapter_names"])
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Phase 2 requires at least two distinct experts")
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
    sums = torch.zeros(len(names), device=distributed.device, dtype=torch.float64)
    counts = torch.zeros(4, device=distributed.device, dtype=torch.int64)
    index = {}
    rank_stats = {}
    for i in tqdm(
        shard_indices(len(dataset), distributed.rank, distributed.world_size),
        desc=f"Council pass rank {distributed.rank}",
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
        tokens = len(value["token_positions"])
        steps = len(value["step_offsets"]) - 1
        counts += counts.new_tensor([1, tokens, steps, int(tokens == 0)])
        for key, number in stats.items():
            rank_stats[key] = rank_stats.get(key, 0) + number
    write_json(root / f"index-rank{distributed.rank}.json", index)
    write_json(root / f"stats-rank{distributed.rank}.json", rank_stats)
    all_reduce_tensor(sums)
    all_reduce_tensor(counts)
    barrier()
    if distributed.is_main:
        combined, statistics = {}, {}
        for rank in range(distributed.world_size):
            rank_index = read_json(root / f"index-rank{rank}.json")
            if set(combined) & set(rank_index):
                raise RuntimeError("Duplicate council samples across ranks")
            combined.update(rank_index)
            for key, number in read_json(root / f"stats-rank{rank}.json").items():
                statistics[key] = statistics.get(key, 0) + number
        if len(combined) != len(dataset) or not int(counts[1]):
            raise RuntimeError("Council cache incomplete or no reasoning tokens")
        write_json(root / "index.json", combined)
        summaries: dict[str, list[torch.Tensor]] = {
            key: [] for key in ("k", "js", "q", "tail_mass", "variance_mask", "gold_in_support")
        }
        for entry in combined.values():
            v = load_file(str(root / entry["file"]))
            summaries["k"].append(v["k"])
            summaries["js"].append(v["js"])
            summaries["q"].append(v["q"])
            summaries["tail_mass"].append(1.0 - v["q"])
            summaries["variance_mask"].append(v["support_mask"])
            summaries["gold_in_support"].append(v["gold_in_support"])
        diagnostics = {
            key: distribution_summary(torch.cat(values)) for key, values in summaries.items()
        }
        all_k = torch.cat(summaries["k"]).long()
        diagnostics["k1_fraction"] = float((all_k == 1).double().mean())
        diagnostics["gold_missing_fraction"] = 1.0 - float(
            torch.cat(summaries["gold_in_support"]).double().mean()
        )
        medoid = _medoid_from_bundle(bundle_path, names, config)
        init_path = root / "student_init.pt"
        init_temporary = init_path.with_suffix(".pt.tmp")
        torch.save(load_adapter_bundle(bundle_path)[medoid.medoid_name], init_temporary)
        init_temporary.replace(init_path)
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
            "expert_sft_scores": (sums / len(dataset)).cpu().tolist(),
            "medoid": medoid.to_json(),
            "selected_expert_index": medoid.medoid_index,
            "selected_expert": medoid.medoid_name,
            "initialization": "medoid_cloning_projection_distance",
            "tie_breaking": "first minimum D_m in adapter_names order",
            "index_file": "index.json",
            "index_file_sha256": file_sha256(root / "index.json"),
            "student_init_file": init_path.name,
            "student_init_file_sha256": file_sha256(init_path),
            "prepared_manifest_fingerprint": fingerprint(prepared),
            "stage1_manifest_fingerprint": fingerprint(teachers),
            "stage1_prepared_manifest_fingerprint": teachers["prepared_manifest_fingerprint"],
            "stage1_training_dataset_identity_verified": teachers["prepared_manifest_fingerprint"]
            == fingerprint(prepared),
            "diagnostics": diagnostics,
            "statistics": statistics,
            "preprocessing_wall_seconds": time.perf_counter() - started,
            "disk_bytes": disk_bytes,
            "initialization_disk_bytes": init_path.stat().st_size,
            "cache_directory": str(root.resolve()),
            "cache_status_at_creation": "miss",
        }
        write_json(root / "manifest.json", manifest)
        LOGGER.info(
            "Council cache built: medoid=%s D_m=%s k_mean=%.2f k1=%.3f js_median=%.4f wall=%.1fs",
            medoid.medoid_name,
            [round(x, 4) for x in medoid.sums.tolist()],
            diagnostics["k"]["mean"],
            diagnostics["k1_fraction"],
            diagnostics["js"]["median"],
            manifest["preprocessing_wall_seconds"],
        )
    barrier()
    return read_json(root / "manifest.json")
