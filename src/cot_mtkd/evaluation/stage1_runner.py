from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

import yaml
from tqdm.auto import tqdm

from ..models.multi_adapter import require_same_model_source
from ..utils.manifest import (
    file_sha256,
    files_fingerprint,
    fingerprint,
    read_json,
    require_file_sha256,
    runtime_metadata,
    write_json,
)
from ..utils.seed import derived_seed
from .grading import grade_math, normalize_answer
from .metrics import macro_average, pass_metrics
from .runner import _answer, _chat_prompt, _load_benchmark, _question

LOGGER = logging.getLogger(__name__)

CHECKPOINTS = ("final", "benchmark")


class ExpertEngine(Protocol):
    tokenizer: Any

    def generate(
        self,
        prompt_token_ids: list[list[int]],
        seeds: list[int],
        adapter_name: str,
        adapter_path: Path,
    ) -> list[list[tuple[str, str | None]]]:
        """Return (text, finish_reason) for every sample of every prompt."""


class VllmExpertEngine:
    def __init__(
        self,
        model_config: dict[str, Any],
        vllm_config: dict[str, Any],
        generation: dict[str, Any],
        seed: int,
    ) -> None:
        try:
            from vllm import LLM
        except ImportError as error:
            raise RuntimeError(
                "Stage-1 expert evaluation needs vLLM: pip install -e '.[vllm]'"
            ) from error

        self.llm = LLM(
            model=model_config["name_or_path"],
            revision=model_config.get("revision"),
            tokenizer_revision=model_config.get("revision"),
            dtype=str(model_config.get("dtype", "bfloat16")),
            max_model_len=int(vllm_config["max_model_len"]),
            tensor_parallel_size=int(vllm_config["tensor_parallel_size"]),
            gpu_memory_utilization=float(vllm_config["gpu_memory_utilization"]),
            enable_lora=True,
            max_lora_rank=int(vllm_config["max_lora_rank"]),
            max_loras=1,
            seed=seed,
            trust_remote_code=False,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.generation = generation
        self._lora_ids: dict[str, int] = {}

    def generate(
        self,
        prompt_token_ids: list[list[int]],
        seeds: list[int],
        adapter_name: str,
        adapter_path: Path,
    ) -> list[list[tuple[str, str | None]]]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        lora_id = self._lora_ids.setdefault(adapter_name, len(self._lora_ids) + 1)
        request = LoRARequest(adapter_name, lora_id, str(adapter_path))
        generation = self.generation
        parameters = [
            SamplingParams(
                n=int(generation["n"]),
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                repetition_penalty=float(generation["repetition_penalty"]),
                max_tokens=int(generation["max_tokens"]),
                seed=seed,
            )
            for seed in seeds
        ]
        outputs = self.llm.generate(
            [{"prompt_token_ids": ids} for ids in prompt_token_ids],
            parameters,
            lora_request=request,
            use_tqdm=True,
        )
        return [
            [(sample.text, sample.finish_reason) for sample in output.outputs]
            for output in outputs
        ]


def resolve_expert_adapters(
    stage1_dir: Path,
    stage1_manifest: dict[str, Any],
    checkpoint: str,
    experts: list[str] | None = None,
) -> list[tuple[str, Path]]:
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"stage1_checkpoint must be one of {CHECKPOINTS}, got {checkpoint!r}")
    available = list(stage1_manifest["adapter_names"])
    selected = list(experts) if experts else available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown Stage-1 experts {unknown}; available: {available}")
    resolved: list[tuple[str, Path]] = []
    for name in selected:
        adapter_dir = stage1_dir / checkpoint / "adapters" / name
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (adapter_dir / filename).is_file():
                raise FileNotFoundError(f"Missing {adapter_dir / filename}")
        if checkpoint == "final":
            files = stage1_manifest["expert_files"][name]
            require_file_sha256(stage1_dir, files, "adapter_config", "adapter_config_sha256")
            require_file_sha256(stage1_dir, files, "adapter_weights", "adapter_weights_sha256")
        resolved.append((name, adapter_dir))
    return resolved


def benchmark_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    metrics = pass_metrics(records)
    reasons = [reason for record in records for reason in record.get("finish_reasons", [])]
    metrics["length_capped_fraction"] = (
        sum(reason == "length" for reason in reasons) / len(reasons) if reasons else 0.0
    )
    return metrics


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _cached_metrics(
    predictions: Path, metadata_path: Path, run_key: str
) -> dict[str, float | int] | None:
    if not (predictions.is_file() and metadata_path.is_file()):
        return None
    metadata = read_json(metadata_path)
    if metadata.get("run_key") != run_key:
        return None
    if metadata.get("predictions_sha256") != file_sha256(predictions):
        return None
    return metadata["metrics"]


def _evaluate_benchmark(
    engine: ExpertEngine,
    config: dict[str, Any],
    benchmark: dict[str, Any],
    expert: str,
    adapter_path: Path,
) -> list[dict[str, Any]]:
    name = str(benchmark["name"])
    generation = config["generation"]
    max_model_len = int(config["vllm"]["max_model_len"])
    dataset = _load_benchmark(benchmark)
    expected_records = benchmark.get("expected_records")
    if expected_records is not None and len(dataset) != int(expected_records):
        raise RuntimeError(
            f"Benchmark {name} has {len(dataset)} rows; expected {int(expected_records)}"
        )
    rows = [dict(dataset[index]) for index in range(len(dataset))]
    problems = [_question(row) for row in rows]
    references = [_answer(row) for row in rows]
    prompts = [
        _chat_prompt(engine.tokenizer, str(generation["prompt_prefix"]), problem)
        for problem in problems
    ]
    prompt_token_ids = [
        list(engine.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for prompt in prompts
    ]
    for index, ids in enumerate(prompt_token_ids):
        if len(ids) >= max_model_len:
            raise RuntimeError(
                f"{name}[{index}] prompt has {len(ids)} tokens; vllm.max_model_len={max_model_len}"
            )
    seeds = [
        derived_seed(int(config["seed"]), "evaluation", name, index)
        for index in range(len(rows))
    ]
    LOGGER.info("Generating %s for %s (%d problems)", name, expert, len(rows))
    outputs = engine.generate(prompt_token_ids, seeds, expert, adapter_path)
    if len(outputs) != len(rows):
        raise RuntimeError(f"Engine returned {len(outputs)} outputs for {len(rows)} prompts")

    prefer_math_verify = bool(config["grading"].get("prefer_math_verify", True))
    timeout = float(config["grading"]["timeout_seconds"])
    records: list[dict[str, Any]] = []
    for index, samples in enumerate(tqdm(outputs, desc=f"Grade {expert}/{name}")):
        responses = [text for text, _ in samples]
        records.append(
            {
                "expert": expert,
                "benchmark": name,
                "index": index,
                "problem": problems[index],
                "reference": references[index],
                "prompt": prompts[index],
                "generations": responses,
                "finish_reasons": [reason for _, reason in samples],
                "parsed_answers": [normalize_answer(response) for response in responses],
                "parsed_reference": normalize_answer(references[index]),
                "correct": [
                    grade_math(response, references[index], prefer_math_verify, timeout)
                    for response in responses
                ],
                "seed": seeds[index],
            }
        )
    return records


def _format_summary(results: dict[str, dict[str, Any]], benchmarks: list[str]) -> str:
    header = ["expert", *benchmarks, "macro"]
    lines = [" | ".join(header), " | ".join("---" for _ in header)]
    for expert, result in results.items():
        cells = [expert]
        for name in benchmarks:
            values = result["benchmarks"][name]
            cells.append(f"{values['pass_at_1']:.4f} / {values['pass_at_3']:.4f}")
        macro = result["macro_average"]
        cells.append(f"{macro['pass_at_1']:.4f} / {macro['pass_at_3']:.4f}")
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def evaluate_stage1_experts(
    config: dict[str, Any],
    experts: list[str] | None = None,
    force: bool = False,
    engine: ExpertEngine | None = None,
) -> dict[str, Any]:
    """Evaluate each Stage-1 LoRA expert in turn on the configured benchmarks.

    Per (expert, benchmark) results are cached under paths.output/<expert>/ and
    reused when the adapter weights and evaluation settings are unchanged.
    """
    stage1_dir = Path(config["paths"]["stage1"])
    output_dir = Path(config["paths"]["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    config_snapshot = output_dir / "config.yaml"
    temporary = config_snapshot.with_suffix(".yaml.tmp")
    temporary.write_text(
        yaml.safe_dump(public_config, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )
    temporary.replace(config_snapshot)

    stage1_manifest = read_json(stage1_dir / "manifest.json")
    require_same_model_source(
        config["model"], stage1_manifest["config"]["model"], "Stage-1 evaluation/Stage 1"
    )
    checkpoint = str(config.get("stage1_checkpoint", "final"))
    adapters = resolve_expert_adapters(stage1_dir, stage1_manifest, checkpoint, experts)
    if int(stage1_manifest["config"]["lora"]["rank"]) > int(config["vllm"]["max_lora_rank"]):
        raise ValueError("vllm.max_lora_rank is smaller than the Stage-1 LoRA rank")

    settings = {
        key: public_config[key] for key in ("seed", "model", "vllm", "generation", "grading")
    }
    benchmark_names = [str(benchmark["name"]) for benchmark in config["benchmarks"]]
    results: dict[str, dict[str, Any]] = {}
    for position, (expert, adapter_path) in enumerate(adapters, start=1):
        LOGGER.info("[%d/%d] Evaluating %s from %s", position, len(adapters), expert, adapter_path)
        expert_dir = output_dir / expert
        expert_dir.mkdir(parents=True, exist_ok=True)
        adapter_fingerprint = files_fingerprint(
            [adapter_path / "adapter_config.json", adapter_path / "adapter_model.safetensors"]
        )
        per_benchmark: dict[str, dict[str, float | int]] = {}
        for benchmark in config["benchmarks"]:
            name = str(benchmark["name"])
            predictions = expert_dir / f"{name}.jsonl"
            metadata_path = expert_dir / f"{name}.metrics.json"
            run_key = fingerprint(
                {"settings": settings, "benchmark": benchmark, "adapter": adapter_fingerprint}
            )
            cached = None if force else _cached_metrics(predictions, metadata_path, run_key)
            if cached is not None:
                LOGGER.info("Reusing %s/%s: %s", expert, name, cached)
                per_benchmark[name] = cached
                continue
            if engine is None:
                engine = VllmExpertEngine(
                    config["model"], config["vllm"], config["generation"], int(config["seed"])
                )
            records = _evaluate_benchmark(engine, config, benchmark, expert, adapter_path)
            _write_jsonl(predictions, records)
            metrics = benchmark_metrics(records)
            write_json(
                metadata_path,
                {
                    "run_key": run_key,
                    "predictions_sha256": file_sha256(predictions),
                    "metrics": metrics,
                },
            )
            LOGGER.info("%s/%s: %s", expert, name, metrics)
            per_benchmark[name] = metrics
        results[expert] = {
            "adapter_path": str(adapter_path),
            "adapter_fingerprint": adapter_fingerprint,
            "benchmarks": per_benchmark,
            "macro_average": macro_average(per_benchmark),
        }
        write_json(expert_dir / "manifest.json", results[expert])

    summary = _format_summary(results, benchmark_names)
    (output_dir / "summary.md").write_text(
        "Pass@1 / Pass@3 per benchmark\n\n" + summary + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "artifact": "stage1_expert_evaluation",
        "stage1_dir": str(stage1_dir),
        "stage1_checkpoint": checkpoint,
        "stage1_manifest_fingerprint": fingerprint(stage1_manifest),
        "experts": results,
        "config": public_config,
        "config_fingerprint": fingerprint(public_config),
        "config_file": config_snapshot.name,
        "config_file_sha256": file_sha256(config_snapshot),
        "runtime": runtime_metadata(config["_project_root"]),
    }
    write_json(output_dir / "manifest.json", manifest)
    LOGGER.info("Stage-1 expert evaluation (Pass@1 / Pass@3):\n%s", summary)
    return manifest
