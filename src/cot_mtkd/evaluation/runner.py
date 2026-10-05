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
    write_config_snapshot,
    write_json,
)
from ..utils.seed import derived_seed
from .grading import grade_math, normalize_answer
from .metrics import macro_average, pass_metrics

LOGGER = logging.getLogger(__name__)


def generation_token_limit(prompt_tokens: int, max_model_len: int, max_tokens: int) -> int:
    """Cap the completion so prompt plus completion fit in ``max_model_len``."""
    if prompt_tokens < 1:
        raise ValueError("prompt_tokens must be positive")
    if max_model_len < 2:
        raise ValueError("vllm.max_model_len must leave room for a completion")
    if max_tokens < 1:
        raise ValueError("generation.max_tokens must be positive")
    room = int(max_model_len) - int(prompt_tokens)
    if room < 1:
        raise RuntimeError(
            f"Prompt has {prompt_tokens} tokens, which does not fit in "
            f"vllm.max_model_len={max_model_len}"
        )
    return min(int(max_tokens), room)


class StudentEngine(Protocol):
    tokenizer: Any

    def generate(
        self,
        prompt_token_ids: list[list[int]],
        seeds: list[int],
    ) -> list[list[tuple[str, str | None]]]:
        """Return ``(text, finish_reason)`` for every sample of every prompt."""


class VllmStudentEngine:
    def __init__(
        self,
        model_config: dict[str, Any],
        vllm_config: dict[str, Any],
        generation: dict[str, Any],
        adapter_path: Path,
        seed: int,
    ) -> None:
        try:
            from vllm import LLM
        except ImportError as error:
            raise RuntimeError(
                "Evaluation needs vLLM. Install it in the project environment and rerun."
            ) from error

        self.n = int(generation["n"])
        self.temperature = float(generation["temperature"])
        self.top_p = float(generation["top_p"])
        self.repetition_penalty = float(generation["repetition_penalty"])
        self.max_tokens = int(generation["max_tokens"])
        self.max_model_len = int(vllm_config["max_model_len"])
        self.llm = LLM(
            model=model_config["name_or_path"],
            revision=model_config.get("revision"),
            tokenizer_revision=model_config.get("revision"),
            dtype=str(model_config.get("dtype", "bfloat16")),
            max_model_len=self.max_model_len,
            tensor_parallel_size=int(vllm_config["tensor_parallel_size"]),
            gpu_memory_utilization=float(vllm_config["gpu_memory_utilization"]),
            enable_lora=True,
            max_lora_rank=int(vllm_config["max_lora_rank"]),
            max_loras=1,
            seed=seed,
            trust_remote_code=False,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.adapter_path = adapter_path

    def generate(
        self,
        prompt_token_ids: list[list[int]],
        seeds: list[int],
    ) -> list[list[tuple[str, str | None]]]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        if len(prompt_token_ids) != len(seeds):
            raise ValueError("Each prompt needs its own sampling seed")
        parameters = [
            SamplingParams(
                n=self.n,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                max_tokens=generation_token_limit(
                    len(ids), self.max_model_len, self.max_tokens
                ),
                seed=seed,
            )
            for ids, seed in zip(prompt_token_ids, seeds, strict=True)
        ]
        request = LoRARequest("student", 1, str(self.adapter_path))
        outputs = self.llm.generate(
            [{"prompt_token_ids": ids} for ids in prompt_token_ids],
            parameters,
            lora_request=request,
            use_tqdm=True,
        )
        if len(outputs) != len(prompt_token_ids):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(prompt_token_ids)} prompts"
            )
        samples = []
        for output in outputs:
            if len(output.outputs) != self.n:
                raise RuntimeError(
                    f"vLLM returned {len(output.outputs)} samples; generation.n={self.n}"
                )
            samples.append(
                [(sample.text, sample.finish_reason) for sample in output.outputs]
            )
        return samples


def _question(record: dict[str, Any]) -> str:
    for key in ("problem", "question", "prompt"):
        if key in record:
            return str(record[key])
    raise KeyError(f"Could not find a question field in {sorted(record)}")


def _answer(record: dict[str, Any]) -> Any:
    for key in ("answer", "solution", "final_answer"):
        if key in record:
            return record[key]
    raise KeyError(f"Could not find an answer field in {sorted(record)}")


def _load_benchmark(specification: dict[str, Any]):
    from datasets import load_dataset

    local_json = specification.get("local_json")
    if local_json:
        return load_dataset(
            "json",
            data_files=str(Path(local_json)),
            split=specification.get("split", "train"),
        )
    return load_dataset(
        specification["dataset"],
        split=specification["split"],
        revision=specification.get("revision"),
    )


def _chat_prompt(tokenizer: Any, prefix: str, problem: str) -> str:
    content = f"{prefix}\n\n{problem}"
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return content


def _student_adapter_dir(stage2_dir: Path, manifest: dict[str, Any]) -> Path:
    name = str(manifest["student_adapter"])
    adapter_dir = stage2_dir / "final" / "adapters" / name
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_dir / filename).is_file():
            raise FileNotFoundError(f"Missing {adapter_dir / filename}")
    return adapter_dir


def _length_capped_fraction(records: list[dict[str, Any]]) -> float:
    reasons = [reason for record in records for reason in record.get("finish_reasons", [])]
    if not reasons:
        return 0.0
    return sum(reason == "length" for reason in reasons) / len(reasons)


def _write_eval_config(path: Path, value: dict[str, Any]) -> None:
    """Replace an older evaluation snapshot so a new sampling setup can rerun."""
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            existing = yaml.safe_load(handle)
        if existing != value:
            LOGGER.warning(
                "Replacing evaluation config snapshot %s with the current settings",
                path,
            )
            path.unlink()
    write_config_snapshot(path, value)


def evaluate(config: dict[str, Any], engine: StudentEngine | None = None) -> dict[str, Any]:
    """Generate n samples per problem with vLLM and grade Pass@1 / Pass@3."""
    stage2_dir = Path(config["paths"]["stage2"])
    output_dir = Path(config["paths"]["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    config_snapshot = output_dir / "config.yaml"
    _write_eval_config(config_snapshot, public_config)

    stage2_manifest = read_json(stage2_dir / "manifest.json")
    require_file_sha256(stage2_dir, stage2_manifest, "adapter_bundle", "adapter_bundle_sha256")
    require_file_sha256(stage2_dir, stage2_manifest, "config_file", "config_file_sha256")
    stage2_config = stage2_manifest["config"]
    require_same_model_source(config["model"], stage2_config["model"], "Evaluation/Stage 2")
    adapter_dir = _student_adapter_dir(stage2_dir, stage2_manifest)
    lora_rank = int(stage2_config["lora"]["rank"])
    if lora_rank > int(config["vllm"]["max_lora_rank"]):
        raise ValueError(
            f"vllm.max_lora_rank={config['vllm']['max_lora_rank']} is smaller than "
            f"the student LoRA rank {lora_rank}"
        )
    if int(config["generation"]["n"]) != 3:
        raise ValueError("generation.n must be 3 so Pass@1 and Pass@3 match the evaluation definition")

    if engine is None:
        engine = VllmStudentEngine(
            config["model"],
            config["vllm"],
            config["generation"],
            adapter_dir,
            int(config["seed"]),
        )
    generation = config["generation"]
    prefer_math_verify = bool(config["grading"].get("prefer_math_verify", True))
    timeout = float(config["grading"]["timeout_seconds"])
    benchmark_metrics: dict[str, dict[str, float | int]] = {}

    for benchmark in config["benchmarks"]:
        name = str(benchmark["name"])
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
        seeds = [
            derived_seed(int(config["seed"]), "evaluation", name, index)
            for index in range(len(rows))
        ]
        LOGGER.info("Generating %s (%d problems, n=%s)", name, len(rows), generation["n"])
        outputs = engine.generate(prompt_token_ids, seeds)
        if len(outputs) != len(rows):
            raise RuntimeError(f"Engine returned {len(outputs)} outputs for {len(rows)} prompts")

        records = []
        for index, samples in enumerate(tqdm(outputs, desc=f"Grade {name}")):
            responses = [text for text, _reason in samples]
            records.append(
                {
                    "benchmark": name,
                    "index": index,
                    "problem": problems[index],
                    "reference": references[index],
                    "prompt": prompts[index],
                    "generations": responses,
                    "finish_reasons": [reason for _text, reason in samples],
                    "parsed_answers": [normalize_answer(response) for response in responses],
                    "parsed_reference": normalize_answer(references[index]),
                    "correct": [
                        grade_math(response, references[index], prefer_math_verify, timeout)
                        for response in responses
                    ],
                    "seed": seeds[index],
                }
            )
        combined_path = output_dir / f"{name}.jsonl"
        temporary = combined_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(combined_path)
        metrics = pass_metrics(records)
        metrics["length_capped_fraction"] = _length_capped_fraction(records)
        benchmark_metrics[name] = metrics
        LOGGER.info("%s: %s", name, metrics)

    prediction_files = [f"{benchmark['name']}.jsonl" for benchmark in config["benchmarks"]]
    manifest = {
        "schema_version": 1,
        "artifact": "evaluation_run",
        "backend": "vllm",
        "student_adapter": str(adapter_dir),
        "benchmarks": benchmark_metrics,
        "macro_average": macro_average(benchmark_metrics),
        "prediction_files": prediction_files,
        "prediction_files_fingerprint": files_fingerprint(
            output_dir / name for name in prediction_files
        ),
        "stage2_manifest_fingerprint": fingerprint(stage2_manifest),
        "config": public_config,
        "config_fingerprint": fingerprint(public_config),
        "config_file": config_snapshot.name,
        "config_file_sha256": file_sha256(config_snapshot),
        "runtime": runtime_metadata(config["_project_root"]),
    }
    write_json(output_dir / "manifest.json", manifest)
    LOGGER.info("Evaluation macro average: %s", manifest["macro_average"])
    return manifest
