from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ..models.multi_adapter import (
    create_student_model,
    load_adapter_bundle,
    load_adapter_state,
    load_tokenizer,
    require_same_model_source,
)
from ..utils.distributed import DistributedContext, barrier, shard_indices
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


def evaluate(config: dict[str, Any], distributed: DistributedContext) -> dict[str, Any]:
    stage2_dir = Path(config["paths"]["stage2"])
    output_dir = Path(config["paths"]["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    public_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    config_snapshot = output_dir / "config.yaml"
    if distributed.is_main:
        write_config_snapshot(config_snapshot, public_config)
    barrier()
    stage2_manifest = read_json(stage2_dir / "manifest.json")
    require_file_sha256(
        stage2_dir, stage2_manifest, "adapter_bundle", "adapter_bundle_sha256"
    )
    require_file_sha256(
        stage2_dir, stage2_manifest, "config_file", "config_file_sha256"
    )
    stage2_config = stage2_manifest["config"]
    require_same_model_source(
        config["model"], stage2_config["model"], "Evaluation/Stage 2"
    )
    tokenizer = load_tokenizer(config["model"])
    model = create_student_model(
        config["model"], stage2_config["lora"], distributed.device, int(config["seed"])
    )
    bundle = load_adapter_bundle(stage2_dir / stage2_manifest["adapter_bundle"])
    load_adapter_state(model, "student", bundle["student"])
    model.eval()
    generation = config["generation"]

    for benchmark in config["benchmarks"]:
        name = str(benchmark["name"])
        dataset = _load_benchmark(benchmark)
        expected_records = benchmark.get("expected_records")
        if expected_records is not None and len(dataset) != int(expected_records):
            raise RuntimeError(
                f"Benchmark {name} has {len(dataset)} rows; expected {int(expected_records)}"
            )
        shard_path = output_dir / (
            f"{name}-rank{distributed.rank:05d}-of{distributed.world_size:05d}.jsonl"
        )
        temporary = shard_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            iterator = shard_indices(
                len(dataset), distributed.rank, distributed.world_size
            )
            for index in tqdm(
                iterator, desc=f"Evaluate {name} rank {distributed.rank}"
            ):
                raw = dict(dataset[index])
                problem = _question(raw)
                reference = _answer(raw)
                prompt = _chat_prompt(
                    tokenizer, str(generation["prompt_prefix"]), problem
                )
                encoded = tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=False
                )
                encoded = {
                    key: value.to(distributed.device) for key, value in encoded.items()
                }
                seed = derived_seed(int(config["seed"]), "evaluation", name, index)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                with torch.inference_mode():
                    outputs = model.generate(
                        **encoded,
                        do_sample=bool(generation["do_sample"]),
                        temperature=float(generation["temperature"]),
                        top_p=float(generation["top_p"]),
                        repetition_penalty=float(generation["repetition_penalty"]),
                        max_new_tokens=int(generation["max_new_tokens"]),
                        num_return_sequences=int(generation["num_return_sequences"]),
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                prompt_length = encoded["input_ids"].shape[1]
                responses = tokenizer.batch_decode(
                    outputs[:, prompt_length:], skip_special_tokens=True
                )
                correct = [
                    grade_math(
                        response,
                        reference,
                        bool(config["grading"].get("prefer_math_verify", True)),
                        float(config["grading"]["timeout_seconds"]),
                    )
                    for response in responses
                ]
                parsed_answers = [normalize_answer(response) for response in responses]
                handle.write(
                    json.dumps(
                        {
                            "benchmark": name,
                            "index": index,
                            "problem": problem,
                            "reference": reference,
                            "prompt": prompt,
                            "generations": responses,
                            "parsed_answers": parsed_answers,
                            "parsed_reference": normalize_answer(reference),
                            "correct": correct,
                            "seed": seed,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        temporary.replace(shard_path)
    barrier()
    if distributed.is_main:
        benchmark_metrics: dict[str, dict[str, float | int]] = {}
        for benchmark in config["benchmarks"]:
            name = str(benchmark["name"])
            records = []
            paths = [
                output_dir
                / f"{name}-rank{rank:05d}-of{distributed.world_size:05d}.jsonl"
                for rank in range(distributed.world_size)
            ]
            for path in paths:
                with path.open("r", encoding="utf-8") as handle:
                    records.extend(json.loads(line) for line in handle if line.strip())
            records.sort(key=lambda value: int(value["index"]))
            expected_records = benchmark.get("expected_records")
            if expected_records is not None and len(records) != int(expected_records):
                raise RuntimeError(
                    f"Generated {len(records)} records for {name}; "
                    f"expected {int(expected_records)}"
                )
            combined_path = output_dir / f"{name}.jsonl"
            with combined_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            benchmark_metrics[name] = pass_metrics(records)
        prediction_files = [
            f"{benchmark['name']}.jsonl" for benchmark in config["benchmarks"]
        ]
        manifest = {
            "schema_version": 1,
            "artifact": "evaluation_run",
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
    barrier()
    return read_json(output_dir / "manifest.json")
