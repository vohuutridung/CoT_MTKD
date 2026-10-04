"""P-ALIGN evaluation, identical in protocol to exp_s1k ``eval/run_eval.py``.

For every seed: render ``"<instruction> <problem>"`` with the chat template
(single user turn, default system prompt), sample k completions, grade them
with :func:`grade_answer` and report Pass@1 as the mean over the k samples.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from ..utils.manifest import runtime_metadata, write_config_snapshot, write_json
from .adapter import adapter_rank, resolve_adapter
from .benchmarks import load_benchmark
from .grading import grade_samples
from .metrics import benchmark_metrics, macro_average, seed_summary

LOGGER = logging.getLogger(__name__)

PROTOCOL_KEYS = ("model", "adapter", "generation", "benchmarks", "limit")


def render_prompt(tokenizer: Any, instruction: str, problem: str) -> str:
    content = f"{instruction} {problem}" if instruction else problem
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _grade(arguments: tuple[list[str], str]) -> list[bool]:
    return grade_samples(*arguments)


def _check_model_source(config: dict[str, Any], manifest: dict[str, Any] | None) -> None:
    if manifest is None:
        LOGGER.warning("No Stage-2 manifest next to the adapter; model source not checked")
        return
    trained = manifest["config"]["model"]
    for key in ("name_or_path", "revision"):
        if trained.get(key) != config["model"].get(key):
            raise RuntimeError(
                f"Adapter was trained on model {key}={trained.get(key)!r}, "
                f"but evaluation uses {config['model'].get(key)!r}"
            )


def _build_engine(config: dict[str, Any], adapter_dir: Path | None, tokenizer: Any):
    name = config["engine"]["name"]
    if name == "vllm":
        from .engines import VLLMEngine

        return VLLMEngine(
            config["model"],
            config["generation"],
            config["engine"],
            adapter_dir,
            adapter_rank(adapter_dir),
        )
    if name == "hf":
        from .engines import HFEngine

        return HFEngine(config["model"], config["generation"], adapter_dir, tokenizer)
    raise ValueError(f"Unknown engine {name!r}; use vllm or hf")


def _evaluate_seed(
    config: dict[str, Any],
    engine: Any,
    tokenizer: Any,
    benchmarks: dict[str, list[Any]],
    seed: int,
    seed_dir: Path,
    executor: ProcessPoolExecutor,
) -> dict[str, Any]:
    seed_dir.mkdir(parents=True, exist_ok=True)
    instruction = str(config["generation"]["instruction"])
    started = time.time()
    results: dict[str, Any] = {}
    for name, items in benchmarks.items():
        prompts = [render_prompt(tokenizer, instruction, item.question) for item in items]
        samples = engine.generate(prompts, seed)
        texts = [[sample.text for sample in group] for group in samples]
        correct = list(
            executor.map(
                _grade, [(group, item.answer) for group, item in zip(texts, items)], chunksize=4
            )
        )
        records = [
            {
                "index": item.index,
                "question": item.question,
                "gold": item.answer,
                "completions": [sample.text for sample in group],
                "token_counts": [sample.num_tokens for sample in group],
                "truncated": [sample.truncated for sample in group],
                "correct": flags,
            }
            for item, group, flags in zip(items, samples, correct)
        ]
        with (seed_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        results[name] = benchmark_metrics(records)
        LOGGER.info(
            "seed %d | %s: Pass@1 %.2f | Pass@%d %.2f | mean tokens %.0f | truncated %.1f%%",
            seed,
            name,
            100 * results[name]["pass@1"],
            results[name]["k"],
            100 * results[name][f"pass@{results[name]['k']}"],
            results[name]["mean_tokens"],
            100 * results[name]["truncated_fraction"],
        )
    summary = {
        "seed": seed,
        "benchmarks": results,
        **macro_average(results),
        "runtime_sec": time.time() - started,
    }
    write_json(seed_dir / "result.json", summary)
    LOGGER.info("seed %d | Avg Pass@1 %.2f", seed, 100 * summary["avg"])
    return summary


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    project_root = Path(config["_project_root"])
    output_dir = Path(config["paths"]["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    # Seeds, engine and overwrite may change between invocations; the protocol may not.
    write_config_snapshot(
        output_dir / "protocol.yaml",
        {key: public_config.get(key) for key in PROTOCOL_KEYS},
    )

    adapter_dir, stage2_manifest = resolve_adapter(config["adapter"], project_root)
    _check_model_source(config, stage2_manifest)
    LOGGER.info("Evaluating adapter %s", adapter_dir or "<base model>")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name_or_path"], revision=config["model"].get("revision")
    )
    limit = config.get("limit")
    benchmarks = {
        str(spec["name"]): load_benchmark(spec, project_root, limit)
        for spec in config["benchmarks"]
    }
    seeds = [int(seed) for seed in config["seeds"]]
    overwrite = bool(config.get("overwrite", False))

    results: dict[int, dict[str, Any]] = {}
    engine = None
    with ProcessPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as executor:
        for seed in seeds:
            seed_dir = output_dir / f"seed{seed}"
            done = seed_dir / "result.json"
            if done.exists() and not overwrite:
                LOGGER.info("seed %d already evaluated; skipping (%s)", seed, done)
                results[seed] = json.loads(done.read_text(encoding="utf-8"))
                continue
            if engine is None:
                engine = _build_engine(config, adapter_dir, tokenizer)
            results[seed] = _evaluate_seed(
                config, engine, tokenizer, benchmarks, seed, seed_dir, executor
            )

    summary = {
        "schema_version": 2,
        "artifact": "evaluation_run",
        "adapter": str(adapter_dir) if adapter_dir else None,
        "stage2_global_step": (stage2_manifest or {}).get("global_step"),
        **seed_summary(results),
        "config": public_config,
        "runtime": runtime_metadata(str(project_root)),
    }
    write_json(output_dir / "summary.json", summary)
    LOGGER.info(
        "Avg Pass@1 over seeds %s: %.2f ± %.2f | Avg Pass@k %.2f",
        seeds,
        100 * summary["avg"]["mean"],
        100 * summary["avg"]["std"],
        100 * summary["avg_passk"]["mean"],
    )
    for name, value in summary["benchmarks"].items():
        LOGGER.info(
            "  %-8s Pass@1 %.2f ± %.2f", name, 100 * value["pass@1"]["mean"],
            100 * value["pass@1"]["std"],
        )
    return summary
