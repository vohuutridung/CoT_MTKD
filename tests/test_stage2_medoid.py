from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from cot_mtkd.data.schema import PreparedRecord, TokenRegion
from cot_mtkd.stage2.medoid import (
    build_stage2_medoid,
    full_vocab_medoid_scores,
    score_record_medoid,
)
from cot_mtkd.utils.distributed import DistributedContext
from cot_mtkd.utils.manifest import file_sha256, fingerprint, write_json


def prepared_record(sample_id: str, token_ids: list[int]) -> PreparedRecord:
    length = len(token_ids)
    return PreparedRecord(
        sample_id=sample_id,
        input_ids=token_ids,
        labels=[-100, -100] + token_ids[2:],
        attention_mask=[1] * length,
        offset_mapping=[(index, index + 1) for index in range(length)],
        region_ids=[
            int(TokenRegion.PROMPT),
            int(TokenRegion.PROMPT),
            int(TokenRegion.ASSISTANT_CONTROL),
        ] + [int(TokenRegion.ANSWER)] * (length - 3),
        step_ids=[-1] * length,
        question="question",
        thinking="",
        attempt="attempt",
        solution="gold solution",
        deepseek_grade="No",
        original_length=length,
        kept_length=length,
        original_steps=0,
        kept_steps=0,
        truncated=False,
        answer_start=3,
        reasoning_start=3,
        tokenizer_fingerprint="test-tokenizer",
    )


class FakeDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 4)
        self.teacher_index = 0
        self.last_attention_mask: torch.Tensor | None = None

    def forward(self, input_ids, attention_mask, **kwargs):
        self.last_attention_mask = attention_mask
        offsets = self.embedding.weight.new_tensor([0.1, -0.3, 0.2, -0.1])
        return SimpleNamespace(
            last_hidden_state=self.embedding(input_ids) + offsets * self.teacher_index
        )


class FakeCouncil(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeDecoder()
        self.head = torch.nn.Linear(4, 11, bias=False)

    def get_output_embeddings(self):
        return self.head

    def set_adapter(self, name: str) -> None:
        self.model.teacher_index = int(name.rsplit("_", 1)[1])
        # Emulate PEFT's automatic re-enabling of active adapter parameters.
        self.requires_grad_(True)


class Stage2MedoidTest(unittest.TestCase):
    def test_chunked_full_vocabulary_scores_match_dense_reference(self) -> None:
        torch.manual_seed(531)
        head = torch.nn.Linear(5, 17, bias=True)
        hidden = [torch.randn(7, 5) for _ in range(3)]
        temperature = 2.0
        with torch.no_grad():
            log_probabilities = torch.stack(
                [F.log_softmax(head(value).float() / temperature, dim=-1) for value in hidden]
            )
            mixture = log_probabilities.exp().mean(dim=0)
            expected = (
                mixture.unsqueeze(0)
                * (mixture.clamp_min(1.0e-30).log().unsqueeze(0) - log_probabilities)
            ).sum(dim=-1).clamp_min(0.0).double().sum(dim=-1)
        for chunk_tokens in (1, 3, 20):
            actual, count = full_vocab_medoid_scores(hidden, head, chunk_tokens, temperature)
            self.assertEqual(count, 7)
            self.assertEqual(actual.dtype, torch.float64)
            self.assertFalse(actual.requires_grad)
            # Token chunk sizes select different FP32 GEMM kernels.
            torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-5)

    def test_identical_teachers_have_zero_score_and_empty_tokens_have_zero_count(self) -> None:
        head = torch.nn.Linear(3, 13)
        hidden = torch.randn(6, 3)
        scores, count = full_vocab_medoid_scores([hidden, hidden], head, 2, 2.0)
        self.assertEqual(count, 6)
        torch.testing.assert_close(scores, torch.zeros_like(scores), atol=1.0e-6, rtol=0.0)
        empty = hidden[:0]
        scores, count = full_vocab_medoid_scores([empty, empty], head, 2, 2.0)
        self.assertEqual(count, 0)
        self.assertEqual(scores.tolist(), [0.0, 0.0])

    def test_record_scores_all_assistant_targets_and_keeps_council_frozen(self) -> None:
        torch.manual_seed(881)
        model = FakeCouncil()
        record = prepared_record("all-regions", [1, 2, 3, 4, 5, 6, 7, 8])
        record.region_ids[3:] = [
            int(TokenRegion.REASONING),
            int(TokenRegion.DELIMITER),
            int(TokenRegion.ANSWER_MARKER),
            int(TokenRegion.ANSWER),
            int(TokenRegion.EOS),
        ]
        scores, count = score_record_medoid(
            model, ["expert_0", "expert_1"], record, torch.device("cpu"), 2, 2.0
        )
        with torch.no_grad():
            hidden = model.model.embedding(torch.tensor(record.input_ids))[1:-1]
            offsets = hidden.new_tensor([0.1, -0.3, 0.2, -0.1])
            expected, _ = full_vocab_medoid_scores([hidden, hidden + offsets], model.head, 10, 2.0)
        self.assertEqual(count, 6)
        torch.testing.assert_close(scores, expected, atol=1.0e-7, rtol=1.0e-6)
        self.assertFalse(model.training)
        self.assertTrue(all(not value.requires_grad for value in model.parameters()))
        self.assertTrue(all(value.grad is None for value in model.parameters()))
        self.assertEqual(model.model.last_attention_mask.tolist(), [record.attention_mask])

    def _build_fixture(self, root: Path):
        prepared = root / "prepared"
        stage1 = root / "stage1"
        prepared.mkdir()
        stage1.mkdir()
        records = [prepared_record("a", [1, 2, 3, 4]), prepared_record("b", [5, 6, 7, 8, 9])]
        data_path = prepared / "data.jsonl"
        data_path.write_text("".join(json.dumps(asdict(value)) + "\n" for value in records))
        (prepared / "config.yaml").write_text("fixture: prepared\n")
        prepared_manifest = {
            "records": len(records),
            "data_file": "data.jsonl",
            "data_file_sha256": file_sha256(data_path),
            "config_file": "config.yaml",
            "config_file_sha256": file_sha256(prepared / "config.yaml"),
            "tokenizer_fingerprint": "test-tokenizer",
        }
        write_json(prepared / "manifest.json", prepared_manifest)
        (stage1 / "config.yaml").write_text("fixture: stage1\n")
        torch.save({"expert_0": {}, "expert_1": {}}, stage1 / "adapters.pt")
        model_config = {"name_or_path": "fixture/model", "revision": "fixed-revision"}
        stage1_manifest = {
            "prepared_manifest_fingerprint": fingerprint(prepared_manifest),
            "tokenizer_fingerprint": "test-tokenizer",
            "adapter_names": ["expert_0", "expert_1"],
            "adapter_bundle": "adapters.pt",
            "adapter_bundle_sha256": file_sha256(stage1 / "adapters.pt"),
            "config_file": "config.yaml",
            "config_file_sha256": file_sha256(stage1 / "config.yaml"),
            "config": {"model": model_config, "lora": {}, "seed": 42},
        }
        write_json(stage1 / "manifest.json", stage1_manifest)
        config = {
            "model": model_config,
            "geometry": {"temperature": 2.0},
            "runtime": {"lm_head_chunk_tokens": 2},
            "paths": {
                "prepared": str(prepared),
                "stage1": str(stage1),
                "medoid": str(root / "medoid"),
            },
            "_project_root": str(root),
        }
        return config, records, prepared_manifest, stage1_manifest

    def test_builder_scores_whole_corpus_and_records_provenance(self) -> None:
        torch.manual_seed(601)
        model = FakeCouncil()
        context = DistributedContext(0, 0, 1, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            config, records, prepared_manifest, stage1_manifest = self._build_fixture(Path(directory))
            expected = torch.zeros(2, dtype=torch.float64)
            total_tokens = 0
            for record in records:
                scores, tokens = score_record_medoid(
                    model, ["expert_0", "expert_1"], record, context.device, 2, 2.0
                )
                expected += scores
                total_tokens += tokens
            expected /= total_tokens
            with (
                patch("cot_mtkd.stage2.medoid.load_tokenizer", return_value=object()),
                patch("cot_mtkd.stage2.medoid.tokenizer_fingerprint", return_value="test-tokenizer"),
                patch(
                    "cot_mtkd.stage2.medoid.create_multi_adapter_model",
                    return_value=(model, ["expert_0", "expert_1"]),
                ),
                patch("cot_mtkd.stage2.medoid.load_adapter_state"),
                patch("cot_mtkd.stage2.medoid.runtime_metadata", return_value={"fixture": True}),
            ):
                manifest = build_stage2_medoid(config, context)
            self.assertEqual(manifest["artifact"], "stage2_medoid")
            self.assertEqual(manifest["records"], 2)
            self.assertEqual(manifest["response_token_count"], 5)
            self.assertEqual(manifest["functional_medoid_index"], int(expected.argmin()))
            torch.testing.assert_close(torch.tensor(manifest["functional_medoid_kl"]), expected.float())
            self.assertEqual(manifest["prepared_manifest_fingerprint"], fingerprint(prepared_manifest))
            self.assertEqual(manifest["stage1_manifest_fingerprint"], fingerprint(stage1_manifest))
            self.assertEqual(manifest["model_source"], config["model"])
            self.assertNotIn("teacher_weights", manifest)
            self.assertTrue(Path(config["paths"]["medoid"], "config.yaml").is_file())

    def test_tampered_dataset_is_rejected_before_teacher_loading(self) -> None:
        context = DistributedContext(0, 0, 1, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            config, _, _, _ = self._build_fixture(Path(directory))
            Path(config["paths"]["prepared"], "data.jsonl").write_text("modified\n")
            with patch("cot_mtkd.stage2.medoid.create_multi_adapter_model") as create_model:
                with self.assertRaisesRegex(RuntimeError, "content hash mismatch"):
                    build_stage2_medoid(config, context)
                create_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
