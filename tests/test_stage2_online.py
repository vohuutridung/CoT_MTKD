from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from cot_mtkd.data.schema import PreparedRecord, TokenRegion
from cot_mtkd.models.multi_adapter import (
    adapter_parameter_map,
    create_multi_adapter_model,
    extract_adapter_state,
    load_adapter_state,
    lora_config,
    set_active_adapter,
)
from cot_mtkd.stage2.geometry import GeometryWeights, task_anchored_weights
from cot_mtkd.stage2.online import compute_record_gradient, plan_record


class CharacterTokenizer:
    name_or_path = "fixture/model"

    def __init__(self):
        self.special_tokens_map = {}

    def __len__(self):
        return 64

    def __call__(self, text: str, **kwargs):
        return {
            "input_ids": [1 + ord(character) % 61 for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def two_step_record() -> PreparedRecord:
    regions = [
        TokenRegion.PROMPT,
        TokenRegion.PROMPT,
        TokenRegion.ASSISTANT_CONTROL,
        TokenRegion.REASONING,
        TokenRegion.REASONING,
        TokenRegion.DELIMITER,
        TokenRegion.REASONING,
        TokenRegion.REASONING,
        TokenRegion.REASONING,
        TokenRegion.DELIMITER,
        TokenRegion.ANSWER_MARKER,
        TokenRegion.ANSWER,
        TokenRegion.EOS,
    ]
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    return PreparedRecord(
        sample_id="tiny-two-steps",
        input_ids=ids,
        labels=[-100, -100] + ids[2:],
        attention_mask=[1] * len(ids),
        offset_mapping=[(i, i + 1) for i in range(len(ids))],
        region_ids=[int(value) for value in regions],
        step_ids=[-1, -1, -1, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1],
        question="question",
        thinking="step one\n\nstep two",
        attempt="wrong attempt",
        solution="42",
        deepseek_grade="No",
        original_length=len(ids),
        kept_length=len(ids),
        original_steps=2,
        kept_steps=2,
        truncated=False,
        answer_start=10,
        reasoning_start=3,
        tokenizer_fingerprint="character-fixture",
    )


LORA = {
    "rank": 2,
    "alpha": 2,
    "dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def tiny_online_council(checkpointing: bool = True):
    torch.manual_seed(17)
    base = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            attention_dropout=0.0,
        )
    )
    if checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
    with patch("cot_mtkd.models.multi_adapter.load_base_causal_lm", return_value=base):
        model, names = create_multi_adapter_model({}, LORA, 3, torch.device("cpu"), 42)
    model.add_adapter("student", lora_config(LORA))
    generator = torch.Generator().manual_seed(703)
    with torch.no_grad():
        for teacher_name in names:
            for key, parameter in adapter_parameter_map(model, teacher_name).items():
                if "lora_B" in key:
                    parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.13)
        load_adapter_state(model, "student", extract_adapter_state(model, names[0]))
    set_active_adapter(model, "student")
    return model, names, list(adapter_parameter_map(model, "student").values())


def dense_record_reference(model, names, parameters, record, tokenizer, config):
    """Independent full-model implementation of proposal equations (2)--(16)."""
    plan = plan_record(record, tokenizer, config["stage2"]["max_length"])
    ids = torch.tensor([plan.input_ids])
    positions = [position for step in plan.step_positions for position in step]
    selected = torch.tensor(positions) - 1
    temperature = config["geometry"]["temperature"]
    teacher_logits = []
    for name in names:
        model.set_adapter(name)
        model.requires_grad_(False)
        model.eval()
        with torch.no_grad():
            teacher_logits.append(model(input_ids=ids, use_cache=False).logits[0, selected].float())
    set_active_adapter(model, "student")
    model.train()
    student_logits = model(input_ids=ids, use_cache=False).logits[0, selected].float()
    student_logp = F.log_softmax(student_logits / temperature, dim=-1)
    total = student_logits.sum() * 0.0
    step_weights, anchors, agreement_values, consensus_values = [], [], [], []
    cursor = 0
    for step_positions, prefix_end in zip(plan.step_positions, plan.prefix_ends, strict=True):
        end = cursor + len(step_positions)
        logps = torch.stack(
            [F.log_softmax(value[cursor:end] / temperature, dim=-1) for value in teacher_logits]
        )
        probs = logps.exp()
        kd_gradients = []
        for teacher in range(len(names)):
            loss = (
                temperature**2
                * (probs[teacher] * (logps[teacher] - student_logp[cursor:end])).sum(dim=-1).mean()
            )
            gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
            kd_gradients.append(
                torch.cat([value.detach().double().reshape(-1) for value in gradients])
            )
        anchor_ids = torch.tensor(
            [plan.input_ids[:prefix_end] + plan.answer_prefix + plan.solution]
        )
        answer_start = prefix_end + len(plan.answer_prefix)
        full_logits = model(input_ids=anchor_ids, use_cache=False).logits[0].float()
        anchor_loss = F.cross_entropy(
            full_logits[answer_start - 1 : answer_start + len(plan.solution) - 1],
            torch.tensor(plan.solution),
        )
        anchor_blocks = torch.autograd.grad(anchor_loss, parameters)
        anchor = torch.cat([value.detach().double().reshape(-1) for value in anchor_blocks])
        anchors.append(anchor)
        gradients = torch.stack(kd_gradients)
        mean = gradients.mean(dim=0)
        common_energy = (len(names) - 1) * mean.square().sum()
        variance = (gradients - mean).square().sum(dim=1).mean()
        agreement = common_energy / (common_energy + variance + config["geometry"]["epsilon_a"])
        utilities = (gradients @ anchor) / (
            torch.linalg.vector_norm(gradients, dim=1) * torch.linalg.vector_norm(anchor)
            + config["geometry"]["epsilon_u"]
        )
        mean_utility = (mean @ anchor) / (
            torch.linalg.vector_norm(mean) * torch.linalg.vector_norm(anchor)
            + config["geometry"]["epsilon_u"]
        )
        positive = utilities.clamp_min(0.0)
        weight = positive.mean().item()
        consensus = (agreement * mean_utility.clamp_min(0.0)).item()
        step_weights.append(weight)
        agreement_values.append(agreement.item())
        consensus_values.append(consensus)
        if weight > 0.0:
            proportions = (positive / positive.sum()).float()
            coverage = (proportions[:, None, None] * probs).sum(dim=0)
            consensus_target = F.softmax((proportions[:, None, None] * logps).sum(dim=0), dim=-1)
            target = (consensus * consensus_target + (1.0 - consensus) * coverage).detach()
            kd = (
                temperature**2
                * (target * (target.log() - student_logp[cursor:end])).sum(dim=-1).mean()
            )
            total = total + weight / plan.num_steps * kd
        cursor = end
    final = torch.autograd.grad(total, parameters)
    return {
        "gradients": [value.detach().float() for value in final],
        "loss": total.item(),
        "weights": step_weights,
        "anchors": anchors,
        "agreement_sum": sum(agreement_values),
        "consensus_sum": sum(consensus_values),
    }


class Stage2OnlineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread_count = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.thread_count)

    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.record = two_step_record()
        self.config = {
            "stage2": {"max_length": 128},
            "runtime": {"lm_head_chunk_tokens": 2},
            "geometry": {"temperature": 2.0, "epsilon_a": 1.0e-12, "epsilon_u": 1.0e-12},
        }

    def test_online_parameter_gradients_and_full_prefix_anchors_match_dense_reference(self):
        model, names, parameters = tiny_online_council(checkpointing=True)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        expected = dense_record_reference(
            model, names, parameters, self.record, self.tokenizer, self.config
        )
        actual_anchors = []

        def capture_geometry(teacher_gradients, anchor_gradients, epsilon_a, epsilon_u):
            actual_anchors.append(
                torch.cat([value.detach().double().reshape(-1) for value in anchor_gradients])
            )
            return task_anchored_weights(teacher_gradients, anchor_gradients, epsilon_a, epsilon_u)

        with patch("cot_mtkd.stage2.online.task_anchored_weights", side_effect=capture_geometry):
            actual = compute_record_gradient(
                model,
                names,
                parameters,
                self.record,
                self.tokenizer,
                self.config,
                torch.device("cpu"),
            )
        self.assertEqual(actual.steps, 2)
        self.assertGreater(actual.active_steps, 0, "Fixture must exercise the final KD VJP")
        self.assertEqual(actual.active_steps, sum(value > 0.0 for value in expected["weights"]))
        self.assertAlmostEqual(actual.loss, expected["loss"], delta=2.0e-7)
        self.assertAlmostEqual(actual.metrics["weight_sum"], sum(expected["weights"]), delta=1.0e-5)
        self.assertAlmostEqual(
            actual.metrics["agreement_sum"], expected["agreement_sum"], delta=1.0e-5
        )
        self.assertAlmostEqual(
            actual.metrics["consensus_sum"], expected["consensus_sum"], delta=1.0e-5
        )
        for anchor, expected_anchor in zip(actual_anchors, expected["anchors"], strict=True):
            torch.testing.assert_close(anchor, expected_anchor, atol=2.0e-7, rtol=2.0e-5)
        for gradient, expected_gradient in zip(
            actual.gradients, expected["gradients"], strict=True
        ):
            torch.testing.assert_close(gradient, expected_gradient, atol=2.0e-7, rtol=2.0e-4)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], atol=0.0, rtol=0.0)
            self.assertIsNone(parameter.grad)
            self.assertEqual(parameter.requires_grad, "lora_" in name and ".student." in name)

    def test_inactive_step_retains_fixed_K_normalizer(self):
        model, names, parameters = tiny_online_council(checkpointing=False)
        selected = GeometryWeights(0.7, [0.5, 0.5, 0.5], 0.5, 0.5, torch.ones(3) / 3, 0.3)
        inactive = GeometryWeights(0.0, [-1.0] * 3, -1.0, 0.0, torch.zeros(3), 0.0)
        with patch(
            "cot_mtkd.stage2.online.task_anchored_weights", side_effect=[selected, inactive]
        ):
            two_steps = compute_record_gradient(
                model,
                names,
                parameters,
                self.record,
                self.tokenizer,
                self.config,
                torch.device("cpu"),
            )
        suffix_length = len(self.tokenizer("\n<|im_start|>answer\nAnswer: ")["input_ids"]) + 2
        first_only = {**self.config, "stage2": {"max_length": 6 + suffix_length}}
        with patch("cot_mtkd.stage2.online.task_anchored_weights", return_value=selected):
            one_step = compute_record_gradient(
                model,
                names,
                parameters,
                self.record,
                self.tokenizer,
                first_only,
                torch.device("cpu"),
            )
        self.assertEqual((two_steps.steps, two_steps.active_steps), (2, 1))
        self.assertEqual(
            (one_step.steps, one_step.active_steps, one_step.discarded_steps), (1, 1, 1)
        )
        self.assertAlmostEqual(two_steps.loss * 2, one_step.loss, delta=1.0e-7)
        for two, one in zip(two_steps.gradients, one_step.gradients, strict=True):
            torch.testing.assert_close(two * 2, one, atol=2.0e-7, rtol=2.0e-4)

    def test_planner_reserves_full_solution_and_excludes_delimiters_from_KD(self):
        self.record.solution = "complete gold solution"
        suffix_length = len(
            self.tokenizer("\n<|im_start|>answer\nAnswer: " + self.record.solution)["input_ids"]
        )
        plan = plan_record(self.record, self.tokenizer, 6 + suffix_length)
        self.assertEqual(plan.input_ids, self.record.input_ids[:6])
        self.assertEqual(plan.step_positions, [[3, 4]])
        self.assertEqual(plan.prefix_ends, [6])
        self.assertEqual(plan.discarded_steps, 1)
        self.assertEqual(plan.solution, self.tokenizer(self.record.solution)["input_ids"])
        self.assertEqual(plan.max_anchor_length, 6 + suffix_length)
        zero = plan_record(self.record, self.tokenizer, 3 + suffix_length)
        self.assertEqual(zero.num_steps, 0)
        self.assertEqual(zero.discarded_steps, 2)
        with self.assertRaisesRegex(ValueError, "gold solutions are never truncated"):
            plan_record(self.record, self.tokenizer, 2 + suffix_length)


if __name__ == "__main__":
    unittest.main()
