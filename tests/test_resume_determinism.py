from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from cot_mtkd.models.multi_adapter import (
    adapter_parameter_groups,
    create_multi_adapter_model,
    extract_adapter_state,
    set_active_adapter,
    set_all_adapters_trainable,
)
from cot_mtkd.stage1.rbf import BandwidthEMA, effective_update_distances
from cot_mtkd.stage1.trainer import (
    _load_training_checkpoint,
    _optimizer_and_scheduler,
    _save_training_checkpoint,
)
from cot_mtkd.utils.seed import derived_seed, deterministic_rng
from cot_mtkd.utils.training import assign_gradients, interaction_scale

LORA = {
    "rank": 2,
    "alpha": 2,
    "dropout": 0.2,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}
TRAINING = {
    "stage1": {"learning_rate": 1.0e-3},
    "optimizer": {
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.0,
    },
    "scheduler": {"warmup_ratio": 0.25, "min_lr_ratio": 0.0},
}


def build_training_state():
    torch.manual_seed(123)
    base = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
        )
    )
    with patch("cot_mtkd.models.multi_adapter.load_base_causal_lm", return_value=base):
        model, names = create_multi_adapter_model({}, LORA, 2, torch.device("cpu"), 42)
    groups = adapter_parameter_groups(model, names)
    parameters = [list(group.values()) for group in groups]
    pairs = [_optimizer_and_scheduler(values, TRAINING, 4) for values in parameters]
    optimizers = [value[0] for value in pairs]
    schedulers = [value[1] for value in pairs]
    bandwidth = BandwidthEMA(decay=0.9, floor=1.0e-12)
    return model, names, groups, parameters, optimizers, schedulers, bandwidth


def run_step(state, step: int) -> tuple[list[float], float, float]:
    model, names, groups, parameters, optimizers, schedulers, bandwidth = state
    model.train()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    attention = torch.ones_like(input_ids)
    losses: list[float] = []
    for expert, (name, current, optimizer, scheduler) in enumerate(
        zip(names, parameters, optimizers, schedulers, strict=True)
    ):
        set_active_adapter(model, name)
        seed = derived_seed(42, "resume-test", step, expert)
        with deterministic_rng(seed, torch.device("cpu")):
            logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
            loss = logits.square().mean() + 0.01 * logits.mean()
        gradients = torch.autograd.grad(loss, current)
        assign_gradients(current, [value.detach().float() for value in gradients])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().item()))
    set_all_adapters_trainable(model, names)
    distances = effective_update_distances(groups, scaling=1.0).detach()
    current_bandwidth = bandwidth.update(distances)
    gamma = interaction_scale(step / 3.0, 0.1, 0.3)
    return losses, gamma, current_bandwidth


class ResumeDeterminismTest(unittest.TestCase):
    def test_interrupted_matches_uninterrupted(self) -> None:
        uninterrupted = build_training_state()
        run_step(uninterrupted, 0)
        run_step(uninterrupted, 1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            _save_training_checkpoint(
                checkpoint,
                uninterrupted[0],
                uninterrupted[1],
                uninterrupted[4],
                uninterrupted[5],
                uninterrupted[6],
                global_step=2,
                epoch=0,
                batch_in_epoch=2,
                run_fingerprint="resume-test",
                base_seed=42,
                world_size=1,
            )
            expected = [run_step(uninterrupted, step) for step in (2, 3)]

            resumed = build_training_state()
            cursor = _load_training_checkpoint(
                checkpoint,
                resumed[0],
                resumed[1],
                resumed[4],
                resumed[5],
                resumed[6],
                expected_run_fingerprint="resume-test",
            )
            self.assertEqual(cursor, (2, 0, 2))
            actual = [run_step(resumed, step) for step in (2, 3)]

        for expected_step, actual_step in zip(expected, actual, strict=True):
            self.assertTrue(
                torch.allclose(
                    torch.tensor(expected_step[0]),
                    torch.tensor(actual_step[0]),
                    atol=1.0e-8,
                    rtol=1.0e-7,
                )
            )
            self.assertEqual(expected_step[1], actual_step[1])
            self.assertAlmostEqual(expected_step[2], actual_step[2], places=12)
        for name in uninterrupted[1]:
            expected_state = extract_adapter_state(uninterrupted[0], name)
            actual_state = extract_adapter_state(resumed[0], name)
            for key in expected_state:
                self.assertTrue(
                    torch.equal(expected_state[key], actual_state[key]),
                    msg=f"Adapter mismatch after resume: {name}/{key}",
                )


if __name__ == "__main__":
    unittest.main()
