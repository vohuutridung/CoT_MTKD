from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..data.schema import PreparedRecord, TokenRegion
from ..models.chunked_head import (
    decoder_and_lm_head,
    forward_hidden,
    full_vocab_probe,
    gather_support_log_probabilities,
)
from ..models.multi_adapter import set_active_adapter
from ..stage1.dpp import marginal_log_uniqueness, normalized_support_features
from ..stage1.kneedle import build_union_support, capped_k_from_probe


@dataclass
class PredictiveSignals:
    competence: torch.Tensor
    agreement: torch.Tensor
    uniqueness: torch.Tensor
    js_disagreement: torch.Tensor
    mean_uncertainty: torch.Tensor
    relative_disagreement: torch.Tensor
    token_counts: torch.Tensor
    answer_competence: torch.Tensor
    answer_agreement: torch.Tensor
    medoid_kl_sum: torch.Tensor
    medoid_token_count: int


def _response_views(record: PreparedRecord, device: torch.device):
    labels = torch.tensor(record.labels[1:], device=device, dtype=torch.long)
    regions = torch.tensor(record.region_ids[1:], device=device, dtype=torch.long)
    steps = torch.tensor(record.step_ids[1:], device=device, dtype=torch.long)
    valid = labels.ne(-100)
    return labels[valid], regions[valid], steps[valid], valid


def _collect_teacher_hidden(
    model: torch.nn.Module,
    adapter_names: list[str],
    record: PreparedRecord,
    valid_mask: torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor]:
    input_ids = torch.tensor([record.input_ids], device=device, dtype=torch.long)
    attention = torch.ones_like(input_ids)
    hidden_by_expert: list[torch.Tensor] = []
    model.eval()
    for adapter_name in adapter_names:
        set_active_adapter(model, adapter_name)
        with torch.no_grad():
            output = forward_hidden(model, input_ids, attention, use_cache=False)
            hidden = output.last_hidden_state[:, :-1, :][0, valid_mask]
        hidden_by_expert.append(hidden.to("cpu"))
    return hidden_by_expert


def _stream_full_vocab_statistics(
    hidden_by_expert: list[torch.Tensor],
    head: torch.nn.Module,
    targets: torch.Tensor,
    chunk_tokens: int,
    feature_temperature: float,
    medoid_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    expert_count = len(hidden_by_expert)
    token_count = hidden_by_expert[0].shape[0]
    nll = torch.empty((expert_count, token_count), dtype=torch.float32)
    agreement_kl = torch.empty((expert_count, token_count), dtype=torch.float32)
    js = torch.empty(token_count, dtype=torch.float32)
    mean_entropy = torch.empty(token_count, dtype=torch.float32)
    medoid_sum = torch.zeros(expert_count, dtype=torch.float64)
    device = next(head.parameters()).device
    dtype = next(head.parameters()).dtype
    with torch.no_grad():
        for start in range(0, token_count, chunk_tokens):
            end = min(token_count, start + chunk_tokens)
            logits = [
                head(hidden[start:end].to(device=device, dtype=dtype)).float()
                for hidden in hidden_by_expert
            ]
            logp1 = [
                F.log_softmax(value / feature_temperature, dim=-1) for value in logits
            ]
            p1 = [value.exp() for value in logp1]
            target = targets[start:end].to(device)
            rows = torch.arange(end - start, device=device)
            for expert in range(expert_count):
                nll[expert, start:end] = -logp1[expert][rows, target].cpu()
            sum_probabilities = torch.stack(p1, dim=0).sum(dim=0)
            mixture = sum_probabilities / expert_count
            log_mixture = mixture.clamp_min(1.0e-30).log()
            js_chunk = (
                torch.stack(
                    [
                        (prob * (logprob - log_mixture)).sum(dim=-1)
                        for prob, logprob in zip(p1, logp1, strict=True)
                    ]
                )
                .mean(dim=0)
                .clamp_min(0.0)
            )
            js[start:end] = js_chunk.cpu()
            entropy = torch.stack(
                [
                    -(prob * logprob).sum(dim=-1)
                    for prob, logprob in zip(p1, logp1, strict=True)
                ]
            ).mean(dim=0)
            mean_entropy[start:end] = entropy.cpu()
            for expert in range(expert_count):
                leave_one_out = (sum_probabilities - p1[expert]) / (expert_count - 1)
                kl = (
                    (
                        p1[expert]
                        * (logp1[expert] - leave_one_out.clamp_min(1.0e-30).log())
                    )
                    .sum(dim=-1)
                    .clamp_min(0.0)
                )
                agreement_kl[expert, start:end] = kl.cpu()

            logp2 = [
                F.log_softmax(value / medoid_temperature, dim=-1) for value in logits
            ]
            p2 = [value.exp() for value in logp2]
            mixture2 = torch.stack(p2, dim=0).mean(dim=0)
            log_mixture2 = mixture2.clamp_min(1.0e-30).log()
            for expert in range(expert_count):
                medoid_sum[expert] += (
                    (mixture2 * (log_mixture2 - logp2[expert]))
                    .sum(dim=-1)
                    .clamp_min(0.0)
                    .sum()
                    .double()
                    .cpu()
                )
    return nll, agreement_kl, js, mean_entropy, medoid_sum


def _dpp_uniqueness(
    hidden_by_expert: list[torch.Tensor],
    head: torch.nn.Module,
    targets: torch.Tensor,
    step_ids: torch.Tensor,
    chunk_tokens: int,
    kneedle: dict[str, int],
    jitter: float,
) -> torch.Tensor:
    expert_count = len(hidden_by_expert)
    steps = sorted(set(int(value) for value in step_ids.tolist()))
    if not steps:
        return torch.empty((0, expert_count), dtype=torch.float32)
    values_by_expert, ids_by_expert, min_by_expert, max_by_expert = [], [], [], []
    for hidden in hidden_by_expert:
        values, ids, minimum, maximum = full_vocab_probe(
            hidden,
            head,
            targets.cpu(),
            int(kneedle["probe_k"]),
            chunk_tokens,
        )
        values_by_expert.append(values)
        ids_by_expert.append(ids)
        min_by_expert.append(minimum)
        max_by_expert.append(maximum)
    selected_k = torch.stack(
        [
            capped_k_from_probe(
                values,
                minimum,
                maximum,
                head.weight.shape[0],
                int(kneedle["min_k"]),
                int(kneedle["max_k"]),
            )
            for values, minimum, maximum in zip(
                values_by_expert, min_by_expert, max_by_expert, strict=True
            )
        ]
    )
    support, support_mask = build_union_support(torch.stack(ids_by_expert), selected_k)
    device = next(head.parameters()).device
    log_probabilities = torch.stack(
        [
            gather_support_log_probabilities(
                hidden, head, support, chunk_tokens, output_device=device
            )
            for hidden in hidden_by_expert
        ]
    )
    features = normalized_support_features(
        log_probabilities, support_mask.to(device)
    )
    uniqueness: list[torch.Tensor] = []
    step_ids_device = step_ids.to(device)
    for step in steps:
        mask = step_ids_device.eq(step)
        current = features[:, mask, :]
        gram = torch.einsum("mtk,ntk->mn", current, current) / mask.sum()
        uniqueness.append(marginal_log_uniqueness(gram, jitter).cpu())
    return torch.stack(uniqueness)


def score_predictive_signals(
    model: torch.nn.Module,
    adapter_names: list[str],
    record: PreparedRecord,
    device: torch.device,
    chunk_tokens: int,
    kneedle: dict[str, int],
    dpp_jitter: float,
    feature_temperature: float = 1.0,
    medoid_temperature: float = 2.0,
) -> PredictiveSignals:
    targets, regions, steps, valid = _response_views(record, device)
    hidden_by_expert = _collect_teacher_hidden(
        model, adapter_names, record, valid, device
    )
    _, head = decoder_and_lm_head(model)
    nll, agreement_kl, js, mean_entropy, medoid_sum = _stream_full_vocab_statistics(
        hidden_by_expert,
        head,
        targets,
        chunk_tokens,
        feature_temperature,
        medoid_temperature,
    )
    targets_cpu, regions_cpu, steps_cpu = targets.cpu(), regions.cpu(), steps.cpu()
    reasoning_mask = regions_cpu.eq(int(TokenRegion.REASONING))
    reasoning_steps = steps_cpu[reasoning_mask]
    reasoning_hidden = [hidden[reasoning_mask] for hidden in hidden_by_expert]
    reasoning_targets = targets_cpu[reasoning_mask]
    uniqueness = _dpp_uniqueness(
        reasoning_hidden,
        head,
        reasoning_targets,
        reasoning_steps,
        chunk_tokens,
        kneedle,
        dpp_jitter,
    )
    unique_steps = sorted(set(int(value) for value in reasoning_steps.tolist()))
    competence_rows: list[torch.Tensor] = []
    agreement_rows: list[torch.Tensor] = []
    js_rows: list[torch.Tensor] = []
    uncertainty_rows: list[torch.Tensor] = []
    token_counts: list[int] = []
    for step in unique_steps:
        mask = reasoning_mask & steps_cpu.eq(step)
        competence_rows.append(-nll[:, mask].mean(dim=-1))
        agreement_rows.append(-agreement_kl[:, mask].mean(dim=-1))
        js_rows.append(js[mask].mean())
        uncertainty_rows.append(mean_entropy[mask].mean())
        token_counts.append(int(mask.sum().item()))
    competence = (
        torch.stack(competence_rows)
        if competence_rows
        else torch.empty((0, len(adapter_names)))
    )
    agreement = (
        torch.stack(agreement_rows)
        if agreement_rows
        else torch.empty((0, len(adapter_names)))
    )
    answer_mask = regions_cpu.eq(int(TokenRegion.ANSWER))
    if not answer_mask.any():
        answer_mask = regions_cpu.eq(int(TokenRegion.ANSWER_MARKER)) | regions_cpu.eq(
            int(TokenRegion.EOS)
        )
    disagreement = torch.stack(js_rows) if js_rows else torch.empty(0)
    mean_uncertainty = (
        torch.stack(uncertainty_rows) if uncertainty_rows else torch.empty(0)
    )
    relative = torch.empty(0)
    if mean_uncertainty.numel():
        relative = disagreement.clamp_min(0.0) / mean_uncertainty.clamp_min(1.0e-6)
    return PredictiveSignals(
        competence=competence,
        agreement=agreement,
        uniqueness=uniqueness,
        js_disagreement=disagreement,
        mean_uncertainty=mean_uncertainty,
        relative_disagreement=relative,
        token_counts=torch.tensor(token_counts, dtype=torch.float32),
        answer_competence=-nll[:, answer_mask].mean(dim=-1),
        answer_agreement=-agreement_kl[:, answer_mask].mean(dim=-1),
        medoid_kl_sum=medoid_sum,
        medoid_token_count=int(targets.numel()),
    )
