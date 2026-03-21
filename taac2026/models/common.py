from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..utils import masked_mean


def masked_attention_pool(values: torch.Tensor, mask: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    if values.size(1) == 0:
        return torch.zeros_like(query)

    scores = (values * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(query.size(-1))
    scores = scores.masked_fill(~mask, -1e9)
    weights = torch.softmax(scores, dim=-1)
    weights = weights * mask.float()
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return (values * weights.unsqueeze(-1)).sum(dim=1)


def build_decomposed_history_embeddings(
    token_embedding: nn.Embedding,
    position_embedding: nn.Embedding,
    sequence_group_embedding: nn.Embedding,
    time_projection: nn.Module,
    component_projection: nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    history_embeddings = token_embedding(batch["history_tokens"])
    component_embeddings = token_embedding(batch["history_component_tokens"])
    component_weights = batch["history_component_mask"].unsqueeze(-1).float()
    component_summary = (component_embeddings * component_weights).sum(dim=2)
    component_summary = component_summary / component_weights.sum(dim=2).clamp_min(1.0)

    positions = torch.arange(
        batch["history_tokens"].size(1),
        device=batch["history_tokens"].device,
    ).unsqueeze(0)
    history_embeddings = history_embeddings + position_embedding(positions)
    history_embeddings = history_embeddings + sequence_group_embedding(batch["history_group_ids"])
    history_embeddings = history_embeddings + time_projection(batch["history_time_gaps"].unsqueeze(-1))
    history_embeddings = component_projection(torch.cat([history_embeddings, component_summary], dim=-1))
    return history_embeddings, component_summary


def build_pooled_memory(
    history_embeddings: torch.Tensor,
    history_mask: torch.Tensor,
    recent_seq_len: int,
    memory_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    far_history = history_embeddings[:, recent_seq_len:]
    far_mask = history_mask[:, recent_seq_len:]
    if far_history.size(1) == 0:
        return history_embeddings[:, :0], history_mask[:, :0]

    masked_far_history = far_history * far_mask.unsqueeze(-1).float()
    pooled_values = F.adaptive_avg_pool1d(masked_far_history.transpose(1, 2), memory_slots).transpose(1, 2)
    pooled_mask = F.adaptive_avg_pool1d(far_mask.float().unsqueeze(1), memory_slots).squeeze(1)
    memory_embeddings = pooled_values / pooled_mask.unsqueeze(-1).clamp_min(1e-6)
    memory_mask = pooled_mask > 0
    return memory_embeddings, memory_mask


class DINActivationUnit(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if keys.size(1) == 0:
            return torch.zeros_like(query)

        query_expanded = query.unsqueeze(1).expand(-1, keys.size(1), -1)
        attention_input = torch.cat(
            [query_expanded, keys, query_expanded - keys, query_expanded * keys],
            dim=-1,
        )
        attention_scores = self.dnn(attention_input).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~mask, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights * mask.float()
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)


__all__ = [
    "DINActivationUnit",
    "build_decomposed_history_embeddings",
    "build_pooled_memory",
    "masked_attention_pool",
    "masked_mean",
]