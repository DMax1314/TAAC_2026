from __future__ import annotations

import math

import torch
from torch import nn

from ..config import ModelConfig
from .common import masked_mean


class CandidateAwareBaseline(nn.Module):
    def __init__(self, config: ModelConfig, dense_dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len + 1, config.embedding_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(1, config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.query_projection = nn.Sequential(
            nn.Linear(config.embedding_dim * 2 + config.hidden_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(),
        )
        fusion_dim = config.embedding_dim * 5 + config.hidden_dim
        self.output = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        candidate_embeddings = self.embedding(batch["candidate_tokens"])
        context_embeddings = self.embedding(batch["context_tokens"])
        history_embeddings = self.embedding(batch["history_tokens"])

        candidate_summary = masked_mean(candidate_embeddings, batch["candidate_mask"])
        context_summary = masked_mean(context_embeddings, batch["context_mask"])

        positions = torch.arange(
            batch["history_tokens"].size(1),
            device=batch["history_tokens"].device,
        ).unsqueeze(0)
        history_embeddings = history_embeddings + self.position_embedding(positions)
        history_embeddings = history_embeddings + self.time_projection(batch["history_time_gaps"].unsqueeze(-1))

        dense_summary = self.dense_projection(batch["dense_features"])
        query = self.query_projection(torch.cat([candidate_summary, context_summary, dense_summary], dim=-1))

        attention_scores = (history_embeddings * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(query.size(-1))
        attention_scores = attention_scores.masked_fill(~batch["history_mask"], -1e9)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights * batch["history_mask"].float()
        normalization = attention_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attention_weights = attention_weights / normalization

        history_summary = (history_embeddings * attention_weights.unsqueeze(-1)).sum(dim=1)
        interaction = candidate_summary * history_summary
        difference = torch.abs(candidate_summary - history_summary)

        fused = torch.cat(
            [
                candidate_summary,
                context_summary,
                history_summary,
                interaction,
                difference,
                dense_summary,
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


__all__ = ["CandidateAwareBaseline"]