"""PCVRDualQ — dualq HyFormer variant on the unified PCVR contract.

Source: https://github.com/zzhlkw-ai/TAAC2026 (MIT License), ``round2best``
variant (ported as ``dualq``; TAAC2026 academic track Round-2 Top-17,
Unified Module Innovation Award). Shared building blocks live in
``hyformer.py``; this module keeps the top-level model class only.

Model structure ported from the source:

* Multi-view sequence expansion: ``seq_interest_ratios`` truncates each
  domain into additional ratio views, or (``use_time_aligned_interleave``)
  merges all domains into a single time-sorted "interleave" sequence.
* DualQ split 4 user + 2 item query tokens.
* Pair cross tokenizer learns per-feature dense-weighted pooling residuals
  (signed softmax weights for fids 89-91, L1-normalised weights otherwise);
  pair fids are diverted from the user side at construction from the schema.
* Item dense features are tokenized per schema field (``ItemDenseTokenizer``)
  with the fid-129 body/stat split.
* User dense tail features (fids 118/120/121/123/130/131/132, when present)
  are folded into two semantic group tokens.

Not ported (documented deltas, see ``docs/experiments/dualq.md``): the
train-set profile driven data features (id-frequency buckets, item dense stat
buckets, item time dense features) and item OOF CTR/CVR encodings are
dataset-side in the source repository and are not part of this port. The
``use_item_time_query_token`` switch is unsupported because it depends on
those profile-derived synthetic columns.

The model reads only ``PCVRModelInput``; every time feature (buckets, gaps,
per-position floats, per-domain stats, global time) is derived inside this
model from raw event timestamps.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from taac2026.api import (
    NUM_TIME_BUCKETS,
    EmbeddingParameterMixin,
    PCVRModelInput,
    PCVRSchema,
    build_pcvr_model_specs,
    compute_sequence_time_buckets,
    maybe_gradient_checkpoint,
)

from .config import DualQModelConfig
from .hyformer import (
    CrossRankMixerNSTokenizer,
    DualQGenerator,
    GroupNSTokenizer,
    ItemDenseTokenizer,
    MultiSeqHyFormerBlock,
    RankMixerNSTokenizer,
    RotaryEmbedding,
    TS_FLOAT_DIM,
    TS_STAT_DIM,
    TimeTokenBuilder,
)
from .layers import (
    build_global_time_features,
    compile_pair_split,
    compute_sequence_gap_buckets,
    compute_sequence_ts_float,
    compute_sequence_ts_stat,
    parse_pair_feature_fids,
)

logger = logging.getLogger(__name__)

# Semantic user dense groups folded into extra NS tokens (source dataset
# layout; only entries present in the schema are materialised).
_USER_DENSE_GROUPS = {
    "g_stat": [118, 120, 121],  # quantiles + ratios + counts
    "g_float": [123, 130, 131, 132],  # float values + mixed
}


def _remap_user_groups(
    ns_groups: dict[str, list[int]],
    pair_fids: set[int],
    fid_to_index: dict[int, int],
) -> list[list[int]]:
    """Map declared fid groups onto the reduced (pair-diverted) layout."""
    groups: list[list[int]] = []
    for group_name, fids in ns_groups.items():
        indices = []
        for fid in fids:
            if fid in pair_fids:
                raise ValueError(
                    f"user group {group_name!r} contains pair fid {fid}; "
                    "pair fids are diverted to the cross pair tokenizer"
                )
            if fid not in fid_to_index:
                raise ValueError(f"user group {group_name!r} contains unknown fid {fid}")
            indices.append(fid_to_index[fid])
        groups.append(indices)
    return groups


class PCVRDualQ(EmbeddingParameterMixin, nn.Module):
    """End-to-end PCVR ranking model (dualq variant)."""

    def __init__(
        self,
        schema: PCVRSchema,
        config: DualQModelConfig,
    ) -> None:
        super().__init__()
        specs = build_pcvr_model_specs(schema, config.ns)
        pair_fids = set(parse_pair_feature_fids(config.pair_feature_fids))
        self._pair_plan = compile_pair_split(schema, sorted(pair_fids))

        # ---- core dimensions ----
        self.d_model = config.d_model
        self.emb_dim = config.emb_dim
        self.action_num = config.action_num

        # ---- sequence-side bookkeeping (sorted → deterministic order) ----
        self.raw_seq_domains = sorted(specs.seq_vocab_sizes.keys())
        self.seq_interest_ratios = config.seq_interest_ratio_list
        self.use_time_aligned_interleave = bool(config.use_time_aligned_interleave)
        if self.use_time_aligned_interleave:
            self.seq_domains = ["interleave"]
            self._seq_domain_ratios = {
                "interleave": ("interleave", self.seq_interest_ratios[0])
            }
        elif len(self.seq_interest_ratios) > 1:
            self.seq_domains = [
                f"{domain}_r{ratio}"
                for domain in self.raw_seq_domains
                for ratio in self.seq_interest_ratios
            ]
            self._seq_domain_ratios = {
                f"{domain}_r{ratio}": (domain, ratio)
                for domain in self.raw_seq_domains
                for ratio in self.seq_interest_ratios
            }
        else:
            self.seq_domains = list(self.raw_seq_domains)
            self._seq_domain_ratios = {
                domain: (domain, 1.0) for domain in self.raw_seq_domains
            }
        self.num_sequences = len(self.seq_domains)

        # ---- DualQ split: per-side query tokens ----
        self.user_q_tokens = int(config.user_q_tokens)
        self.item_q_tokens = int(config.item_q_tokens)
        self.num_queries = int(config.num_queries)
        if self.num_queries != self.user_q_tokens + self.item_q_tokens:
            raise ValueError(
                "DualQ requires "
                f"num_queries == user_q_tokens + item_q_tokens == "
                f"{self.user_q_tokens + self.item_q_tokens}"
            )

        # ---- time-bucket / RoPE switches ----
        self.num_time_buckets = NUM_TIME_BUCKETS if config.use_time_buckets else 0
        self.use_rope = config.use_rope
        self.use_seq_gap_buckets = bool(config.use_seq_gap_buckets) and self.num_time_buckets > 0
        self.use_time_gap_domain_gates = bool(config.use_time_gap_domain_gates)
        self.use_global_time_token = bool(config.use_global_time_token)

        # ---- NS / embedding policy ----
        self.rank_mixer_mode = config.rank_mixer_mode
        self.ns_tokenizer_type = config.ns.tokenizer_type
        self.emb_skip_threshold = int(config.emb_skip_threshold)
        self.seq_id_threshold = int(config.seq_id_threshold)
        self.gradient_checkpointing = bool(config.gradient_checkpointing)
        self.use_fid87_token_residual = bool(config.use_fid87_token_residual)
        self.use_time_decay_summary = bool(config.use_time_decay_summary)

        # ---- user int side: pair fids diverted into the cross tokenizer ----
        schema_fid_to_vocab = {column.fid: column.vocab_size for column in schema.user_int}
        user_int_feature_specs = [
            (schema_fid_to_vocab[fid], offset, length)
            for fid, offset, length in self._pair_plan.user_int_reduced
        ]
        pair_int_feature_specs = [
            (schema_fid_to_vocab[fid], offset, length)
            for fid, offset, length in self._pair_plan.pair_int_reduced
        ]
        user_fid_to_index = {
            fid: index
            for index, (fid, _, _) in enumerate(self._pair_plan.user_int_reduced)
        }
        if config.ns.grouping_strategy == "singleton":
            user_ns_groups = [[index] for index in range(len(user_int_feature_specs))]
        else:
            user_ns_groups = _remap_user_groups(
                config.ns.user_groups, pair_fids, user_fid_to_index
            )

        # ---- NS Tokens Construction ----
        if self.ns_tokenizer_type == "group":
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=config.emb_dim,
                d_model=config.d_model,
                emb_skip_threshold=self.emb_skip_threshold,
            )
            num_user_ns = len(config.ns.user_groups)
            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=specs.item_int_feature_specs,
                groups=specs.item_ns_groups,
                emb_dim=config.emb_dim,
                d_model=config.d_model,
                emb_skip_threshold=self.emb_skip_threshold,
            )
            num_item_ns = len(specs.item_ns_groups)
        elif self.ns_tokenizer_type == "rankmixer":
            user_ns_tokens = int(config.ns.user_tokens)
            item_ns_tokens = int(config.ns.item_tokens)
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(specs.item_ns_groups)

            # Pair feature embedder with dense-weighted pooling residuals.
            self.cross_ns_tokenizer = None
            if pair_int_feature_specs:
                self.cross_ns_tokenizer = CrossRankMixerNSTokenizer(
                    pair_int_feature_specs,
                    self.d_model,
                    emb_dim=config.emb_dim,
                    feature_fids=[fid for fid, _, _ in self._pair_plan.pair_int_reduced],
                    use_weighted_residual=True,
                )
                pair_emb_dim = self.cross_ns_tokenizer.out_dim
            else:
                pair_emb_dim = 0

            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=config.emb_dim,
                d_model=config.d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=self.emb_skip_threshold,
                extra_emb_dim=pair_emb_dim,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=specs.item_int_feature_specs,
                groups=specs.item_ns_groups,
                emb_dim=config.emb_dim,
                d_model=config.d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=self.emb_skip_threshold,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {self.ns_tokenizer_type}")

        # ---- user dense: fid=61 embedding | fid=87 history pool | group tokens ----
        user_dense_entries = self._pair_plan.user_dense_reduced
        self.has_user_dense = len(user_dense_entries) > 0
        user_dense_dim = sum(dim for _, _, dim in user_dense_entries)
        self._user_emb_dim = int(config.user_emb_dim)
        self._user_seq_block_dim = int(config.user_seq_block_dim)
        self._user_seq_num = int(config.user_seq_num)
        entry_by_fid = {fid: (offset, dim) for fid, offset, dim in user_dense_entries}
        self._user_dense_split = bool(
            self.has_user_dense
            and user_dense_dim >= self._user_emb_dim + self._user_seq_block_dim * self._user_seq_num
        )
        self._dense_group_entries: dict[str, list[tuple[int, int]]] = {}
        if self._user_dense_split:
            self._user_emb_entry = entry_by_fid.get(61)
            self._user_seq_entry = entry_by_fid.get(87)
            if self._user_emb_entry is None or self._user_emb_entry[1] != self._user_emb_dim:
                raise ValueError(
                    "user dense split requires fid=61 with dim "
                    f"{self._user_emb_dim}, got {self._user_emb_entry}"
                )
            if self._user_seq_entry is None or self._user_seq_entry[1] != (
                self._user_seq_block_dim * self._user_seq_num
            ):
                raise ValueError(
                    "user dense split requires fid=87 with dim "
                    f"{self._user_seq_block_dim * self._user_seq_num}, "
                    f"got {self._user_seq_entry}"
                )

            self.user_emb_proj = nn.Sequential(
                nn.Linear(self._user_emb_dim, self.d_model),
                nn.LayerNorm(self.d_model),
            )
            self.user_seq_attn = nn.Linear(self._user_seq_block_dim, 1)
            self.user_seq_proj = nn.Sequential(
                nn.Linear(self._user_seq_block_dim, self.d_model),
                nn.LayerNorm(self.d_model),
            )

            for group_name, fids in _USER_DENSE_GROUPS.items():
                group_entries = [
                    (offset, dim)
                    for fid, (offset, dim) in entry_by_fid.items()
                    if fid in fids and fid not in (61, 87)
                ]
                if group_entries:
                    self._dense_group_entries[group_name] = group_entries
            self._dense_group_projs = nn.ModuleDict()
            self._dense_group_names = list(self._dense_group_entries.keys())
            for group_name, entries in self._dense_group_entries.items():
                total_dim = sum(dim for _, dim in entries)
                self._dense_group_projs[group_name] = nn.Sequential(
                    nn.LayerNorm(total_dim) if total_dim > 1 else nn.Identity(),
                    nn.Linear(total_dim, self.d_model),
                    nn.SiLU(),
                    nn.LayerNorm(self.d_model),
                )
            num_user_dense_tokens = 2 + len(self._dense_group_names)
        elif self.has_user_dense:
            self._user_emb_entry = None
            self._user_seq_entry = None
            self.user_dense_proj = nn.Sequential(
                nn.Linear(user_dense_dim, self.d_model),
                nn.LayerNorm(self.d_model),
            )
            num_user_dense_tokens = 1
        else:
            self._user_emb_entry = None
            self._user_seq_entry = None
            num_user_dense_tokens = 0

        # ---- item dense: per-field tokens (fid-129 body/stat split) ----
        item_dense_entries: list[tuple[int, int, int]] = []
        offset = 0
        for column in schema.item_dense:
            item_dense_entries.append((column.fid, offset, column.dim))
            offset += column.dim
        self.has_item_dense = len(item_dense_entries) > 0
        if self.has_item_dense:
            self.item_dense_tokenizer = ItemDenseTokenizer(item_dense_entries, self.d_model)
            self.num_item_dense_tokens = self.item_dense_tokenizer.num_tokens
        else:
            self.num_item_dense_tokens = 0

        # ---- global time token: hour / dow / weekend from request timestamp ----
        self.global_time_vocab_sizes = [24, 7, 2]
        if self.use_global_time_token:
            self.global_time_embs = nn.ModuleList(
                [
                    nn.Embedding(vs + 1, self.emb_dim, padding_idx=0, sparse=True)
                    for vs in self.global_time_vocab_sizes
                ]
            )
            self.global_time_proj = nn.Sequential(
                nn.Linear(len(self.global_time_vocab_sizes) * self.emb_dim, self.d_model),
                nn.LayerNorm(self.d_model),
            )

        # Total NS token count (cross pair tokens fold into user_ns, not a row).
        self.num_ns = (
            num_user_ns
            + num_user_dense_tokens
            + (1 if self.use_global_time_token else 0)
            + num_item_ns
            + self.num_item_dense_tokens
        )

        # ================== Check d_model % T == 0 constraint (full mode only) ==================
        T = self.num_queries * self.num_sequences
        if self.rank_mixer_mode == "full" and self.d_model % T != 0:
            valid_T_values = [t for t in range(1, self.d_model + 1) if self.d_model % t == 0]
            raise ValueError(
                f"d_model={self.d_model} must be divisible by T=num_queries*num_sequences="
                f"{self.num_queries}*{self.num_sequences}={T}. "
                f"Valid T values for d_model={self.d_model}: {valid_T_values}"
            )

        # ================== Seq Tokens Embedding ==================
        # seq_id_threshold decides which features inside the seq tokenizer are
        # treated as id features (they receive extra dropout). It is fully
        # independent of emb_skip_threshold (which skips Embedding creation).
        self.seq_id_emb_dropout = nn.Dropout(config.dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            """Build a per-feature ``nn.Embedding`` ladder for one seq domain."""
            kept_embs: list[nn.Embedding | None] = []
            for vs in vocab_sizes:
                vsz = int(vs)
                drop_this = vsz <= 0 or (
                    self.emb_skip_threshold > 0 and vsz > self.emb_skip_threshold
                )
                if drop_this:
                    kept_embs.append(None)
                    continue
                kept_embs.append(nn.Embedding(vsz + 1, self.emb_dim, padding_idx=0, sparse=True))

            module_list = nn.ModuleList([e for e in kept_embs if e is not None])
            index_map: list[int] = []
            cursor = 0
            for slot in kept_embs:
                if slot is None:
                    index_map.append(-1)
                else:
                    index_map.append(cursor)
                    cursor += 1
            is_id = [int(vs) > self.seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        # ================== Dynamic Sequence Embeddings ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_proj = nn.ModuleDict()
        self._seq_ts_float_proj = nn.ModuleDict()
        self._seq_emb_index: dict[str, list[int]] = {}
        self._seq_is_id: dict[str, list[bool]] = {}
        self._seq_vocab_sizes: dict[str, list[int]] = {}

        for domain in self.raw_seq_domains:
            domain_vocabs = specs.seq_vocab_sizes[domain]
            embs, idx_map, is_id_flags = _make_seq_embs(domain_vocabs)

            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id_flags
            self._seq_vocab_sizes[domain] = domain_vocabs

            self._seq_ts_float_proj[domain] = nn.Sequential(
                nn.Linear(TS_FLOAT_DIM, self.emb_dim),
                nn.LayerNorm(self.emb_dim),
            )
            # Main projection consumes (len(vocabs)+1) cat blocks: id-side
            # embeddings plus the float-feat block.
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear((len(domain_vocabs) + 1) * self.emb_dim, self.d_model),
                nn.LayerNorm(self.d_model),
            )

        # ================== Time Interval Bucket Embedding (optional) ==================
        if self.num_time_buckets > 0:
            self.time_embedding = nn.Embedding(self.num_time_buckets, self.d_model, padding_idx=0, sparse=True)
        if self.use_seq_gap_buckets:
            self.gap_embedding = nn.Embedding(self.num_time_buckets, self.d_model, padding_idx=0, sparse=True)
        if self.use_time_gap_domain_gates:
            self.seq_time_gates = nn.ParameterDict(
                {domain: nn.Parameter(torch.ones(1)) for domain in self.raw_seq_domains}
            )
            self.seq_gap_gates = nn.ParameterDict(
                {domain: nn.Parameter(torch.ones(1)) for domain in self.raw_seq_domains}
            )

        # ================== HyFormer Components ==================
        self.time_token_builder = TimeTokenBuilder(
            num_sequences=self.num_sequences,
            d_model=self.d_model,
            hod_dim=1,
            ts_stat_dim=TS_STAT_DIM,
        )

        self.query_generator = DualQGenerator(
            d_model=self.d_model,
            num_heads=config.num_heads,
            num_sequences=self.num_sequences,
            ts_stat_dim=TS_STAT_DIM,
            dropout=config.dropout_rate,
            user_q_tokens=self.user_q_tokens,
            item_q_tokens=self.item_q_tokens,
        )

        self.blocks = nn.ModuleList(
            [
                MultiSeqHyFormerBlock(
                    d_model=self.d_model,
                    num_heads=config.num_heads,
                    num_queries=self.num_queries,
                    num_ns=self.num_ns,
                    num_sequences=self.num_sequences,
                    seq_encoder_type=config.seq_encoder_type,
                    hidden_mult=config.hidden_mult,
                    dropout=config.dropout_rate,
                    top_k=config.seq_top_k,
                    causal=config.seq_causal,
                    rank_mixer_mode=self.rank_mixer_mode,
                )
                for _ in range(config.num_blocks)
            ]
        )

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                dim=self.d_model // config.num_heads, base=config.rope_base
            )
        else:
            self.rotary_emb = None

        # ================== Output head ==================
        self.output_proj = nn.Sequential(
            nn.Linear(self.num_queries * self.num_sequences * self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.emb_dropout = nn.Dropout(config.dropout_rate)
        self.clsfier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.SiLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.d_model, self.action_num),
        )

        # ================== Time-decay summary fusion (optional) ==================
        if self.use_time_decay_summary:
            self.tds_alpha = nn.Parameter(torch.full((self.num_sequences,), 0.5))
            self.tds_beta = nn.Parameter(torch.full((self.num_sequences,), 0.1))
            self.tds_summary_proj = nn.ModuleList(
                [nn.Linear(self.d_model, self.d_model) for _ in range(self.num_sequences)]
            )
            self.tds_gate = nn.ModuleList(
                [nn.Linear(2 * self.d_model, self.d_model) for _ in range(self.num_sequences)]
            )
            for proj, gate in zip(self.tds_summary_proj, self.tds_gate, strict=True):
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, -3.0)

        self._init_params()

        if self.emb_skip_threshold > 0:
            for domain in self.raw_seq_domains:
                idx_map = self._seq_emb_index[domain]
                skipped = sum(1 for idx in idx_map if idx == -1)
                if skipped > 0:
                    logger.info(
                        f"emb_skip_threshold={self.emb_skip_threshold}: "
                        f"{domain} skipped {skipped}/{len(idx_map)} features"
                    )
            for tag, tokenizer in (
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ):
                skipped = sum(1 for idx in tokenizer._emb_index if idx == -1)
                total = len(tokenizer._emb_index)
                if skipped > 0:
                    logger.info(
                        f"emb_skip_threshold={self.emb_skip_threshold}: "
                        f"{tag} skipped {skipped}/{total} features"
                    )

    def _init_params(self) -> None:
        """Xavier-init every Embedding weight, then zero-out the padding row."""

        def _xavier_zero_pad(emb: nn.Embedding) -> None:
            nn.init.xavier_normal_(emb.weight.data)
            emb.weight.data[0, :] = 0

        for domain in self.raw_seq_domains:
            for emb in self._seq_embs[domain]:
                _xavier_zero_pad(emb)

        for tokenizer in (self.user_ns_tokenizer, self.item_ns_tokenizer):
            for emb in tokenizer.embs:
                _xavier_zero_pad(emb)

        if self.num_time_buckets > 0:
            _xavier_zero_pad(self.time_embedding)
        if self.use_seq_gap_buckets:
            _xavier_zero_pad(self.gap_embedding)
        if self.use_global_time_token:
            for emb in self.global_time_embs:
                _xavier_zero_pad(emb)

    def _embed_seq_domain(
        self,
        domain: str,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: list[bool],
        emb_index: list[int],
        time_bucket_ids: torch.Tensor,
        gap_bucket_ids: torch.Tensor | None,
        ts_float_feats: torch.Tensor,
        ts_float_proj: nn.Module,
    ) -> torch.Tensor:
        """Project one seq domain's id+float side-info into ``d_model`` tokens."""
        B, S, L = seq.shape

        side_embs: list[torch.Tensor] = []
        idx_len = len(emb_index)
        for i in range(S):
            real_idx = emb_index[i] if i < idx_len else -1
            if real_idx < 0:
                # feature filtered by emb_skip_threshold → zeros preserve cat shape
                side_embs.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
                continue
            slot = sideinfo_embs[real_idx](seq[:, i, :])  # (B, L, emb_dim)
            if is_id[i] and self.training:
                slot = self.seq_id_emb_dropout(slot)
            side_embs.append(slot)

        # ts-float branch: (B, F, L) → (B, L, F) → emb_dim projection
        ts_emb = ts_float_proj(ts_float_feats.transpose(1, 2).contiguous())
        side_embs.append(ts_emb)

        token_emb = F.gelu(proj(torch.cat(side_embs, dim=-1)))

        if self.num_time_buckets > 0:
            tb_emb = self.time_embedding(time_bucket_ids)
            if self.use_time_gap_domain_gates:
                tb_emb = tb_emb * self.seq_time_gates[domain].view(1, 1, 1)
            token_emb = token_emb + tb_emb
        if self.use_seq_gap_buckets and gap_bucket_ids is not None:
            gb_emb = self.gap_embedding(gap_bucket_ids)
            if self.use_time_gap_domain_gates:
                gb_emb = gb_emb * self.seq_gap_gates[domain].view(1, 1, 1)
            token_emb = token_emb + gb_emb

        return token_emb

    def _make_padding_mask(self, seq_len: torch.Tensor, max_len: int) -> torch.Tensor:
        """``True`` at positions ≥ seq_len[b] (i.e. positions to be masked out)."""
        positions = torch.arange(max_len, device=seq_len.device).unsqueeze(0)
        return positions >= seq_len.unsqueeze(1)

    def _embed_global_time(self, global_time_feats: torch.Tensor) -> torch.Tensor:
        """Embed [hour, dow, weekend] ids into a ``(B, 1, D)`` token."""
        per_feat: list[torch.Tensor] = []
        for fi, emb in enumerate(self.global_time_embs):
            ids = global_time_feats[:, fi].long().clamp(0, emb.num_embeddings - 1)
            per_feat.append(emb(ids))
        cat = torch.cat(per_feat, dim=-1)
        return F.silu(self.global_time_proj(cat)).unsqueeze(1)

    def _encode_user_dense(self, user_dense_feats: torch.Tensor) -> list[torch.Tensor]:
        """Slice user dense into tokens: fid=61, fid=87 pool, semantic groups."""
        batch = user_dense_feats.size(0)

        # ---- token 1: fid=61 embedding ----
        emb_offset, emb_dim = self._user_emb_entry
        head = user_dense_feats[:, emb_offset : emb_offset + emb_dim]
        tok1 = F.silu(self.user_emb_proj(head)).unsqueeze(1)

        # ---- token 2: fid=87 history blocks via masked attention pool ----
        seq_offset, seq_dim = self._user_seq_entry
        history = user_dense_feats[:, seq_offset : seq_offset + seq_dim].view(
            batch, self._user_seq_num, self._user_seq_block_dim
        )
        nonzero = history.norm(dim=-1) > 0.1
        any_valid = nonzero.any(dim=-1, keepdim=True)

        raw_scores = self.user_seq_attn(history).squeeze(-1)
        gated = raw_scores.masked_fill(~nonzero, float("-inf"))
        gated = gated.masked_fill(~any_valid, 0.0)
        weights = torch.softmax(gated, dim=-1).unsqueeze(-1)
        pooled = (history * weights).sum(dim=1)
        tok2 = F.silu(self.user_seq_proj(pooled)).unsqueeze(1)

        if self.use_fid87_token_residual:
            tok1 = tok1 + tok2

        tokens = [tok1, tok2]

        for group_name in self._dense_group_names:
            entries = self._dense_group_entries[group_name]
            slices = [
                user_dense_feats[:, offset : offset + length]
                for offset, length in entries
            ]
            group_feats = torch.cat(slices, dim=-1)
            tokens.append(self._dense_group_projs[group_name](group_feats).unsqueeze(1))

        return tokens

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        seq_tokens_list: list,
        seq_masks_list: list,
        apply_dropout: bool = True,
        seq_ts_float_feats: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run the HyFormer block stack over per-seq Q/seq tokens, then project."""
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        # Snapshot pre-block seq tokens + masks aligned with the original L
        pre_block_seqs = list(seq_tokens_list)
        pre_block_masks = list(seq_masks_list)

        running_qs = q_tokens_list
        running_seqs = seq_tokens_list
        running_masks = seq_masks_list

        for layer in self.blocks:
            cos_buf: list[torch.Tensor] | None = None
            sin_buf: list[torch.Tensor] | None = None
            if self.rotary_emb is not None:
                cos_buf, sin_buf = [], []
                rope_device = running_seqs[0].device
                for seq_tensor in running_seqs:
                    cos_i, sin_i = self.rotary_emb(seq_tensor.shape[1], rope_device)
                    cos_buf.append(cos_i)
                    sin_buf.append(sin_i)

            running_qs, running_seqs, running_masks = maybe_gradient_checkpoint(
                layer,
                q_tokens_list=running_qs,
                seq_tokens_list=running_seqs,
                seq_padding_masks=running_masks,
                rope_cos_list=cos_buf,
                rope_sin_list=sin_buf,
                enabled=self.gradient_checkpointing,
            )

        if self.use_time_decay_summary and seq_ts_float_feats is not None:
            running_qs = self._apply_time_decay_summary(
                running_qs, pre_block_seqs, pre_block_masks, seq_ts_float_feats,
            )

        B = running_qs[0].shape[0]
        flat = torch.cat(running_qs, dim=1).view(B, -1)
        return self.output_proj(flat)

    def _apply_time_decay_summary(
        self,
        running_qs: list[torch.Tensor],
        pre_block_seqs: list[torch.Tensor],
        pre_block_masks: list[torch.Tensor],
        seq_ts_float_feats: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        """Per-domain recency-weighted summary fused into the item-side query."""
        out: list[torch.Tensor] = []
        for i, domain in enumerate(self.seq_domains):
            ts_feats = seq_ts_float_feats[domain]   # (B, F, L)
            delta = ts_feats[:, 0, :]               # log1p(diff_days)
            gap = ts_feats[:, 7, :]                 # inter-event gap
            alpha = F.softplus(self.tds_alpha[i])
            beta = F.softplus(self.tds_beta[i])
            logits = -alpha * delta - beta * gap    # (B, L)
            logits = logits.masked_fill(pre_block_masks[i], float("-inf"))
            all_pad = pre_block_masks[i].all(dim=-1, keepdim=True)
            logits = torch.where(all_pad, torch.zeros_like(logits), logits)
            w = F.softmax(logits, dim=-1).unsqueeze(-1)   # (B, L, 1)
            summary = (pre_block_seqs[i] * w).sum(dim=1)  # (B, D)
            summary = self.tds_summary_proj[i](summary)
            item_q = running_qs[i][:, -1, :]              # (B, D)
            gate = torch.sigmoid(
                self.tds_gate[i](torch.cat([item_q, summary], dim=-1))
            )
            new_q = running_qs[i].clone()
            new_q[:, -1, :] = item_q + gate * summary
            out.append(new_q)
        return out

    def _build_time_token(
        self,
        inputs: PCVRModelInput,
        ts_stats: list[torch.Tensor] | None = None,
        seq_lens: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Assemble the global time token from hod feats + per-domain ts stats."""
        B = inputs.user.int_values.shape[0]
        hod_slice = inputs.user.int_values.new_zeros(B, 1, dtype=torch.float32)
        return self.time_token_builder(hod_slice, ts_stats or [], seq_lens or [])

    def _build_ns_tokens(
        self, inputs: PCVRModelInput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialise the user-side and item-side NS token sequences."""
        user_int, pair_int = self._pair_plan.split_user_int(inputs.user.int_values)
        user_dense, pair_dense = self._pair_plan.split_user_dense(inputs.user.dense_values)

        cross_emb = None
        if self.cross_ns_tokenizer is not None and pair_int.shape[1] > 0:
            cross_emb = self.cross_ns_tokenizer(pair_int, pair_dense)

        user_seg: list[torch.Tensor] = [
            self.user_ns_tokenizer(user_int, extra_emb=cross_emb)
        ]
        if self._user_dense_split:
            user_seg.extend(self._encode_user_dense(user_dense))
        elif self.has_user_dense:
            user_seg.append(
                F.silu(self.user_dense_proj(user_dense)).unsqueeze(1)
            )
        if self.use_global_time_token:
            user_seg.append(
                self._embed_global_time(build_global_time_features(inputs.request_timestamp))
            )
        user_ns_seq = torch.cat(user_seg, dim=1)

        item_seg: list[torch.Tensor] = [
            self.item_ns_tokenizer(inputs.item.int_values)
        ]
        if self.has_item_dense:
            item_seg.append(self.item_dense_tokenizer(inputs.item.dense_values))
        item_ns_seq = torch.cat(item_seg, dim=1)

        return user_ns_seq, item_ns_seq

    def _derive_time_features(
        self, inputs: PCVRModelInput
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor] | None,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """Derive per-domain time features from raw event timestamps."""
        time_buckets: dict[str, torch.Tensor] = {}
        ts_float: dict[str, torch.Tensor] = {}
        ts_stat: dict[str, torch.Tensor] = {}
        for domain, sequence_input in inputs.sequences.items():
            time_buckets[domain] = compute_sequence_time_buckets(
                sequence_input.timestamps, inputs.request_timestamp
            )
            ts_float[domain] = compute_sequence_ts_float(
                sequence_input.timestamps,
                sequence_input.lengths,
                inputs.request_timestamp,
                domain,
            )
            ts_stat[domain] = compute_sequence_ts_stat(
                sequence_input.timestamps,
                sequence_input.lengths,
                inputs.request_timestamp,
            )
        gap_buckets: dict[str, torch.Tensor] | None = None
        if self.use_seq_gap_buckets:
            gap_buckets = {
                domain: compute_sequence_gap_buckets(
                    sequence_input.timestamps, sequence_input.lengths
                )
                for domain, sequence_input in inputs.sequences.items()
            }
        return time_buckets, gap_buckets, ts_float, ts_stat

    def _build_seq_tokens(
        self,
        inputs: PCVRModelInput,
        time_buckets: dict[str, torch.Tensor],
        gap_buckets: dict[str, torch.Tensor] | None,
        ts_float: dict[str, torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, torch.Tensor]]:
        """Embed every raw sequence domain and return per-view (tokens, mask)."""
        cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for domain in self.raw_seq_domains:
            sequence_input = inputs.sequences[domain]
            tokens = self._embed_seq_domain(
                domain,
                sequence_input.values,
                self._seq_embs[domain],
                self._seq_proj[domain],
                self._seq_is_id[domain],
                self._seq_emb_index[domain],
                time_buckets[domain],
                gap_buckets.get(domain) if gap_buckets is not None else None,
                ts_float[domain],
                self._seq_ts_float_proj[domain],
            )
            mask = self._make_padding_mask(
                sequence_input.lengths, sequence_input.values.shape[2]
            )
            cache[domain] = (tokens, mask)

        output_domains = self.raw_seq_domains if self.use_time_aligned_interleave else self.seq_domains
        dom_token_seqs: list[torch.Tensor] = []
        dom_pad_masks: list[torch.Tensor] = []
        dom_ts_float: dict[str, torch.Tensor] = {}
        for exp_domain in output_domains:
            orig_domain, ratio = self._seq_domain_ratios.get(exp_domain, (exp_domain, 1.0))
            tokens, mask = cache[orig_domain]
            view_ts_float = ts_float[orig_domain]
            if ratio < 1.0:
                L = tokens.size(1)
                keep = max(1, int(L * ratio))
                tokens = tokens[:, :keep, :]
                mask = mask[:, :keep]
                view_ts_float = view_ts_float[:, :, :keep]
            dom_token_seqs.append(tokens)
            dom_pad_masks.append(mask)
            dom_ts_float[exp_domain] = view_ts_float

        return dom_token_seqs, dom_pad_masks, dom_ts_float

    def _build_interleaved_seq_tokens(
        self,
        inputs: PCVRModelInput,
        time_buckets: dict[str, torch.Tensor],
        gap_buckets: dict[str, torch.Tensor] | None,
        ts_float: dict[str, torch.Tensor],
        ts_stat: dict[str, torch.Tensor],
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        dict[str, torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """Build a single global sequence by time-sorting all domain tokens."""
        raw_tokens, raw_masks, _ = self._build_seq_tokens(
            inputs, time_buckets, gap_buckets, ts_float
        )
        token_cat = torch.cat(raw_tokens, dim=1)
        mask_cat = torch.cat(raw_masks, dim=1)
        timestamp_cat = torch.cat(
            [inputs.sequences[d].timestamps for d in self.raw_seq_domains], dim=1
        )
        ts_float_cat = torch.cat([ts_float[d] for d in self.raw_seq_domains], dim=2)

        request_delta = (inputs.request_timestamp[:, None] - timestamp_cat).clamp_min(0)
        padding_key = torch.full_like(request_delta, torch.iinfo(request_delta.dtype).max)
        sort_key = torch.where(mask_cat, padding_key, request_delta)
        sort_idx = torch.argsort(sort_key, dim=1, stable=True)
        token_idx = sort_idx.unsqueeze(-1).expand(-1, -1, token_cat.shape[-1])
        ts_idx = sort_idx.unsqueeze(1).expand(-1, ts_float_cat.shape[1], -1)

        inter_tokens = torch.gather(token_cat, 1, token_idx)
        inter_mask = torch.gather(mask_cat, 1, sort_idx)
        inter_ts_float = torch.gather(ts_float_cat, 2, ts_idx)

        stat_parts = [ts_stat[d] for d in self.raw_seq_domains]
        inter_stats = torch.stack(stat_parts, dim=0).mean(dim=0)
        inter_lens = torch.stack(
            [inputs.sequences[d].lengths for d in self.raw_seq_domains], dim=0
        ).sum(dim=0)
        inter_lens = inter_lens.clamp(max=inter_tokens.shape[1])

        return (
            [inter_tokens],
            [inter_mask],
            {"interleave": inter_ts_float},
            [inter_stats],
            [inter_lens],
        )

    def _compute_logits(
        self, inputs: PCVRModelInput, apply_dropout: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared core for forward/predict — only ``apply_dropout`` differs."""
        time_buckets, gap_buckets, ts_float, ts_stat = self._derive_time_features(inputs)
        user_ns_seq, item_ns_seq = self._build_ns_tokens(inputs)
        if self.use_time_aligned_interleave:
            (
                dom_token_seqs,
                dom_pad_masks,
                seq_ts_float_feats,
                ts_stats,
                seq_lens,
            ) = self._build_interleaved_seq_tokens(
                inputs, time_buckets, gap_buckets, ts_float, ts_stat
            )
        else:
            dom_token_seqs, dom_pad_masks, seq_ts_float_feats = self._build_seq_tokens(
                inputs, time_buckets, gap_buckets, ts_float
            )
            ts_stats = [ts_stat[self._seq_domain_ratios.get(d, (d, 1.0))[0]] for d in self.seq_domains]
            seq_lens = [inputs.sequences[self._seq_domain_ratios.get(d, (d, 1.0))[0]].lengths for d in self.seq_domains]

        time_tok = self._build_time_token(inputs, ts_stats=ts_stats, seq_lens=seq_lens)
        q_per_seq = self.query_generator(user_ns_seq, item_ns_seq, time_tok, ts_stats)

        embedding = self._run_multi_seq_blocks(
            q_per_seq, dom_token_seqs, dom_pad_masks, apply_dropout=apply_dropout,
            seq_ts_float_feats=seq_ts_float_feats,
        )
        return self.clsfier(embedding), embedding

    def forward(self, inputs: PCVRModelInput) -> torch.Tensor:
        """Runs the forward pass of the PCVRDualQ model."""
        logits, _embedding = self._compute_logits(inputs, apply_dropout=self.training)
        return logits

    def predict(self, inputs: PCVRModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs inference without dropout, returning both logits and embeddings."""
        return self._compute_logits(inputs, apply_dropout=False)
