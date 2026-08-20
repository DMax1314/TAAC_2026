"""Deterministic row-level hash split primitives for PCVR datasets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from numbers import Integral
from pathlib import Path
import zlib


PCVR_HASH_SPLIT_DENOMINATOR = 1_000_003
PCVR_HASH_SPLIT_STRATEGIES = ("user_hash", "sample_hash")
PCVR_HASH_SPLIT_ROLES = ("train", "valid")


@dataclass(frozen=True, slots=True)
class PCVRHashSplitFilter:
    strategy: str
    role: str
    valid_ratio: float
    train_ratio: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.strategy not in PCVR_HASH_SPLIT_STRATEGIES:
            raise ValueError(f"unsupported hash split strategy={self.strategy!r}")
        if self.role not in PCVR_HASH_SPLIT_ROLES:
            raise ValueError(f"unsupported hash split role={self.role!r}")
        if not 0.0 < self.valid_ratio < 1.0:
            raise ValueError("hash split requires 0 < valid_ratio < 1")
        if not 0.0 < self.train_ratio <= 1.0:
            raise ValueError("hash split requires 0 < train_ratio <= 1")


def pcvr_hash_split_filter_to_dict(
    split_filter: PCVRHashSplitFilter | None,
) -> dict[str, str | float | int] | None:
    if split_filter is None:
        return None
    return {
        "strategy": split_filter.strategy,
        "role": split_filter.role,
        "valid_ratio": float(split_filter.valid_ratio),
        "train_ratio": float(split_filter.train_ratio),
        "seed": int(split_filter.seed),
    }


def pcvr_hash_split_mask(
    sample_positions: Sequence[int],
    *,
    file_path: str | Path,
    row_group_index: int,
    split_filter: PCVRHashSplitFilter,
    user_values: Sequence[object] | None = None,
) -> tuple[bool, ...]:
    positions = tuple(int(value) for value in sample_positions)
    file_salt = zlib.crc32(Path(file_path).name.encode("utf-8"))
    sample_salt = int(split_filter.seed) ^ file_salt ^ (int(row_group_index) << 32)
    if split_filter.strategy == "user_hash" and user_values is not None:
        users = tuple(stable_pcvr_user_hash_value(value) for value in user_values)
        if len(users) != len(positions):
            raise ValueError("user_values must have the same length as sample_positions")
        values_and_salts = (
            (user_value, int(split_filter.seed))
            if user_value is not None
            else (sample_position, sample_salt)
            for sample_position, user_value in zip(positions, users, strict=True)
        )
    else:
        values_and_salts = ((sample_position, sample_salt) for sample_position in positions)

    valid_threshold = max(
        1,
        min(
            PCVR_HASH_SPLIT_DENOMINATOR - 1,
            round(float(split_filter.valid_ratio) * PCVR_HASH_SPLIT_DENOMINATOR),
        ),
    )
    train_threshold = valid_threshold + round(
        (PCVR_HASH_SPLIT_DENOMINATOR - valid_threshold) * float(split_filter.train_ratio)
    )
    train_threshold = min(PCVR_HASH_SPLIT_DENOMINATOR, train_threshold)

    membership: list[bool] = []
    for value, salt in values_and_salts:
        score = stable_pcvr_hash_score(value, salt)
        if split_filter.role == "valid":
            membership.append(score < valid_threshold)
        else:
            membership.append(valid_threshold <= score < train_threshold)
    return tuple(membership)


def stable_pcvr_hash_score(value: int, salt: int) -> int:
    mixed = (int(value) ^ int(salt)) & 0xFFFFFFFFFFFFFFFF
    mixed ^= mixed >> 30
    mixed = (mixed * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    mixed ^= mixed >> 27
    mixed = (mixed * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    mixed ^= mixed >> 31
    return int(mixed % PCVR_HASH_SPLIT_DENOMINATOR)


def stable_pcvr_user_hash_value(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, bytes):
        encoded = value
    else:
        text = str(value)
        if not text.strip():
            return None
        encoded = text.encode("utf-8")
    digest = hashlib.blake2b(b"pcvr-user-id\0" + encoded, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


__all__ = [
    "PCVRHashSplitFilter",
    "pcvr_hash_split_filter_to_dict",
    "pcvr_hash_split_mask",
    "stable_pcvr_hash_score",
    "stable_pcvr_user_hash_value",
]
