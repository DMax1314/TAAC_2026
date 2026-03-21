from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig, load_config
from .data import load_dataloaders
from .model import build_model
from .utils import ensure_dir, resolve_device, set_seed, write_json


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()

        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions: list[float] = []
    labels: list[float] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch_to_device(batch, device)
        logits = model(batch)
        loss = criterion(logits, batch["labels"])

        probabilities = torch.sigmoid(logits)
        predictions.extend(probabilities.detach().cpu().tolist())
        labels.extend(batch["labels"].detach().cpu().tolist())

        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    auc = roc_auc_score(labels, predictions) if len(set(labels)) > 1 else 0.5
    return {
        "loss": total_loss / max(total_examples, 1),
        "auc": float(auc),
    }


@torch.no_grad()
def benchmark_latency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_steps: int = 10,
    measure_steps: int = 30,
) -> dict[str, float]:
    model.eval()
    iterator = iter(loader)
    timings: list[float] = []

    for step in range(warmup_steps + measure_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        batch = move_batch_to_device(batch, device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.perf_counter() - start

        if step >= warmup_steps:
            timings.append(duration / max(batch["labels"].size(0), 1))

    if not timings:
        return {"mean_ms_per_sample": 0.0, "p95_ms_per_sample": 0.0}

    timings_ms = torch.tensor(timings, dtype=torch.float32) * 1000.0
    return {
        "mean_ms_per_sample": float(timings_ms.mean().item()),
        "p95_ms_per_sample": float(torch.quantile(timings_ms, 0.95).item()),
    }


def run_training(config: ExperimentConfig) -> None:
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)
    output_dir = ensure_dir(config.train.output_dir)

    train_loader, val_loader, data_stats = load_dataloaders(
        config=config.data,
        vocab_size=config.model.vocab_size,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
    )

    model = build_model(
        config=config.model,
        dense_dim=int(data_stats["dense_dim"]),
        max_seq_len=config.data.max_seq_len,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    pos_weight = torch.tensor([data_stats["pos_weight"]], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    best_auc = -1.0
    history: list[dict[str, float]] = []

    print(f"device={device}")
    print(f"model={config.model.name} parameters={parameter_count}")
    print(
        f"train_size={int(data_stats['train_size'])} val_size={int(data_stats['val_size'])} train_positive_rate={data_stats['train_positive_rate']:.4f}"
    )

    for epoch in range(1, config.train.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, criterion, device)
        epoch_result = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": metrics["loss"],
            "val_auc": metrics["auc"],
        }
        history.append(epoch_result)
        print(
            f"epoch={epoch} train_loss={train_loss:.5f} val_loss={metrics['loss']:.5f} val_auc={metrics['auc']:.5f}"
        )

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "data_stats": data_stats,
                    "best_val_auc": best_auc,
                },
                output_dir / "best.pt",
            )

    latency = benchmark_latency(model, val_loader, device)
    summary = {
        "model_name": config.model.name,
        "parameter_count": parameter_count,
        "best_val_auc": best_auc,
        "latency": latency,
        "data_stats": data_stats,
        "history": history,
    }
    write_json(output_dir / "summary.json", summary)
    print(
        f"best_val_auc={best_auc:.5f} mean_latency_ms_per_sample={latency['mean_ms_per_sample']:.4f} p95_latency_ms_per_sample={latency['p95_ms_per_sample']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TAAC 2026 baseline model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    run_training(config)


if __name__ == "__main__":
    main()
