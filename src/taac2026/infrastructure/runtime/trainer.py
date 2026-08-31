"""Shared PCVR pointwise trainer."""

from __future__ import annotations

import sys
import time
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from taac2026.domain.metrics import binary_auc, binary_score_diagnostics
from taac2026.domain.config import (
    PCVR_EARLY_STOPPING_METRIC_CHOICES,
    PCVRTrainConfig,
)
from taac2026.infrastructure.data.batches import PCVRBatch
from taac2026.infrastructure.logging import logger
from taac2026.infrastructure.modeling.tensors import sigmoid_probabilities_numpy
from taac2026.infrastructure.runtime.checkpoint_io import PCVRTrainerSupportMixin
from taac2026.infrastructure.runtime.ema import ExponentialMovingAverage
from taac2026.infrastructure.runtime.execution import (
    EarlyStopping,
    build_sparse_optimizer,
    compute_pcvr_loss,
    create_grad_scaler,
    maybe_compile_callable,
    maybe_prepare_internal_compile,
    runtime_autocast_context,
    runtime_execution_summary,
)
from taac2026.infrastructure.runtime.reporting import NoopTrainReporter, TrainReporter
from taac2026.infrastructure.checkpoints import preferred_checkpoint_path
from taac2026.infrastructure.runtime.protocols import SparseParameterModel


def clip_grad_norms_with_sparse(
    parameters: Any,
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """Gradient norm clipping that supports sparse COO gradients.

    ``torch.nn.utils.clip_grad_norm_`` in torch 2.13 no longer special-cases
    sparse gradients and calls ``linalg_vector_norm`` on them directly, which
    raises ``NotImplementedError`` for the SparseCUDA backend. This mirrors the
    classic behavior: norms are computed over ``_values()`` for sparse tensors,
    and clipping scales ``_values()`` in place for coalesced sparse gradients.
    """
    norms: list[torch.Tensor] = []
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse:
            # L2 norm over stored values (including duplicates for uncoalesced
            # gradients) matches the classic torch clip_grad_norm_ behavior.
            norms.append(gradient._values().norm(norm_type))
        else:
            norms.append(gradient.norm(norm_type))
    if not norms:
        return torch.zeros((), dtype=torch.float32)
    total_norm = torch.linalg.vector_norm(torch.stack(norms), norm_type)
    clip_coef = float(max_norm) / (float(total_norm) + 1e-6)
    if clip_coef < 1.0:
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                if not gradient.is_coalesced():
                    parameter.grad = gradient.coalesce()
                    gradient = parameter.grad
                gradient._values().mul_(clip_coef)
            else:
                gradient.mul_(clip_coef)
    return total_norm


def _use_interactive_progress() -> bool:
    isatty = getattr(sys.stderr, "isatty", None)
    return bool(isatty and isatty())


def _should_log_progress(current_batch: int, total_batches: int, interval: int) -> bool:
    return current_batch == 1 or current_batch == total_batches or current_batch % interval == 0


def _format_duration(seconds: float) -> str:
    return str(timedelta(seconds=max(0, round(seconds))))


class PCVRPointwiseTrainer(PCVRTrainerSupportMixin):
    """PCVR trainer for binary pointwise classification with AUC monitoring."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: PCVRTrainConfig,
        save_dir: str | Path,
        schema_path: str | Path,
        reporter: TrainReporter | None = None,
    ) -> None:
        optimizer_config = config.optimizer
        ema_config = config.ema
        sparse_config = config.sparse_optimizer
        self.config = config
        self.device = optimizer_config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.reporter = reporter or NoopTrainReporter()
        self.schema_path = Path(schema_path).expanduser().resolve()
        self.dense_optimizer_type = optimizer_config.dense_optimizer_type
        self.scheduler_type = optimizer_config.scheduler_type
        self.warmup_steps = optimizer_config.warmup_steps
        self.min_lr_ratio = optimizer_config.min_lr_ratio
        self.base_dense_lr = float(optimizer_config.lr)
        self.current_dense_lr = self.base_dense_lr
        self.optim_step = 0
        self.dense_params: list[nn.Parameter] = []

        self.sparse_optimizer: torch.optim.Optimizer | None
        if isinstance(model, SparseParameterModel):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            if not sparse_params:
                logger.info(
                    "Model exposes get_sparse_params but has no embedding parameters; using {} for all params",
                    self._dense_optimizer_display_name(),
                )
                self.sparse_optimizer = None
                self.dense_params = list(model.parameters())
                self.dense_optimizer = self._build_dense_optimizer(self.dense_params, self.base_dense_lr)
            else:
                self.dense_params = list(dense_params)
                sparse_param_count = sum(parameter.numel() for parameter in sparse_params)
                dense_param_count = sum(parameter.numel() for parameter in dense_params)
                logger.info(
                    "Sparse params: {} tensors, {} parameters (Adagrad lr={})",
                    len(sparse_params),
                    f"{sparse_param_count:,}",
                    sparse_config.sparse_lr,
                )
                logger.info(
                    "Dense params: {} tensors, {} parameters ({} lr={})",
                    len(dense_params),
                    f"{dense_param_count:,}",
                    self._dense_optimizer_display_name(),
                    self.base_dense_lr,
                )
                self.sparse_optimizer = build_sparse_optimizer(
                    sparse_params,
                    sparse_lr=sparse_config.sparse_lr,
                    sparse_weight_decay=sparse_config.sparse_weight_decay,
                    runtime_execution=config.runtime,
                    device=self.device,
                )
                self.dense_optimizer: torch.optim.Optimizer = self._build_dense_optimizer(self.dense_params, self.base_dense_lr)
        else:
            self.sparse_optimizer = None
            self.dense_params = list(model.parameters())
            self.dense_optimizer = self._build_dense_optimizer(self.dense_params, self.base_dense_lr)

        self.max_steps = int(optimizer_config.max_steps)
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.early_stopping = EarlyStopping(
            checkpoint_path=preferred_checkpoint_path(self.save_dir),
            patience_steps=optimizer_config.patience_steps,
            label="model",
        )
        self.runtime_execution = config.runtime
        uses_internal_compile = maybe_prepare_internal_compile(
            self.model,
            enabled=self.runtime_execution.compile,
            label="PCVR training model",
        )
        self.forward_model = maybe_compile_callable(
            self.model,
            enabled=self.runtime_execution.compile and not uses_internal_compile,
            label="PCVR training forward",
        )
        self.predict_fn = maybe_compile_callable(
            self.model.predict,
            enabled=self.runtime_execution.compile and not uses_internal_compile,
            label="PCVR trainer predict",
        )
        self.grad_scaler = create_grad_scaler(self.runtime_execution, self.device)
        self.ema: ExponentialMovingAverage | None = None
        if ema_config.enabled:
            self.ema = ExponentialMovingAverage.from_model(
                self.model,
                decay=float(ema_config.decay),
                start_step=int(ema_config.start_step),
                update_every_n_steps=int(ema_config.update_every_n_steps),
            )
            logger.info(
                "PCVR model EMA enabled: decay={}, start_step={}, update_every_n_steps={}",
                self.ema.decay,
                self.ema.start_step,
                self.ema.update_every_n_steps,
            )
        self.loss_config = config.loss
        self.last_train_loss_components: dict[str, float] = {}
        self.reinit_sparse_every_n_steps = sparse_config.reinit_sparse_every_n_steps
        self.reinit_cardinality_threshold = sparse_config.reinit_cardinality_threshold
        self.sparse_lr = sparse_config.sparse_lr
        self.sparse_weight_decay = sparse_config.sparse_weight_decay
        self.eval_every_n_steps = config.data.eval_every_n_steps
        self.early_stopping_metric = config.validation.early_stopping_metric
        if self.early_stopping_metric not in PCVR_EARLY_STOPPING_METRIC_CHOICES:
            raise ValueError(f"unsupported early stopping metric: {self.early_stopping_metric}")
        self.last_eval_diagnostics: dict[str, float | int] = {}
        self.last_eval_metrics: dict[str, float] = {}
        self.last_train_model_scalars: dict[str, float] = {}
        self.last_eval_model_scalars: dict[str, float] = {}

        logger.info(
            "PCVRPointwiseTrainer loss_terms={}, "
            "dense_optimizer_type={}, scheduler_type={}, warmup_steps={}, min_lr_ratio={}, "
            "max_steps={}, reinit_sparse_every_n_steps={}, "
            "ema_enabled={}, early_stopping_metric={}",
            self.loss_config.summary(),
            self.dense_optimizer_type,
            self.scheduler_type,
            self.warmup_steps,
            self.min_lr_ratio,
            self.max_steps,
            self.reinit_sparse_every_n_steps,
            self.ema is not None,
            self.early_stopping_metric,
        )
        logger.info("PCVRPointwiseTrainer runtime: {}", runtime_execution_summary(self.runtime_execution, self.device))

    def _log_loop_progress(
        self,
        phase: str,
        current_batch: int,
        total_batches: int,
        *,
        loop_started_at: float | None = None,
        loss: float | None = None,
    ) -> None:
        message = f"{phase} progress {current_batch}/{total_batches} ({current_batch / total_batches:.1%})"
        if loop_started_at is not None and current_batch > 0:
            elapsed_seconds = max(0.0, time.monotonic() - loop_started_at)
            eta_seconds = elapsed_seconds * max(0, total_batches - current_batch) / current_batch
            message = f"{message} | eta={_format_duration(eta_seconds)}"
        if loss is not None:
            message = f"{message} | loss={loss:.4f}"
        logger.info(message)

    def train(self) -> None:
        logger.info("Start Training (PCVR pointwise)")
        self.model.train()

        total_step = 0
        total_train_steps = self.max_steps if self.max_steps > 0 else self._logical_train_sweep_steps()
        use_tqdm = _use_interactive_progress()
        log_interval = self.runtime_execution.progress_log_interval_steps
        loop_started_at = time.monotonic()
        eval_interval = self.eval_every_n_steps if self.eval_every_n_steps > 0 else self._logical_train_sweep_steps()
        train_iter = self._infinite_train_batches()
        train_pbar = tqdm(total=total_train_steps, dynamic_ncols=True) if use_tqdm else None
        window_loss_sum = 0.0
        window_loss_steps = 0
        evaluated_on_last_step = False

        while total_step < total_train_steps:
            batch = next(train_iter)
            loss = self._train_step(batch)
            total_step += 1
            window_loss_sum += loss
            window_loss_steps += 1

            self.reporter.train_step(
                step=total_step,
                loss=loss,
                loss_components=self.last_train_loss_components,
                dense_lr=self.current_dense_lr,
            )
            self._write_model_training_scalars("train", self.last_train_model_scalars, total_step)

            if train_pbar is not None:
                train_pbar.update(1)
                train_pbar.set_postfix({"loss": f"{loss:.4f}"})
            elif _should_log_progress(total_step, total_train_steps, log_interval):
                self._log_loop_progress(
                    "Train",
                    total_step,
                    total_train_steps,
                    loop_started_at=loop_started_at,
                    loss=loss,
                )

            if self.reinit_sparse_every_n_steps > 0 and total_step % self.reinit_sparse_every_n_steps == 0:
                self._rebuild_sparse_optimizer(total_step)

            if total_step % eval_interval != 0 and total_step != total_train_steps:
                continue

            logger.info("Train step {}, Average Loss: {}", total_step, window_loss_sum / max(1, window_loss_steps))
            window_loss_sum = 0.0
            window_loss_steps = 0

            logger.info("Evaluating at step {}", total_step)
            val_auc, val_logloss = self.evaluate(step=total_step)
            self.model.train()
            torch.cuda.empty_cache()

            logger.info("Step {} Validation | AUC: {}, LogLoss: {}", total_step, val_auc, val_logloss)

            self._report_validation(total_step, val_auc, val_logloss)

            self._handle_validation_result(total_step, val_auc, val_logloss)
            evaluated_on_last_step = total_step == total_train_steps

            if self.early_stopping.early_stop:
                logger.info("Early stopping at step {}", total_step)
                if train_pbar is not None:
                    train_pbar.close()
                return

        if train_pbar is not None:
            train_pbar.close()

        if not evaluated_on_last_step:
            logger.info("Evaluating at step {}", total_step)
            val_auc, val_logloss = self.evaluate(step=total_step)
            self.model.train()
            torch.cuda.empty_cache()
            logger.info("Step {} Validation | AUC: {}, LogLoss: {}", total_step, val_auc, val_logloss)
            self._report_validation(total_step, val_auc, val_logloss)
            self._handle_validation_result(total_step, val_auc, val_logloss)

    @contextmanager
    def _ema_evaluation_context(self) -> Iterator[None]:
        if self.ema is None:
            yield
            return
        with self.ema.apply_to(self.model):
            yield

    def _update_ema(self, step: int) -> None:
        if self.ema is not None:
            self.ema.update(self.model, step=step)

    def _sync_ema_after_model_reinit(self) -> None:
        if self.ema is not None:
            self.ema.copy_from(self.model)
            logger.info("Synchronized model EMA after sparse parameter reinitialization")

    def _train_step(self, batch: PCVRBatch) -> float:
        device_batch = self._batch_to_device(batch)
        label = device_batch.label.float()
        self._set_dense_learning_rate(self.optim_step + 1)

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        collect_model_scalars = self._should_collect_train_model_scalars(self.optim_step + 1)
        self._set_model_training_diagnostics_enabled(collect_model_scalars)
        model_input = device_batch.inputs
        with runtime_autocast_context(self.runtime_execution, self.device):
            logits = self.forward_model(model_input).squeeze(-1)
            loss, loss_components = compute_pcvr_loss(logits, label, self.loss_config, model=self.model)

        # Detect NaN/inf in logits and loss before backward to prevent silent
        # model corruption, especially under AMP where reduced precision can
        # produce NaN that GradScaler (float16) skips silently or bfloat16
        # propagates directly.
        if not torch.isfinite(loss).all():
            n_bad_logits = int((~torch.isfinite(logits)).sum())
            loss_value = loss.detach().float().item()
            logger.warning(
                "Train step skipped: non-finite loss={:.6f}, non-finite logits={}/{}. "
                "Skipping backward and optimizer step to avoid parameter corruption.",
                loss_value,
                n_bad_logits,
                logits.numel(),
            )
            self.last_train_loss_components = {
                name: float("nan") for name in loss_components
            }
            self.last_train_model_scalars = {}
            self._set_model_training_diagnostics_enabled(False)
            # Do NOT increment optim_step; this step produced no valid gradient.
            return float("nan")

        self.last_train_loss_components = {name: float(value.detach().float().cpu()) for name, value in loss_components.items()}

        if collect_model_scalars:
            self.last_train_model_scalars = self._consume_model_training_scalars("train")
        else:
            self.last_train_model_scalars = {}
        self._set_model_training_diagnostics_enabled(False)

        optimizer_step_applied = True
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.grad_scaler.unscale_(self.sparse_optimizer)

            # After unscale, check whether any gradient is inf/nan.  GradScaler
            # will skip the optimizer step internally when this happens, but we
            # log it so the event is visible instead of silent.
            scale_before_step = self.grad_scaler.get_scale()

            clip_grad_norms_with_sparse(self.model.parameters(), max_norm=1.0)
            self._orthogonalize_dense_gradients()

            self.grad_scaler.step(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.grad_scaler.step(self.sparse_optimizer)
            self.grad_scaler.update()

            scale_after_update = self.grad_scaler.get_scale()
            if scale_after_update < scale_before_step:
                optimizer_step_applied = False
                logger.warning(
                    "Train step: GradScaler reduced scale {:.1e} -> {:.1e}, "
                    "indicating inf/nan gradients were found and optimizer step was skipped.",
                    scale_before_step,
                    scale_after_update,
                )
        else:
            loss.backward()
            clip_grad_norms_with_sparse(self.model.parameters(), max_norm=1.0)
            self._orthogonalize_dense_gradients()

            self.dense_optimizer.step()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.step()

        self.optim_step += 1
        if optimizer_step_applied:
            self._update_ema(self.optim_step)

        return loss.item()

    def evaluate(self, step: int | None = None) -> tuple[float, float]:
        logger.info("Start Evaluation (PCVR pointwise) - validation")
        self.model.eval()

        total_valid_batches = len(self.valid_loader)
        use_tqdm = _use_interactive_progress()
        log_interval = self.runtime_execution.progress_log_interval_steps
        loop_started_at = time.monotonic()
        valid_iter = enumerate(self.valid_loader)
        pbar = (
            tqdm(valid_iter, total=total_valid_batches, dynamic_ncols=True)
            if use_tqdm
            else valid_iter
        )
        all_logits_list = []
        all_labels_list = []
        model_scalar_sums: dict[str, float] = {}
        model_scalar_counts: dict[str, int] = {}
        collect_model_scalars = self.reporter.should_collect_model_scalars(phase="valid", step=step, trainer=self)

        with self._ema_evaluation_context():
            with torch.inference_mode():
                for step_index, batch in pbar:
                    self._set_model_training_diagnostics_enabled(collect_model_scalars)
                    logits, labels = self._evaluate_step(batch)
                    if collect_model_scalars:
                        self._accumulate_model_training_scalars("valid", model_scalar_sums, model_scalar_counts)
                    self._set_model_training_diagnostics_enabled(False)
                    all_logits_list.append(logits.detach().clone())
                    all_labels_list.append(labels.detach())
                    current_batch = step_index + 1
                    if not use_tqdm and _should_log_progress(current_batch, total_valid_batches, log_interval):
                        self._log_loop_progress(
                            "Validation",
                            current_batch,
                            total_valid_batches,
                            loop_started_at=loop_started_at,
                        )

        if use_tqdm:
            pbar.close()

        all_logits = torch.cat(all_logits_list, dim=0).float()
        all_labels = torch.cat(all_labels_list, dim=0).long()
        auc, logloss, diagnostics = self._compute_validation_metrics(all_logits, all_labels, label="Evaluate")

        self.last_eval_diagnostics = diagnostics
        self.last_eval_metrics = {"auc": auc, "logloss": logloss}
        self.last_eval_model_scalars = {
            tag: model_scalar_sums[tag] / model_scalar_counts[tag]
            for tag in model_scalar_sums
            if model_scalar_counts[tag] > 0
        }
        logger.info(
            "Validation score diagnostics | pos={} neg={} pos_mean={:.6f} neg_mean={:.6f} margin={:.6f} score_std={:.6f}",
            self.last_eval_diagnostics["positive_count"],
            self.last_eval_diagnostics["negative_count"],
            self.last_eval_diagnostics["positive_score_mean"],
            self.last_eval_diagnostics["negative_score_mean"],
            self.last_eval_diagnostics["score_margin_mean"],
            self.last_eval_diagnostics["score_std"],
        )

        return auc, logloss

    def _report_validation(self, total_step: int, val_auc: float, val_logloss: float) -> None:
        self.reporter.validation_step(
            step=total_step,
            auc=val_auc,
            logloss=val_logloss,
            metrics=self.last_eval_metrics,
            score_diagnostics=self.last_eval_diagnostics,
        )
        self._write_model_training_scalars("valid", self.last_eval_model_scalars, total_step)

    def _set_model_training_diagnostics_enabled(self, enabled: bool) -> None:
        self.reporter.set_model_diagnostics_enabled(self.model, enabled)

    def _should_collect_train_model_scalars(self, step: int) -> bool:
        return self.reporter.should_collect_model_scalars(phase="train", step=step, trainer=self)

    def _consume_model_training_scalars(self, phase: str) -> dict[str, float]:
        scalars = self.reporter.consume_model_scalars(self.model, phase=phase)
        if not isinstance(scalars, Mapping):
            return {}
        cleaned: dict[str, float] = {}
        for tag, value in scalars.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric_value):
                cleaned[str(tag)] = numeric_value
        return cleaned

    def _accumulate_model_training_scalars(
        self,
        phase: str,
        scalar_sums: dict[str, float],
        scalar_counts: dict[str, int],
    ) -> None:
        for tag, value in self._consume_model_training_scalars(phase).items():
            scalar_sums[tag] = scalar_sums.get(tag, 0.0) + value
            scalar_counts[tag] = scalar_counts.get(tag, 0) + 1

    def _write_model_training_scalars(self, phase: str, scalars: dict[str, float], total_step: int) -> None:
        self.reporter.model_scalars(phase=phase, step=total_step, scalars=scalars)

    def _compute_validation_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        label: str,
    ) -> tuple[float, float, dict[str, float | int]]:
        valid_logit_mask = ~torch.isnan(logits)
        valid_logits = logits[valid_logit_mask]
        valid_labels = labels[valid_logit_mask]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float("inf")

        probabilities = sigmoid_probabilities_numpy(logits)
        labels_np = labels.detach().cpu().numpy()
        nan_mask = np.isnan(probabilities)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logger.warning("[{}] {}/{} predictions are NaN, filtering them out", label, n_nan, len(probabilities))
            valid_mask = ~nan_mask
            probabilities = probabilities[valid_mask]
            labels_np = labels_np[valid_mask]

        auc = binary_auc(labels_np, probabilities)
        diagnostics = binary_score_diagnostics(labels_np, probabilities)
        return auc, logloss, diagnostics

    def validation_metric_score(self, metric_name: str, val_auc: float, val_logloss: float) -> float:
        if metric_name == "auc":
            return float(val_auc)
        if metric_name == "logloss":
            return -float(val_logloss)
        raise ValueError(f"unsupported validation metric: {metric_name}")

    def validation_early_stopping_score(self, val_auc: float, val_logloss: float) -> float:
        return self.validation_metric_score(self.early_stopping_metric, val_auc, val_logloss)

    def _evaluate_step(self, batch: PCVRBatch) -> tuple[torch.Tensor, torch.Tensor]:
        device_batch = self._batch_to_device(batch)
        label = device_batch.label

        model_input = device_batch.inputs
        with runtime_autocast_context(self.runtime_execution, self.device):
            logits, _embeddings = self.predict_fn(model_input)
        logits = logits.squeeze(-1)

        return logits, label
