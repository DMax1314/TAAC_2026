from pathlib import Path

from taac2026.api import FunctionExperiment


def _train(request):
    request.run_dir.mkdir(parents=True, exist_ok=True)
    marker = request.run_dir / "smoke_train.txt"
    marker.write_text("ok", encoding="utf-8")
    return {"run_dir": str(request.run_dir), "marker": str(marker)}


def _evaluate(request):
    return {"run_dir": str(request.run_dir)}


def _infer(request):
    return {"result_dir": str(request.result_dir)}


EXPERIMENT = FunctionExperiment(
    name="smoke_experiment",
    package_dir=Path(__file__).resolve().parent,
    train_fn=_train,
    evaluate_fn=_evaluate,
    infer_fn=_infer,
    requires_dataset=False,
)

__all__ = ["EXPERIMENT"]
