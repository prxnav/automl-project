from __future__ import annotations

from typing import TYPE_CHECKING

from automl_image.types import RunResult

if TYPE_CHECKING:
    from torch import nn


def n_params(model: "nn.Module") -> int:
    raise NotImplementedError


def flops_g(model: "nn.Module", resolution: int) -> float:
    raise NotImplementedError


def measure_latency_ms(
    model: "nn.Module",
    resolution: int,
    *,
    device: str,
    batch_size: int = 1,
    warmup: int = 20,
    iters: int = 100,
) -> float:
    raise NotImplementedError


def cost_scalar(result: RunResult, mode: str = "params") -> float:
    raise NotImplementedError
