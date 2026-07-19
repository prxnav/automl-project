from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader


def evaluate(model: "nn.Module", loader: "DataLoader", *, device: str) -> tuple[float, float, float]:
    raise NotImplementedError
