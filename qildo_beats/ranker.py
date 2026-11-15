"""Версионированное ранжирование сеток, откалиброванное на валидации."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .grid import Beat
from .selection import FEATURE_NAMES, Hypothesis

RANKER_VERSION = 1
BASE_WEIGHTS = np.array([1.0, 0.22, 1.25, *([0.0] * (len(FEATURE_NAMES) - 3))])


def checkpoint_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class GridRanker:
    weights: np.ndarray
    include_acoustic: bool = False

    def best(self, options: list[Hypothesis]) -> list[Beat] | None:
        if not options:
            return None
        features = np.stack([option.features for option in options])
        index = np.einsum("ij,j->i", features, self.weights).argmax()
        return options[index].grid


def load_ranker(path: Path, checkpoint: Path | None) -> GridRanker:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("version") != RANKER_VERSION:
        raise ValueError(f"Неизвестный формат ранжировщика: {path}")
    if config.get("features") != list(FEATURE_NAMES):
        raise ValueError(
            f"Ранжировщик использует несовместимый набор признаков: {path}"
        )
    if checkpoint is None or config.get("checkpoint_sha256") != checkpoint_digest(
        checkpoint
    ):
        raise ValueError(f"Ранжировщик требует исходный CNN-чекпойнт: {path}")
    weights = np.asarray(config.get("weights"), dtype=float)
    if weights.shape != (len(FEATURE_NAMES),) or not np.isfinite(weights).all():
        raise ValueError(f"Некорректные веса ранжировщика: {path}")
    include_acoustic = config.get("include_acoustic", False)
    if type(include_acoustic) is not bool:
        raise ValueError(f"Некорректный список источников ранжировщика: {path}")
    return GridRanker(weights, include_acoustic)
