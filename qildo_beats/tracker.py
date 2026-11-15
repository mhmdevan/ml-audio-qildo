"""Единый вход для алгоритмического и обученного трекера."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import FRAME_RATE, Features, extract_features
from .decoder import (
    activation_peaks,
    anchor_coverage,
    make_grid,
    pick_pulse,
    track_pulse,
)
from .grid import Beat
from .meter import track_meter
from .periodic import recover_periodic_grid
from .ranker import GridRanker
from .selection import hypotheses

DECODERS = ("hybrid", "joint", "sequence", "peaks")


@dataclass(frozen=True)
class TrackerConfig:
    meters: tuple[int, ...] = (4, 3)
    decoder: str = "hybrid"
    peak_threshold: float = 0.35


def _acoustic_grid_is_stable(features: Features, grid: list[Beat]) -> bool:
    if len(grid) < 8:
        return False
    frames = np.rint([event.time * FRAME_RATE for event in grid]).astype(int)
    intervals = np.diff(frames)
    stable = np.abs(np.diff(intervals)) <= np.maximum(
        2, 0.055 * np.minimum(intervals[:-1], intervals[1:])
    )
    anchors = activation_peaks(features.onset, 0.27, 8)
    return bool(
        len(anchors)
        and stable.mean() >= 0.55
        and features.onset[frames].mean() >= 0.72
        and anchor_coverage(frames, anchors, features.onset[anchors]) >= 0.82
    )


def predict(
    features: Features,
    config: TrackerConfig,
    activations: np.ndarray | None = None,
    ranker: GridRanker | None = None,
) -> list[Beat]:
    if not features.audible.any():
        return []
    acoustic = None
    if config.decoder == "hybrid" or (ranker is not None and ranker.include_acoustic):
        acoustic = track_meter(
            features.onset,
            features.accent,
            config.meters,
            features.duration,
            learned=False,
        )
    if config.decoder == "hybrid" and acoustic is not None:
        if _acoustic_grid_is_stable(features, acoustic):
            return acoustic
        if activations is None:
            return recover_periodic_grid(features, config.meters, acoustic) or acoustic
    if activations is None:
        pulse, accent = features.onset, features.accent
    else:
        pulse, accent = np.where(features.audible[None, :], activations, 0)
    if config.decoder in ("hybrid", "joint"):
        if ranker is not None and activations is not None:
            acoustic_options = acoustic if ranker.include_acoustic else None
            ranked = ranker.best(
                hypotheses(features, activations, config.meters, acoustic_options)
            )
            if ranked is not None:
                return ranked
            if acoustic_options:
                return (
                    recover_periodic_grid(features, config.meters, acoustic_options)
                    or acoustic_options
                )
        return track_meter(
            pulse,
            accent,
            config.meters,
            features.duration,
            learned=activations is not None,
        )
    if config.decoder == "peaks":
        frames = pick_pulse(pulse, config.peak_threshold)
    else:
        rhythm = 0.78 * pulse + 0.22 * accent
        frames = track_pulse(pulse, rhythm)
    return make_grid(frames, accent, config.meters, features.duration)


def track_file(
    path: Path,
    config: TrackerConfig,
    checkpoint: Path | None = None,
    device: str = "cpu",
    ranker: GridRanker | None = None,
) -> tuple[list[Beat], float]:
    features = extract_features(path)
    activations = None
    if checkpoint is not None:
        from .model import load_predictor

        activations = load_predictor(checkpoint, device)(features.log_mel)
    return predict(features, config, activations, ranker), features.duration
