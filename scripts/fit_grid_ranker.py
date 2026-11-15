#!/usr/bin/env python3
"""Калибровка оценки ритмической сетки на записях validation."""

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qildo_beats.audio import extract_features
from qildo_beats.dataset import load_manifest
from qildo_beats.grid import write_json
from qildo_beats.meter import track_meter
from qildo_beats.metrics import score_track
from qildo_beats.model import load_predictor
from qildo_beats.ranker import BASE_WEIGHTS, RANKER_VERSION, checkpoint_digest
from qildo_beats.selection import FEATURE_NAMES, hypotheses

METRICS = ("beat_f1", "downbeat_f1", "cmlt", "first_correct", "tempo_error_rate")
UTILITY = np.array([0.40, 0.30, 0.25, 0.05, -0.25])
METERS = (4, 3)


@dataclass(frozen=True)
class Example:
    domain: str
    group: str
    features: np.ndarray
    metrics: np.ndarray


def fit(examples: list[Example], regularization: float) -> np.ndarray:
    by_domain = Counter(example.domain for example in examples)
    size = sum(len(example.features) for example in examples)
    features = np.empty((size, len(FEATURE_NAMES)))
    target = np.empty(size)
    sample_weights = np.empty(size)
    offset = 0
    for row in examples:
        end = offset + len(row.features)
        features[offset:end] = row.features - row.features.mean(axis=0)
        utility = np.einsum("ij,j->i", row.metrics, UTILITY)
        target[offset:end] = utility - utility.mean()
        sample_weights[offset:end] = 1 / (by_domain[row.domain] * len(row.features))
        offset = end
    scale = np.sqrt(np.average(features**2, axis=0, weights=sample_weights)).clip(0.01)
    sample_weights /= sample_weights.sum()
    normalized = features / scale
    gram = np.einsum("ni,nj,n->ij", normalized, normalized, sample_weights)
    covariance = np.einsum("ni,n,n->i", normalized, target, sample_weights)
    return (
        np.linalg.solve(gram + regularization * np.eye(len(scale)), covariance) / scale
    )


def mean_scores(examples: list[Example], weights: np.ndarray) -> np.ndarray:
    return np.mean(
        [
            item.metrics[np.einsum("ij,j->i", item.features, weights).argmax()]
            for item in examples
        ],
        axis=0,
    )


def calibrate(args: argparse.Namespace) -> None:
    model = load_predictor(args.checkpoint, args.device)
    examples = []
    seen = set()
    for manifest in args.manifests:
        for track in load_manifest(manifest):
            if track.split != "validation":
                continue
            if track.track_id in seen:
                raise ValueError(f"Повтор трека при калибровке: {track.track_id}")
            seen.add(track.track_id)
            features = extract_features(track.audio)
            events = track.grid()
            acoustic = (
                track_meter(
                    features.onset,
                    features.accent,
                    METERS,
                    features.duration,
                    learned=False,
                )
                if args.include_acoustic
                else None
            )
            candidates = hypotheses(features, model(features.log_mel), METERS, acoustic)
            if len(candidates) < 2 or not any(event.beat == 1 for event in events):
                continue
            scores = [score_track(events, option.grid) for option in candidates]
            for score in scores:
                score["tempo_error_rate"] = score["tempo_error_bars"] / max(
                    1, score["stable_bars"]
                )
            examples.append(
                Example(
                    str(manifest),
                    track.group,
                    np.stack([option.features for option in candidates]),
                    np.array(
                        [[score[key] for key in METRICS] for score in scores],
                        dtype=float,
                    ),
                )
            )
        print(f"{manifest}: {len(examples)} записей суммарно", flush=True)
    groups = sorted({row.group for row in examples})
    if len(groups) < 3:
        raise ValueError("Для калибровки нужны минимум три независимые группы")
    weights = fit(examples, args.regularization)
    comparison = []
    for group in groups:
        learning = [example for example in examples if example.group != group]
        held_out = [example for example in examples if example.group == group]
        comparison.append(
            {
                "group": group,
                "domain": held_out[0].domain,
                "tracks": len(held_out),
                "baseline": mean_scores(held_out, BASE_WEIGHTS).tolist(),
                "ranked": mean_scores(
                    held_out, fit(learning, args.regularization)
                ).tolist(),
            }
        )
    write_json(
        args.output,
        {
            "version": RANKER_VERSION,
            "checkpoint_sha256": checkpoint_digest(args.checkpoint),
            "features": FEATURE_NAMES,
            "weights": weights.tolist(),
            "include_acoustic": args.include_acoustic,
            "regularization": args.regularization,
            "manifests": [str(name) for name in args.manifests],
            "tracks": len(examples),
            "metrics": METRICS,
            "utility": UTILITY.tolist(),
            "validation_by_group": comparison,
        },
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("checkpoint", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("manifests", type=Path, nargs="+")
    cli.add_argument("--regularization", type=float, default=0.10)
    cli.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    cli.add_argument("--include-acoustic", action="store_true")
    args = cli.parse_args()
    if not np.isfinite(args.regularization) or args.regularization <= 0:
        cli.error("Регуляризация должна быть конечным положительным числом")
    try:
        calibrate(args)
    except (OSError, ValueError, RuntimeError) as error:
        cli.exit(1, f"Ошибка калибровки: {error}\n")
