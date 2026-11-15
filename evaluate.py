#!/usr/bin/env python3
"""Оценить предсказания на полной песне или на части манифеста."""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

from qildo_beats.audio import extract_features
from qildo_beats.dataset import SPLITS, load_manifest
from qildo_beats.grid import read_grid, validate_grid, write_grid, write_json
from qildo_beats.metrics import score_track, summarize
from qildo_beats.ranker import checkpoint_digest, load_ranker
from qildo_beats.tracker import DECODERS, TrackerConfig, predict


def compare(groups: list[str], rows: list[dict], baselines: list[dict]) -> dict:
    rng = np.random.default_rng(2026)
    comparison = {}
    for metric in ("beat_f1", "downbeat_f1", "cmlt", "amlt"):
        by_group: dict[str, list[float]] = {}
        for group, row, baseline in zip(groups, rows, baselines, strict=True):
            if row[metric] is not None and baseline[metric] is not None:
                by_group.setdefault(group, []).append(row[metric] - baseline[metric])
        means = np.array([np.mean(values) for values in by_group.values()])
        interval = None
        if len(means) >= 2:
            sampled = rng.choice(means, (2000, len(means)), replace=True).mean(axis=1)
            interval = np.quantile(sampled, [0.025, 0.975]).tolist()
        comparison[metric] = {
            "mean_group_delta": float(means.mean()) if len(means) else None,
            "groups": len(means),
            "ci95": interval,
        }
    return comparison


def evaluate_manifest(args: argparse.Namespace) -> None:
    tracks = [
        track for track in load_manifest(args.manifest) if track.split == args.split
    ]
    if not tracks:
        raise ValueError(f"Выборка {args.split} пуста")
    if args.predictions and args.checkpoint:
        raise ValueError("Используйте либо готовые предсказания, либо веса")
    if args.ranker and args.decoder not in ("hybrid", "joint"):
        raise ValueError("Ранжировщик используется в режимах hybrid и joint")
    if (
        args.compare
        and args.save_predictions
        and (args.compare.resolve() == args.save_predictions.resolve())
    ):
        raise ValueError("Каталоги baseline и выходных предсказаний должны различаться")
    predictor = None
    if args.checkpoint:
        from qildo_beats.model import load_predictor

        predictor = load_predictor(args.checkpoint, args.device)
    ranker = load_ranker(args.ranker, args.checkpoint) if args.ranker else None
    config = TrackerConfig(tuple(args.meters), args.decoder, args.peak_threshold)
    rows, baselines = [], []
    for track in tracks:
        reference = track.grid()
        # Идентификатор может включать имя композиции, но не путь.
        if Path(track.track_id).name != track.track_id or track.track_id in (".", ".."):
            raise ValueError(f"Некорректный id трека: {track.track_id}")
        name = f"{track.track_id}.json"
        if args.predictions:
            result = read_grid(args.predictions / name)
        else:
            features = extract_features(track.audio)
            validate_grid(reference, features.duration)
            activations = predictor(features.log_mel) if predictor else None
            result = predict(features, config, activations, ranker)
        rows.append(
            {
                "id": track.track_id,
                "group": track.group,
                "tags": track.tags,
                **score_track(reference, result),
            }
        )
        if args.save_predictions:
            write_grid(args.save_predictions / name, result)
        if args.compare:
            baselines.append(score_track(reference, read_grid(args.compare / name)))
    report = {
        "split": args.split,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "config": {
            "predictions": str(args.predictions) if args.predictions else None,
            "decoder": config.decoder,
            "meters": config.meters,
            "peak_threshold": config.peak_threshold,
            "ranker_sha256": checkpoint_digest(args.ranker) if args.ranker else None,
            "checkpoint_sha256": hashlib.sha256(
                args.checkpoint.read_bytes()
            ).hexdigest()
            if args.checkpoint
            else None,
        },
        "summary": summarize(rows),
        "per_track": rows,
        "by_tag": {
            tag: summarize([row for row in rows if tag in row["tags"]])
            for tag in sorted({tag for row in rows for tag in row["tags"]})
        },
    }
    if baselines:
        report["baseline"] = summarize(baselines)
        report["paired"] = compare([track.group for track in tracks], rows, baselines)
        eligible = candidate_errors = baseline_errors = 0
        for candidate, baseline in zip(rows, baselines, strict=True):
            common = set(candidate["phase_eligible_indices"]) & set(
                baseline["phase_eligible_indices"]
            )
            eligible += len(common)
            candidate_errors += len(
                common.intersection(candidate["phase_error_indices"])
            )
            baseline_errors += len(common.intersection(baseline["phase_error_indices"]))
        report["common_phase_bars"] = {
            "eligible": eligible,
            "candidate_errors": candidate_errors,
            "baseline_errors": baseline_errors,
        }
    write_json(args.report, report)
    print(report["summary"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("track")
    single.add_argument("reference", type=Path)
    single.add_argument("prediction", type=Path)
    single.add_argument("report", type=Path)
    batch = commands.add_parser("manifest")
    batch.add_argument("manifest", type=Path)
    batch.add_argument("report", type=Path)
    batch.add_argument("--split", choices=SPLITS, default="validation")
    batch.add_argument("--checkpoint", type=Path)
    batch.add_argument("--ranker", type=Path)
    batch.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    batch.add_argument("--decoder", choices=DECODERS, default="hybrid")
    batch.add_argument("--peak-threshold", type=float, default=0.35)
    batch.add_argument("--meters", type=int, nargs="+", default=[4, 3])
    batch.add_argument("--predictions", type=Path)
    batch.add_argument("--save-predictions", type=Path)
    batch.add_argument("--compare", type=Path, help="Каталог предсказаний baseline")
    args = parser.parse_args()
    if args.command == "manifest" and (
        not 0 < args.peak_threshold <= 1 or min(args.meters) < 2
    ):
        parser.error("Порог должен быть в (0, 1], число долей в такте — от 2")
    try:
        if args.command == "track":
            write_json(
                args.report,
                score_track(read_grid(args.reference), read_grid(args.prediction)),
            )
        else:
            evaluate_manifest(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        sys.exit(f"Ошибка оценки: {error}")


if __name__ == "__main__":
    main()
