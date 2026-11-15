#!/usr/bin/env python3
"""Оценка первых долей на 24 записях HU33 с ручной разметкой границ тактов."""

import argparse
import csv
from pathlib import Path
from statistics import mean

import numpy as np
from mir_eval.beat import f_measure

from qildo_beats.audio import extract_features
from qildo_beats.grid import write_json
from qildo_beats.metrics import TOLERANCE
from qildo_beats.ranker import checkpoint_digest, load_ranker
from qildo_beats.tracker import TrackerConfig, predict


def measure_starts(path: Path, duration: float) -> np.ndarray:
    with path.open(encoding="utf-8-sig", newline="") as file:
        # В отдельной версии первой песни CSV без заголовка и с запятой.
        header = file.readline().strip()
        if header == "start;measure":
            rows = list(csv.reader(file, delimiter=";"))
        else:
            file.seek(0)
            rows = list(csv.reader(file))
    starts = [
        float(time)
        for time, measure in rows
        if (bar := float(measure)) >= 1 and bar.is_integer()
    ]
    times = np.asarray(starts, dtype=float)
    if (
        not len(times)
        or not np.isfinite(times).all()
        or not (0 <= times[0] and times[-1] < duration)
        or (np.diff(times) <= 0).any()
    ):
        raise ValueError(f"Некорректная разметка тактов: {path}")
    return times


def evaluate(args: argparse.Namespace) -> None:
    predictor = None
    if args.checkpoint:
        from qildo_beats.model import load_predictor

        predictor = load_predictor(args.checkpoint, "cpu")
    ranker = load_ranker(args.ranker, args.checkpoint) if args.ranker else None
    config = TrackerConfig(tuple(args.meters))
    with (args.root / "ann_score_overview.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        signatures = {
            int(row["SongID"]): row["TimeSign"]
            for row in csv.DictReader(file, delimiter=";")
        }
    results = []
    for number in range(1, 25):
        name = f"Schubert_D911-{number:02d}_HU33"
        features = extract_features(args.root / f"{name}.wav")
        reference = measure_starts(args.root / f"{name}.csv", features.duration)
        activations = predictor(features.log_mel) if predictor else None
        grid = predict(features, config, activations, ranker)
        downbeats = np.array(
            [beat.time for beat in grid if beat.beat == 1], dtype=float
        )
        results.append(
            {
                "id": name,
                "time_signature": signatures[number],
                "duration": features.duration,
                "downbeat_f1": f_measure(reference, downbeats, TOLERANCE)
                if len(downbeats)
                else 0.0,
                "first_downbeat_correct": bool(
                    len(downbeats) and abs(downbeats[0] - reference[0]) <= TOLERANCE
                ),
            }
        )
    summary = {
        "tracks": len(results),
        "downbeat_f1": mean(row["downbeat_f1"] for row in results),
        "first_downbeat_correct_tracks": sum(
            row["first_downbeat_correct"] for row in results
        ),
    }
    quarter_meters = [row for row in results if row["time_signature"] in ("3/4", "4/4")]
    summary["simple_3_or_4"] = {
        "tracks": len(quarter_meters),
        "downbeat_f1": mean(row["downbeat_f1"] for row in quarter_meters),
    }
    write_json(
        args.report,
        {
            "summary": summary,
            "per_track": results,
            "tolerance_seconds": TOLERANCE,
            "meters": config.meters,
            "checkpoint_sha256": checkpoint_digest(args.checkpoint)
            if args.checkpoint
            else None,
            "ranker_sha256": checkpoint_digest(args.ranker) if args.ranker else None,
        },
    )
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--ranker", type=Path)
    parser.add_argument("--meters", type=int, nargs="+", default=[4, 3])
    args = parser.parse_args()
    try:
        evaluate(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Ошибка оценки: {error}\n")
