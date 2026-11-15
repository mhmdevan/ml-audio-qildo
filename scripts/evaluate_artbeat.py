#!/usr/bin/env python3
"""Проверка пульса на размеченных ритмических задачах ARTBeaT."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from mir_eval.beat import continuity, f_measure

from qildo_beats.audio import extract_features
from qildo_beats.grid import write_json
from qildo_beats.metrics import TOLERANCE
from qildo_beats.ranker import checkpoint_digest, load_ranker
from qildo_beats.tracker import TrackerConfig, predict


def summarize(rows: list[dict]) -> dict:
    return {
        "tracks": len(rows),
        "beat_f1": mean(row["beat_f1"] for row in rows),
        "cmlt": mean(row["cmlt"] for row in rows),
        "amlt": mean(row["amlt"] for row in rows),
    }


def evaluate(args: argparse.Namespace) -> None:
    predictor = None
    if args.checkpoint:
        from qildo_beats.model import load_predictor

        predictor = load_predictor(args.checkpoint, "cpu")
    ranker = load_ranker(args.ranker, args.checkpoint) if args.ranker else None
    config = TrackerConfig()
    rows = []
    categories = defaultdict(list)
    root = "ARTBeaT_Dataset"
    with ZipFile(args.archive) as archive, TemporaryDirectory() as work:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith(f"{root}/Audio/") and name.endswith(".wav")
        )
        if len(members) != 25:
            raise ValueError("В архиве ARTBeaT ожидается 25 WAV")
        for member in members:
            name = Path(member).stem
            number, _, category, _ = name.split("_", 3)
            if number == "01":  # Калибровочный метроном.
                continue
            audio = Path(work) / "track.wav"
            audio.write_bytes(archive.read(member))
            features = extract_features(audio)
            annotation = archive.read(f"{root}/Annotations/gt_csv/{name}_gt.csv")
            reference = np.fromstring(annotation.decode("ascii"), sep="\n")
            reference = reference[(0 <= reference) & (reference < features.duration)]
            if len(reference) < 2 or (np.diff(reference) <= 0).any():
                raise ValueError(f"Некорректный эталон ARTBeaT: {name}")
            activations = predictor(features.log_mel) if predictor else None
            grid = predict(features, config, activations, ranker)
            estimate = np.array([event.time for event in grid], dtype=float)
            scores = continuity(reference, estimate) if len(estimate) else (0,) * 4
            result = {
                "id": name,
                "beat_f1": f_measure(reference, estimate, TOLERANCE)
                if len(estimate)
                else 0.0,
                "cmlt": scores[1],
                "amlt": scores[3],
            }
            rows.append(result)
            categories[category].append(result)
    report = {
        "summary": summarize(rows),
        "by_category": {key: summarize(group) for key, group in categories.items()},
        "per_track": rows,
        "tolerance_seconds": TOLERANCE,
        "checkpoint_sha256": checkpoint_digest(args.checkpoint)
        if args.checkpoint
        else None,
        "ranker_sha256": checkpoint_digest(args.ranker) if args.ranker else None,
    }
    write_json(args.report, report)
    print(report["summary"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--ranker", type=Path)
    args = parser.parse_args()
    try:
        evaluate(args)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        parser.exit(1, f"Ошибка оценки: {error}\n")
