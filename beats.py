#!/usr/bin/env python3
"""Построить сетку долей для одного WAV."""

import argparse
import os
from importlib.util import find_spec
from pathlib import Path

from qildo_beats.grid import write_grid
from qildo_beats.ranker import load_ranker
from qildo_beats.tracker import DECODERS, TrackerConfig, track_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--checkpoint", type=Path, help="Веса собственного обучения")
    weights.add_argument(
        "--no-model", action="store_true", help="Только акустический трекер"
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--ranker", type=Path, help="JSON калибровки сетки для выбранных весов"
    )
    parser.add_argument("--decoder", default="hybrid", choices=DECODERS)
    parser.add_argument("--peak-threshold", type=float, default=0.35)
    parser.add_argument("--meters", type=int, nargs="+", default=[4, 3])
    args = parser.parse_args()
    if not 0 < args.peak_threshold <= 1 or any(meter < 2 for meter in args.meters):
        parser.error("Порог должен быть в (0, 1], число долей в такте — от 2")
    if args.audio.resolve() == args.output.resolve():
        parser.error("Пути входного WAV и выходного JSON должны различаться")
    config = TrackerConfig(tuple(args.meters), args.decoder, args.peak_threshold)
    checkpoint = args.checkpoint
    ranker_path = args.ranker
    try:
        if checkpoint is None and not args.no_model:
            enabled = os.environ.get("QILDO_DEFAULT_MODEL")
            if enabled == "1" or (enabled is None and find_spec("torch") is not None):
                bundled = Path(__file__).resolve().parent / "checkpoints"
                checkpoint = bundled / "quality.pt"
                if ranker_path is None and args.decoder in ("hybrid", "joint"):
                    ranker_path = bundled / "quality-ranker.json"
        if ranker_path and args.decoder not in ("hybrid", "joint"):
            raise ValueError("Ранжировщик используется в режимах hybrid и joint")
        ranker = load_ranker(ranker_path, checkpoint) if ranker_path else None
        grid, duration = track_file(args.audio, config, checkpoint, args.device, ranker)
        write_grid(args.output, grid, duration)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
