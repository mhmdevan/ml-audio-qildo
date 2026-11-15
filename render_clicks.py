#!/usr/bin/env python3
"""Подготовить случайно переставленную пару WAV для слепого сравнения."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from qildo_beats.audio import read_wav
from qildo_beats.grid import Beat, read_grid, validate_grid, write_json


def render(
    samples: np.ndarray, rate: int, grid: list[Beat], start: float, end: float
) -> np.ndarray:
    validate_grid(grid, len(samples) / rate)
    first, last = round(start * rate), round(end * rate)
    # Одинаковый запас по амплитуде у обеих версий.
    audio = samples[first:last].copy() * (
        0.70 / max(float(np.abs(samples).max()), 1e-6)
    )
    t = np.arange(round(0.04 * rate)) / rate
    ordinary = (0.16 * np.sin(2 * np.pi * 950 * t) * np.exp(-100 * t)).astype(
        np.float32
    )
    accent = (0.26 * np.sin(2 * np.pi * 1500 * t) * np.exp(-80 * t)).astype(np.float32)
    for event in grid:
        offset = round(event.time * rate) - first
        click = accent if event.beat == 1 else ordinary
        lo, hi = max(0, offset), min(len(audio), offset + len(click))
        if hi > lo:
            audio[lo:hi] += click[lo - offset : hi - offset, None]
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--seed", type=int, required=True, help="Уникальное зерно для каждой пары"
    )
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    args = parser.parse_args()
    try:
        rate, samples = read_wav(args.audio)
        duration = len(samples) / rate
        end = args.end if args.end is not None else duration
        if not 0 <= args.start < end <= duration or round(args.start * rate) >= round(
            end * rate
        ):
            raise ValueError("Границы фрагмента должны находиться внутри записи")
        versions = [
            ("baseline", read_grid(args.baseline)),
            ("candidate", read_grid(args.candidate)),
        ]
        random.Random(args.seed).shuffle(versions)
        audio_pairs = [
            render(samples, rate, grid, args.start, end) for _, grid in versions
        ]
        args.output.mkdir(parents=True, exist_ok=True)
        for label, audio in zip(("A", "B"), audio_pairs, strict=True):
            wavfile.write(args.output / f"{label}.wav", rate, audio)
        write_json(
            args.output / "answer-key.json",
            {
                label: version
                for label, (version, _) in zip(("A", "B"), versions, strict=True)
            },
        )
    except (OSError, ValueError) as error:
        sys.exit(f"Ошибка подготовки прослушивания: {error}")


if __name__ == "__main__":
    main()
