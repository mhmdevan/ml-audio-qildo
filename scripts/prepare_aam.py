#!/usr/bin/env python3
"""Импорт полных миксов AAM по зафиксированному списку композиций."""

import argparse
import csv
from os.path import relpath
from pathlib import Path
from zipfile import ZipFile

from scipy.io import wavfile

from qildo_beats.grid import Beat, write_grid, write_json

SOURCE = "AAM v1.1.0 — https://zenodo.org/records/5794629"
LICENSE = "CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/"


def parse_beats(payload: bytes, duration: float) -> list[Beat]:
    entries = (
        line
        for line in payload.decode("utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith(("@", "%"))
    )
    candidates = sorted(
        (
            Beat(float(time), int(quarter))
            for time, _, quarter, _ in csv.reader(entries, quotechar="'")
            if 0 <= float(time) < duration
        ),
        key=lambda event: event.time,
    )
    grid = []
    for event in candidates:
        # В местах стыка секций ARFF иногда содержит две копии одного такта.
        if grid and event.time - grid[-1].time < 0.001:
            if event.beat != grid[-1].beat:
                raise ValueError(f"Несовместимые номера долей около {event.time:.3f} с")
            continue
        grid.append(event)
    if not grid:
        raise ValueError("Пустая разметка долей AAM")
    return grid


def prepare(annotations: Path, audio_root: Path, selection: Path, output: Path) -> None:
    tracks = []
    with (
        selection.open(encoding="utf-8", newline="") as selected,
        ZipFile(annotations) as archive,
    ):
        for row in csv.DictReader(selected):
            identity = row["id"]
            if not (len(identity) == 4 and identity.isascii() and identity.isdigit()):
                raise ValueError(f"Некорректный ID AAM: {identity}")
            audio = audio_root / f"{identity}_mix.wav"
            rate, samples = wavfile.read(audio, mmap=True)
            duration = len(samples) / rate
            grid = parse_beats(archive.read(f"{identity}_beatinfo.arff"), duration)
            reference = output / "reference" / f"{identity}.json"
            write_grid(reference, grid, duration)
            tracks.append(
                {
                    "id": f"aam-{identity}",
                    "group": f"aam/{identity}",
                    "split": row["split"],
                    "audio": relpath(audio, output),
                    "reference": relpath(reference, output),
                    "source": SOURCE,
                    "license": LICENSE,
                    "tags": ["aam", "synthetic", "full-mix"],
                }
            )
    write_json(output / "manifest.json", tracks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("annotations", "audio_root", "selection", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    try:
        prepare(args.annotations, args.audio_root, args.selection, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Ошибка импорта AAM: {error}\n")
