#!/usr/bin/env python3
"""Импортировать архивы GuitarSet в манифест Qildo."""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

from qildo_beats.grid import Beat, write_grid, write_json

SOURCE = "GuitarSet v1.1.0 — https://zenodo.org/records/3371780"
LICENSE = "CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/"


def file_md5(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "md5").hexdigest()


def split_groups(groups: set[str]) -> dict[str, str]:
    ordered = sorted(groups, key=lambda group: hashlib.sha256(group.encode()).digest())
    if len(ordered) < 3:
        raise ValueError(
            "Для трёх выборок нужны хотя бы три независимые музыкальные основы"
        )
    held_out = max(1, len(ordered) // 5)
    return {
        group: "test"
        if index < held_out
        else "validation"
        if index < 2 * held_out
        else "train"
        for index, group in enumerate(ordered)
    }


def guitarset(annotations: Path, audio: Path, output: Path, verify: bool) -> None:
    if verify and (
        file_md5(annotations) != "b39b78e63d3446f2e54ddb7a54df9b10"
        or file_md5(audio) != "275966d6610ac34999b58426beb119c3"
    ):
        raise ValueError("Контрольные суммы архивов не совпадают с GuitarSet v1.1.0")
    output.mkdir(parents=True, exist_ok=True)
    excluded = []
    with ZipFile(annotations) as labels, ZipFile(audio) as waves:
        wavs = {
            Path(name).stem.removesuffix("_mic"): name
            for name in waves.namelist()
            if name.endswith("_mic.wav")
        }
        records = []
        for name in sorted(labels.namelist()):
            if not name.endswith(".jams"):
                continue
            track_id = Path(name).stem
            if track_id not in wavs:
                excluded.append({"id": track_id, "reason": "missing microphone WAV"})
                continue
            jams = json.loads(labels.read(name))
            layer = next(
                item
                for item in jams["annotations"]
                if item["namespace"] == "beat_position"
            )
            grid = [
                Beat(float(row["time"]), int(row["value"]["position"]))
                for row in layer["data"]
                if row["time"] < jams["file_metadata"]["duration"]
            ]
            if not grid:
                excluded.append({"id": track_id, "reason": "empty beat annotation"})
                continue
            # Например, Rock1 — одна основа для разных игроков, тональностей и темпов.
            group = track_id.split("_")[1].split("-")[0]
            records.append((track_id, group, grid))
        splits = split_groups({group for _, group, _ in records})
        rows = []
        for track_id, group, grid in records:
            wav = output / "audio" / f"{track_id}.wav"
            reference = output / "reference" / f"{track_id}.json"
            wav.parent.mkdir(exist_ok=True)
            with waves.open(wavs[track_id]) as source, wav.open("wb") as target:
                shutil.copyfileobj(source, target)
            write_grid(reference, grid)
            rows.append(
                {
                    "id": track_id,
                    "group": f"guitarset/{group}",
                    "split": splits[group],
                    "audio": os.fspath(wav.relative_to(output)),
                    "reference": os.fspath(reference.relative_to(output)),
                    "source": SOURCE,
                    "license": LICENSE,
                    "tags": ["guitarset", "solo-instrument", "excerpt"],
                }
            )
    write_json(output / "manifest.json", rows)
    write_json(output / "excluded.json", excluded)
    print(
        f"Записей: {len(rows)}, исключено: {len(excluded)}. {output / 'manifest.json'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="annotation.zip")
    parser.add_argument("audio", type=Path, help="audio_mono-mic.zip")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--skip-checksum", action="store_true", help="Для локальных подмножеств"
    )
    args = parser.parse_args()
    try:
        guitarset(args.annotations, args.audio, args.output, not args.skip_checksum)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        sys.exit(f"Ошибка импорта: {error}")


if __name__ == "__main__":
    main()
