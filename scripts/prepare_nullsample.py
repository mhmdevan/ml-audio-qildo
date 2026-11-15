#!/usr/bin/env python3
"""Подготовить полные синтетические миксы Nullsample для отдельной проверки."""

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from qildo_beats.grid import Beat, write_grid, write_json

REPOSITORY = "https://huggingface.co/datasets/Maqsudjonpolatov/nullsample-multitrack"
REVISION = "3b22000b2f3dffc69b9d7654ca62e53f1b153754"


def download(path: str) -> bytes:
    with urlopen(f"{REPOSITORY}/resolve/{REVISION}/{path}", timeout=60) as response:
        return response.read()


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = csv.DictReader(download("mix/metadata.csv").decode().splitlines())

    def convert(row: dict[str, str]) -> dict:
        seed = row["seed"]
        label = json.loads(download(f"annotations/{seed}.json"))
        if label["time_signature"] != "4/4":
            raise ValueError(f"Неожиданный размер в Nullsample: {seed}")
        audio = output / f"{seed}.wav"
        compressed = output / f"{seed}.mp3"
        if not audio.exists():
            if not compressed.exists():
                compressed.write_bytes(download(f"mix/{seed}.mp3"))
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(compressed),
                    "-c:a",
                    "pcm_s16le",
                    str(audio),
                ],
                check=True,
            )
        reference = output / f"{seed}.json"
        grid = [
            Beat(start + offset * 60 / label["tempo_bpm"], offset + 1)
            for start in label["downbeats"]
            for offset in range(4)
        ]
        write_grid(reference, grid)
        return {
            "id": seed,
            "group": f"nullsample/{seed}",
            "split": "test",
            "audio": audio.name,
            "reference": reference.name,
            "source": f"{REPOSITORY}/tree/{REVISION}",
            "license": "CC BY 4.0 — Nullsample (nullsample.maqsudjon.com)",
            "tags": ["external-synthetic", "electronic", "4/4"],
        }

    with ThreadPoolExecutor(max_workers=6) as workers:
        write_json(output / "manifest.json", list(workers.map(convert, rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output)
