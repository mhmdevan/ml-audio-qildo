"""Импорт Groove MIDI Dataset с разбиением по барабанщикам."""

import csv
import warnings
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
from os.path import relpath
from pathlib import Path
from zipfile import ZipFile

from scipy.io import wavfile

from .dataset import SPLITS
from .grid import Beat, write_grid, write_json

GMD_SOURCE = (
    "Groove MIDI Dataset v1.0.0 — https://magenta.withgoogle.com/datasets/groove"
)
GMD_LICENSE = "CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/"


def midi_grid(payload: bytes, duration: float) -> list[Beat]:
    import mido

    midi = mido.MidiFile(file=BytesIO(payload))
    duration = min(duration, midi.length)
    meter, denominator, tempo = 4, 4, 500_000
    absolute_tick = 0
    for event in mido.merge_tracks(midi.tracks):
        absolute_tick += event.time
        if event.type not in ("time_signature", "set_tempo"):
            continue
        if absolute_tick != 0:
            raise ValueError("Для импорта нужна постоянная MIDI-сетка")
        if event.type == "time_signature":
            meter, denominator = event.numerator, event.denominator
        else:
            tempo = event.tempo
    if denominator != 4 or meter not in (3, 4, 5):
        raise ValueError("Ожидается 3/4, 4/4 или 5/4")
    period = mido.tick2second(midi.ticks_per_beat, midi.ticks_per_beat, tempo)
    first_note = next(
        (
            time
            for time, event in _timed_messages(midi)
            if event.type == "note_on" and event.velocity
        ),
        duration,
    )
    return [
        Beat(index * period, index % meter + 1)
        for index in range(int(duration / period) + 1)
        if first_note - 0.12 <= index * period < duration
    ]


def _timed_messages(midi):
    seconds = 0.0
    for event in midi:
        seconds += event.time
        yield seconds, event


def import_groove(
    midi_zip: Path, wav_root: Path, output: Path, selection: Path | None = None
) -> None:
    # drum patterns groove1–10 повторяются у разных барабанщиков и зарезервированы для теста.
    held_out = {"drummer8": "test", "drummer5": "validation", "drummer9": "validation"}
    caps = {"train": 180, "validation": 35, "test": 35}
    rows = defaultdict(list)
    missing = []
    excluded = []
    selected = None
    if selection is not None:
        with selection.open(encoding="utf-8", newline="") as file:
            selected = {row["id"]: row.get("split") for row in csv.DictReader(file)}
        if not selected or any(
            split and split not in SPLITS for split in selected.values()
        ):
            raise ValueError("Некорректный список треков или split")
    drummer_splits = {}
    with ZipFile(midi_zip) as archive:
        metadata = csv.DictReader(archive.read("groove/info.csv").decode().splitlines())
        candidates = sorted(
            metadata,
            key=lambda row: (
                row["time_signature"] == "4-4",
                row["split"] == "test" and row["drummer"] != "drummer8",
                sha256(row["id"].encode()).digest(),
            ),
        )
        for row in candidates:
            if selected is not None and row["id"] not in selected:
                continue
            split = held_out.get(row["drummer"], "train")
            if selected is not None:
                split = selected[row["id"]] or split
            if (
                (selected is None and len(rows[split]) >= caps[split])
                or row["beat_type"] != "beat"
                or float(row["duration"]) < 12
                or row["time_signature"] not in {"3-4", "4-4", "5-4"}
                or (
                    "groove" in row["style"]
                    and row["session"].endswith("/eval_session")
                    and split != "test"
                )
            ):
                continue
            if drummer_splits.setdefault(row["drummer"], split) != split:
                raise ValueError(f"Барабанщик в нескольких выборках: {row['drummer']}")
            path = wav_root / row["audio_filename"]
            if not path.is_file():
                missing.append(row["id"])
                continue
            payload = archive.read("groove/" + row["midi_filename"])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", wavfile.WavFileWarning)
                    rate, wav = wavfile.read(path)
                grid = midi_grid(payload, len(wav) / rate)
                if len(grid) < 8:
                    raise ValueError("менее восьми долей")
            except (ValueError, EOFError) as error:
                excluded.append({"id": row["id"], "reason": str(error)})
                continue
            track_id = "gmd-" + row["id"].replace("/", "-")
            reference = output / "reference" / f"{track_id}.json"
            write_grid(reference, grid)
            rows[split].append(
                {
                    "id": track_id,
                    "group": f"gmd/{row['drummer']}",
                    "split": split,
                    "audio": relpath(path, output),
                    "reference": relpath(reference, output),
                    "source": GMD_SOURCE,
                    "license": GMD_LICENSE,
                    "tags": ["gmd", "drums", row["time_signature"].replace("-", "/")],
                }
            )
    if selected is None and any(not rows[split] for split in caps):
        raise ValueError("Нужны записи Groove MIDI для train, validation и test")
    if selected is not None and (
        missing or excluded or sum(map(len, rows.values())) != len(selected)
    ):
        raise ValueError("Не удалось воспроизвести все записи из файла selection")
    write_json(output / "manifest.json", [row for split in caps for row in rows[split]])
    write_json(output / "excluded.json", excluded)
    print(
        f"Groove MIDI: {dict((k, len(rows[k])) for k in caps)}; отсутствующих WAV: {len(missing)}"
    )
