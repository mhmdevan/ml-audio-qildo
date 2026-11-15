"""Манифест для обучения и оценки. Все пути относятся к файлу манифеста."""

import json
from dataclasses import dataclass
from pathlib import Path

from .grid import Beat, read_grid

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Track:
    track_id: str
    group: str
    split: str
    audio: Path
    reference: Path
    source: str
    license: str
    tags: tuple[str, ...]

    def grid(self) -> list[Beat]:
        return read_grid(self.reference)


def load_manifest(path: Path) -> list[Track]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Манифест должен содержать непустой массив треков")
    tracks = []
    ids = set()
    groups = {}
    audio_splits = {}
    for row in rows:
        try:
            track = Track(
                row["id"],
                row["group"],
                row["split"],
                (path.parent / row["audio"]).resolve(),
                (path.parent / row["reference"]).resolve(),
                row["source"],
                row["license"],
                tuple(row.get("tags", [])),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"Неполная запись манифеста: {row!r}") from error
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                track.track_id,
                track.group,
                track.source,
                track.license,
                *track.tags,
            )
        ) or isinstance(row.get("tags"), str):
            raise ValueError(f"Некорректная запись манифеста: {row!r}")
        if track.split not in SPLITS or track.track_id in ids:
            raise ValueError(f"Некорректный split или повтор id: {track.track_id}")
        if (
            groups.setdefault(track.group, track.split) != track.split
            or audio_splits.setdefault(track.audio, track.split) != track.split
        ):
            raise ValueError(f"Утечка между выборками: {track.group} ({track.audio})")
        ids.add(track.track_id)
        tracks.append(track)
    return tracks
