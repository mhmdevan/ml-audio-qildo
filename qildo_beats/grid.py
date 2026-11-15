"""Контракт сетки и запись JSON."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile


@dataclass(frozen=True)
class Beat:
    time: float
    beat: int


def validate_grid(beats: list[Beat], duration: float = math.inf) -> None:
    previous = -1.0
    for event in beats:
        if (
            type(event.time) not in (int, float)
            or not math.isfinite(event.time)
            or not 0 <= event.time < duration
            or event.time <= previous
        ):
            raise ValueError(
                "Времена долей должны строго возрастать и быть в пределах аудио"
            )
        if type(event.beat) is not int or event.beat < 1:
            raise ValueError("Номер доли должен быть положительным целым числом")
        previous = event.time


def read_grid(path: Path) -> list[Beat]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Ожидается массив долей в {path}")
    if any(not isinstance(row, dict) or row.keys() != {"time", "beat"} for row in rows):
        raise ValueError(f"Ожидаются поля time и beat в {path}")
    beats = [Beat(**row) for row in rows]
    validate_grid(beats)
    return beats


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, mode="w", encoding="utf-8", delete=False
        ) as file:
            temp_path = Path(file.name)
            json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_grid(path: Path, beats: list[Beat], duration: float = math.inf) -> None:
    validate_grid(beats, duration)
    write_json(path, [asdict(beat) for beat in beats])
