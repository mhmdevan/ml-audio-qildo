"""Управляемые записи для регрессий трекера и замера инференса."""

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from qildo_beats.grid import Beat


def make_song(
    path: Path,
    *,
    meter: int = 4,
    pickup: int = 0,
    seconds: float = 12,
    soft_intro: bool = False,
    change_tempo: bool = False,
) -> list[Beat]:
    sample_rate = 44_100
    samples = np.zeros((round(seconds * sample_rate), 2), dtype=np.float32)
    time = 0.47
    grid = []
    index = 0
    while time < seconds - 0.12:
        position = (index - pickup) % meter + 1
        grid.append(Beat(round(time, 4), position))
        length = round(0.10 * sample_rate)
        local = np.arange(length) / sample_rate
        offset = round(time * sample_rate)
        gentle = soft_intro and index < 2 * meter
        fundamental = (
            (130 if gentle else 62) if position == 1 else (325 if gentle else 242)
        )
        amplitude = (0.23 if position == 1 else 0.11) * (0.18 if gentle else 1)
        hit = amplitude * np.cos(2 * np.pi * fundamental * local) * np.exp(-local * 55)
        samples[offset : offset + length, 0] += hit
        samples[offset : offset + length, 1] -= hit * 0.9
        interval = 0.60 if not change_tempo or time < seconds * 0.55 else 0.50
        time += interval
        index += 1
    wavfile.write(path, sample_rate, (samples * 32767).astype(np.int16))
    return grid
