#!/usr/bin/env python3
"""Сгенерировать оригинальные стереоаранжировки с точной сеткой долей."""

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from qildo_beats.grid import Beat, write_grid, write_json


def generate(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rate = 22050
    settings = [
        (4, 97, 0, "rock-steady"),
        (3, 69, 1, "waltz-solo"),
        (4, 126, 2, "pop-pickup"),
        (4, 83, 0, "soul-syncopated"),
        (4, 148, 0, "punk-fill"),
        (4, 110, 0, "funk-break"),
        (4, 72, 0, "ballad-tempo"),
        (3, 117, 0, "waltz-tempo"),
        (4, 105, 1, "jazz-rubato"),
        (4, 94, 0, "mixed-meter"),
        (3, 141, 2, "waltz-pickup"),
        (4, 113, 0, "electronic-no-drums"),
    ]
    rows = []
    for song, (meter, bpm, pickup, label) in enumerate(settings):
        rng = np.random.default_rng(8831 + song)
        length = 155 + song * 3
        mix = np.zeros((round(rate * length), 2), np.float32)
        tune = [0, 2, 3, 5, 7, 9, 10] if song % 2 else [0, 2, 4, 5, 7, 9, 11]
        motifs = [rng.choice(tune, 4, replace=True) for _ in range(4)]
        roots = rng.integers(38, 53, 10)
        pitches = np.array([0, 3 if song % 2 else 4, 7, 12])
        kits = []
        for j in range(5):
            duration = 0.20 if j != 2 else 0.09
            time = np.arange(round(rate * duration)) / rate
            noise = rng.uniform(-1, 1, len(time))
            if j == 0:
                sound = np.sin(
                    2
                    * np.pi
                    * ((45 + song) * time + 36 * (1 - np.exp(-35 * time)) / 35)
                ) * np.exp(-time * 26)
            elif j == 1:
                sound = (0.65 * noise + 0.35 * np.sin(2 * np.pi * 170 * time)) * np.exp(
                    -time * 27
                )
            elif j == 2:
                sound = np.r_[0, np.diff(noise)] * np.exp(-time * 42) * 0.6
            else:
                sound = (
                    np.sin(2 * np.pi * (100 + j * 63) * time) + 0.25 * noise
                ) * np.exp(-time * 22)
            kits.append(sound.astype(np.float32))

        def add(time, signal, gain, pan):
            idx = round(time * rate)
            count = min(len(signal), len(mix) - idx)
            if idx >= 0 and count > 0:
                mix[idx : idx + count] += signal[:count, None] * np.array(
                    [gain * (1 - pan), gain * (1 + pan)], np.float32
                )

        def notes(time, duration, midi, gain, pan, pluck=True):
            t = np.arange(round(duration * rate)) / rate
            wave = np.zeros(len(t))
            for midi_note in np.atleast_1d(midi):
                f = 440 * 2 ** ((midi_note - 69) / 12)
                wave += (
                    np.sin(2 * np.pi * f * t)
                    + 0.42 * np.sin(2 * np.pi * 2 * f * t)
                    + 0.23 * np.sin(2 * np.pi * 3 * f * t)
                )
            wave *= np.minimum(1, t / (0.006 if pluck else 0.07)) * np.exp(
                -t / (0.18 if pluck else 0.55)
            )
            add(time, wave / len(np.atleast_1d(midi)), gain, pan)

        beats = []
        current = 0.34 + song * 0.046
        beat = (meter - pickup) % meter + 1
        bar = 0
        i = 0
        while current < length - 0.9:
            if label == "mixed-meter" and beat == 1:
                meter = 3 if 12 <= bar < 24 or bar >= 38 else 4
            scale = 1
            if "tempo" in label:
                scale = 1.21 if current > length * 0.53 else 1
            if label == "jazz-rubato":
                scale = (
                    1 + 0.11 * math.sin(current / 12) + 0.035 * math.sin(current / 1.7)
                )
            period = 60 / bpm / scale
            beats.append(Beat(round(current, 6), beat))
            intro = bar < 3
            rest = label == "funk-break" and 14 <= bar < 16
            if not rest:
                root_note = roots[(bar // 2 + bar % 3 + song) % len(roots)]
                chord_gain = 0.12 if intro else 0.19
                if beat == 1 or (beat == 3 and song % 3 == 1):
                    notes(
                        current,
                        period * 1.5,
                        root_note + 12 + pitches,
                        chord_gain,
                        -0.28,
                        pluck=song % 4 != 2,
                    )
                else:
                    notes(
                        current,
                        period * 0.61,
                        root_note + 24 + motifs[bar % 4][(beat - 1) % 4],
                        chord_gain * 0.57,
                        -0.24,
                    )
                if song % 3 == 0 and i % 2:
                    notes(
                        current + period * 0.52,
                        period * 0.31,
                        root_note + 24 + motifs[bar % 4][beat % 4],
                        chord_gain * 0.42,
                        0.31,
                    )
                if not intro:
                    notes(
                        current + 0.005,
                        period * 0.68,
                        root_note + (7 if beat == 3 and song % 3 == 1 else 0),
                        0.17,
                        0.02,
                    )
                if not intro and label != "electronic-no-drums":
                    off = (0.006 if song % 2 else -0.009) + rng.uniform(-0.013, 0.013)
                    if beat == 1 or (beat == 3 and meter == 4):
                        add(current + off, kits[0], 0.27 if beat == 1 else 0.22, 0.03)
                    if beat in (2, 4):
                        add(current + off, kits[1], 0.22, 0.09)
                    for fraction in (0, 0.5 if song % 3 else 0.62):
                        add(
                            current + period * fraction + off,
                            kits[2],
                            0.06 if fraction else 0.09,
                            -0.23,
                        )
                    if bar % 8 == 7 and beat == meter:
                        add(current + period * 0.75, kits[4], 0.12, 0.3)
            current += period
            if beat == meter:
                beat = 1
                bar += 1
            else:
                beat += 1
            i += 1
        output = np.clip(mix * 0.82, -1, 1)
        audio = root / f"{label}.wav"
        wavfile.write(audio, rate, (output * 32767).astype(np.int16))
        write_grid(root / f"{label}.json", beats, length)
        rows.append(
            {
                "id": label,
                "group": label,
                "split": "test",
                "audio": audio.name,
                "reference": label + ".json",
                "source": "Original programmatic arrangement; score grid",
                "license": "Original generated sound and annotations",
                "tags": ["original-mix", "synthetic", label],
            }
        )
    write_json(root / "manifest.json", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate(parser.parse_args().output)
