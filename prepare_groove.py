#!/usr/bin/env python3
"""Создать манифест из MIDI-архива и распакованных WAV Groove MIDI Dataset."""

import argparse
from pathlib import Path

from qildo_beats.groove import import_groove


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("midi_zip", type=Path)
    parser.add_argument("wav_root", type=Path, help="Распакованный каталог groove/")
    parser.add_argument("output", type=Path)
    parser.add_argument("--selection", type=Path, help="CSV с фиксированным списком id")
    args = parser.parse_args()
    try:
        import_groove(args.midi_zip, args.wav_root, args.output, args.selection)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"Ошибка импорта: {error}\n")


if __name__ == "__main__":
    main()
