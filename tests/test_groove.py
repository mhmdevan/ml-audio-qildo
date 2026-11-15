from io import BytesIO

import pytest

mido = pytest.importorskip("mido")

from qildo_beats.groove import midi_grid  # noqa: E402


def test_midi_grid_preserves_pickup_count_and_bounds():
    midi = mido.MidiFile(ticks_per_beat=480)
    track = midi.add_track()
    track.extend(
        [
            mido.MetaMessage("time_signature", numerator=3, denominator=4),
            mido.MetaMessage("set_tempo", tempo=600_000),
            mido.Message("note_on", note=36, velocity=100, time=480),
            mido.Message("note_off", note=36, time=1440),
        ]
    )
    buffer = BytesIO()
    midi.save(file=buffer)
    grid = midi_grid(buffer.getvalue(), 2.0)
    assert [event.beat for event in grid] == [2, 3, 1]
    assert [event.time for event in grid] == pytest.approx([0.6, 1.2, 1.8])
