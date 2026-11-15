import numpy as np

from qildo_beats.meter import align_bars


def test_changes_meter_at_a_bar_boundary():
    numbers = [1, 2, 3, 4] * 7 + [1, 2, 3] * 8
    frames = np.arange(len(numbers)) * 50 + 47
    pulse = np.zeros(frames[-1] + 20)
    downbeats = np.zeros_like(pulse)
    pulse[frames] = 0.85
    downbeats[frames] = np.where(np.array(numbers) == 1, 0.9, 0.1)
    actual_frames, actual_numbers, _ = align_bars(
        frames, pulse, downbeats, (4, 3), learned=True
    )
    np.testing.assert_array_equal(actual_frames, frames)
    np.testing.assert_array_equal(actual_numbers, numbers)


def test_downbeat_can_realign_without_emitting_an_extra_click():
    expected = [1, 2, 3, 4] * 5
    frames = np.arange(len(expected)) * 50 + 50
    pulse = np.zeros(frames[-1] + 20)
    downbeats = np.zeros_like(pulse)
    pulse[frames] = 0.72
    pulse[frames[4] + 5] = 0.69
    downbeats[frames[::4]] = 0.9
    downbeats[frames[4]] = 0
    downbeats[frames[4] + 5] = 0.99
    refined, numbers, _ = align_bars(frames, pulse, downbeats, (4,), learned=True)
    assert 0 <= refined[4] - frames[4] <= 5
    assert len(refined) == len(frames)
    assert numbers.tolist() == expected
