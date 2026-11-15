import numpy as np

from qildo_beats.grid import Beat
from render_clicks import render


def test_click_accents_and_excerpt_alignment():
    samples = np.zeros((24000, 2), dtype=np.float32)
    mixed = render(samples, 24000, [Beat(0.25, 4), Beat(0.5, 1)], 0.25, 0.75)
    assert mixed.shape == (12000, 2)
    assert np.any(mixed[:960])
    assert not np.any(mixed[960:6000])
    assert mixed[6000:].max() > mixed[:960].max()
