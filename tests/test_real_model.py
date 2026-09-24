"""Checks against the real MusicGen model: a 2.4 GB download, slow on a CPU.

Skipped unless RUN_REAL_MODEL=1 is set. These tests prove that FakeMusicgen
(fakes.py) behaves like the real model: one progress call per generation step,
the exact clip length, and the loudness target reached on real audio.
"""
import os

import numpy as np
import pyloudnorm
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_MODEL") != "1", reason="set RUN_REAL_MODEL=1 to run (downloads 2.4 GB)"
)


@pytest.fixture(scope="module")
def real_musicgen():
    from radio_imaging.model import load_musicgen

    return load_musicgen()


def test_real_model_reports_each_step_once_and_makes_the_exact_length(real_musicgen):
    from radio_imaging.model import generate, tokens_for

    processor, musicgen_model = real_musicgen
    seen = []

    rate, [clip] = generate(processor, musicgen_model, "upbeat pop radio jingle with synth and drums, 120 bpm",
                            seconds=3, seed=1, on_step=lambda done, total: seen.append((done, total)))

    assert tokens_for(3, musicgen_model) == 153
    assert seen == [(step, 153) for step in range(1, 154)]
    assert rate == 32_000
    assert clip.shape == (96_000,)


def test_real_clip_meets_the_loudness_target_without_clipping(real_musicgen):
    from radio_imaging.audio import finish_clip
    from radio_imaging.model import generate

    processor, musicgen_model = real_musicgen

    rate, [clip] = generate(processor, musicgen_model, "calm ambient radio jingle with piano, 70 bpm",
                            seconds=3, seed=2)
    out = finish_clip(clip, rate, -23.0)

    assert pyloudnorm.Meter(rate).integrated_loudness(out) == pytest.approx(-23.0, abs=0.1)
    assert np.max(np.abs(out)) <= 10 ** (-1 / 20) + 1e-6
