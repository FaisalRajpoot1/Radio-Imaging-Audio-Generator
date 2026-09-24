"""Tests for radio_imaging.audio, using synthetic signals.

Loudness is measured with pyloudnorm, a meter for the ITU-R BS.1770 standard
that radio loudness targets (like EBU R128) are defined with.
"""
import io

import numpy as np
import pyloudnorm
import pytest
import scipy.io.wavfile

RATE = 32_000
CEILING = 10 ** (-1 / 20)  # -1 dBFS as a sample value, about 0.891


def noise(seconds, level=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * RATE)) * level).astype(np.float32)


def loudness(samples):
    return pyloudnorm.Meter(RATE).integrated_loudness(samples)


def peaky_clip(seconds=5):
    """Quiet noise with short loud hits, like drums: high peaks, low average loudness."""
    clip = noise(seconds, level=0.03)
    hit = (0.8 * np.hanning(int(0.005 * RATE))).astype(np.float32)  # a 5 ms hit
    for start in range(0, len(clip) - len(hit), RATE // 2):
        clip[start:start + len(hit)] += hit
    return clip


@pytest.mark.parametrize("target", [-23.0, -16.0])
def test_normalize_reaches_the_loudness_target(target):
    # Break caught: a wrong gain formula or meter setup, so clips miss the
    # radio loudness target.
    from radio_imaging.audio import normalize_loudness

    out = normalize_loudness(noise(5), RATE, target)

    assert loudness(out) == pytest.approx(target, abs=0.1)


def test_normalize_never_pushes_peaks_above_minus_1_dbfs():
    # Break caught: boosting a quiet but peaky clip until it clips. A sparse
    # click track is quiet on average but has full-size peaks.
    from radio_imaging.audio import normalize_loudness

    clicks = np.zeros(5 * RATE, dtype=np.float32)
    clicks[::RATE // 2] = 0.5

    out = normalize_loudness(clicks, RATE, -16.0)

    assert np.max(np.abs(out)) <= CEILING + 1e-6
    assert np.max(np.abs(out)) == pytest.approx(CEILING, rel=1e-4)


def test_online_target_is_reached_on_a_peaky_clip_without_clipping():
    # Break caught: stopping short of the target when peaks leave no headroom.
    # This happened on 6 of 8 real MusicGen clips at -16 LUFS (up to 7.5 LU
    # short). A peak limiter must lower only the peaks instead.
    from radio_imaging.audio import normalize_loudness

    out = normalize_loudness(peaky_clip(), RATE, -16.0)

    assert loudness(out) == pytest.approx(-16.0, abs=0.2)
    assert np.max(np.abs(out)) <= CEILING + 1e-6


def test_limiter_changes_gain_smoothly():
    # Break caught: hard clipping (cutting peaks off), which distorts. A
    # limiter's gain may only change a little from one sample to the next.
    from radio_imaging.audio import limit_peaks

    clip = peaky_clip() * 4  # peaks far above the ceiling

    out = limit_peaks(clip, RATE)

    audible = np.abs(clip) > 1e-3
    gain = np.divide(out, clip, out=np.ones_like(clip), where=audible)
    both = audible[:-1] & audible[1:]
    assert np.max(np.abs(np.diff(gain)[both])) <= 0.01
    assert np.max(np.abs(out)) <= CEILING + 1e-6


def test_normalize_leaves_silence_alone():
    # Break caught: dividing by zero loudness and returning NaN or inf.
    from radio_imaging.audio import normalize_loudness

    out = normalize_loudness(np.zeros(RATE * 2, dtype=np.float32), RATE, -16.0)

    assert np.all(out == 0)


def test_fades_start_and_end_at_silence():
    # Break caught: clicks at the start or end of the clip, or fades that eat
    # into the middle of it.
    from radio_imaging.audio import apply_fades

    out = apply_fades(np.ones(3 * RATE, dtype=np.float32), RATE, fade_in=0.05, fade_out=1.0)

    assert out[0] == 0.0 and out[-1] == 0.0
    assert out[int(0.05 * RATE)] == 1.0            # fade-in is over after 50 ms
    assert out[RATE] == 1.0                        # the middle is untouched
    assert out[-RATE // 2] == pytest.approx(0.5, abs=1e-3)   # halfway through the 1 s fade-out


def test_wav_bytes_round_trip():
    # Break caught: a wrong sample rate or sample format in the WAV download.
    from radio_imaging.audio import to_wav_bytes

    clip = noise(1)

    rate, back = scipy.io.wavfile.read(io.BytesIO(to_wav_bytes(clip, RATE)))

    assert rate == RATE
    assert np.array_equal(back, clip)


def test_mp3_is_a_192_kbps_stream_of_the_right_length():
    # Break caught: an empty or broken MP3, or the wrong bit rate or length.
    # At 192 kbit/s, 10 s of audio is about 192_000 / 8 * 10 = 240_000 bytes.
    from radio_imaging.audio import to_mp3_bytes

    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(10 * RATE) / RATE)).astype(np.float32)

    mp3 = to_mp3_bytes(tone, RATE)

    assert mp3[0] == 0xFF and mp3[1] & 0xE0 == 0xE0   # MPEG audio frame sync
    assert len(mp3) == pytest.approx(240_000, rel=0.05)


def test_finished_clip_is_faded_and_at_the_target_loudness():
    # Break caught: the app skipping a step, or normalising before the fades
    # (which would leave the clip quieter than the target).
    from radio_imaging.audio import finish_clip

    out = finish_clip(noise(5), RATE, target_lufs=-23.0)

    assert out[0] == 0.0 and out[-1] == 0.0
    assert loudness(out) == pytest.approx(-23.0, abs=0.1)
