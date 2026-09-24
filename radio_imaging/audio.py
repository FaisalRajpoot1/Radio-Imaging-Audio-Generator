import io

import lameenc
import numpy as np
import pyloudnorm
import scipy.io.wavfile

PEAK_CEILING_DBFS = -1.0

# Integrated loudness targets in LUFS. EBU R128 is the European broadcast
# standard; -16 LUFS is common for podcasts and online audio.
LOUDNESS_TARGETS = {"Broadcast (EBU R128, -23 LUFS)": -23.0, "Online / podcast (-16 LUFS)": -16.0}


def normalize_loudness(samples, rate, target_lufs):
    """Scale a clip to the target loudness (ITU-R BS.1770, measured by pyloudnorm).

    The gain is capped so the highest sample stays at or below -1 dBFS. A very
    peaky clip then ends up a little quieter than the target instead of clipping.
    """
    peak = np.max(np.abs(samples))
    if peak == 0:
        return samples
    loudness = pyloudnorm.Meter(rate).integrated_loudness(samples)
    gain_db = min(target_lufs - loudness, PEAK_CEILING_DBFS - 20 * np.log10(peak))
    return (samples * 10 ** (gain_db / 20)).astype(np.float32)


def apply_fades(samples, rate, fade_in=0.05, fade_out=1.0):
    """Short fade in against clicks, longer fade out for a clean ending."""
    faded = np.array(samples, dtype=np.float32, copy=True)
    fade_in_len = min(round(fade_in * rate), len(faded))
    fade_out_len = min(round(fade_out * rate), len(faded))
    faded[:fade_in_len] *= np.linspace(0.0, 1.0, fade_in_len, dtype=np.float32)
    faded[len(faded) - fade_out_len:] *= np.linspace(1.0, 0.0, fade_out_len, dtype=np.float32)
    return faded


def finish_clip(samples, rate, target_lufs):
    """Fades first, then loudness, so the final clip is at the target."""
    return normalize_loudness(apply_fades(samples, rate), rate, target_lufs)


def to_wav_bytes(samples, rate):
    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, rate, np.asarray(samples, dtype=np.float32))
    return buffer.getvalue()


def to_mp3_bytes(samples, rate, kbps=192):
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(kbps)
    encoder.set_in_sample_rate(rate)
    encoder.set_channels(1)
    encoder.set_quality(2)  # 2 = high quality, 7 = fastest
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    return bytes(encoder.encode(pcm) + encoder.flush())
