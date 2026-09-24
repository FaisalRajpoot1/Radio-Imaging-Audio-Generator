import io

import lameenc
import numpy as np
import pyloudnorm
import scipy.io.wavfile
from scipy.ndimage import minimum_filter1d, uniform_filter1d

PEAK_CEILING_DBFS = -1.0

# Integrated loudness targets in LUFS. EBU R128 is the European broadcast
# standard; -16 LUFS is common for podcasts and online audio.
LOUDNESS_TARGETS = {"Broadcast (EBU R128, -23 LUFS)": -23.0, "Online / podcast (-16 LUFS)": -16.0}


def limit_peaks(samples, rate, ceiling_dbfs=PEAK_CEILING_DBFS, lookahead=0.005, hold=0.05):
    """Look-ahead peak limiter: no sample above the ceiling, gain changes smooth.

    For each sample, take the lowest gain any sample needs from `hold` seconds
    back to `lookahead` seconds ahead, then average that over the look-ahead
    window. Every window that is averaged at a peak contains the peak itself,
    so the gain there is never above what the peak needs. The gain ramps down
    over the look-ahead time instead of jumping, which would distort.
    """
    ceiling = 10 ** (ceiling_dbfs / 20)
    needed = np.minimum(1.0, ceiling / np.maximum(np.abs(samples), 1e-12))
    ahead = max(1, round(lookahead * rate))
    back = round(hold * rate)
    # window [n - back, n + ahead - 1]; scipy's origin shifts the window left
    envelope = minimum_filter1d(needed, size=back + ahead, origin=back - (back + ahead) // 2, mode="nearest")
    # window [n - ahead + 1, n]
    gain = uniform_filter1d(envelope, size=ahead, origin=(ahead - 1) - ahead // 2, mode="nearest")
    # Rounding in the average can overshoot by about 1e-7; keep the promise exactly.
    return np.clip(samples * gain, -ceiling, ceiling).astype(np.float32)


def normalize_loudness(samples, rate, target_lufs):
    """Bring a clip to the target loudness (ITU-R BS.1770, measured by pyloudnorm).

    No sample goes above -1 dBFS: where peaks leave no headroom, the peak
    limiter lowers only the peaks. Limiting takes a little loudness away, so
    gain and limiting repeat a few times to land on the target.
    """
    if np.max(np.abs(samples)) == 0:
        return samples
    meter = pyloudnorm.Meter(rate)
    out = np.asarray(samples, dtype=np.float32)
    for _ in range(4):
        out = limit_peaks(out * 10 ** ((target_lufs - meter.integrated_loudness(out)) / 20), rate)
    return out


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
