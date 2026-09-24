"""Measure how loud MusicGen's raw clips are, and how close finish_clip() gets
them to each loudness target. Uses the real model; nothing is mocked.

Usage, from the repo root, in an environment with requirements-dev.txt installed:

    python benchmarks/bench_loudness.py --seconds 5

Prints one JSON line per prompt, then a JSON summary.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyloudnorm

# Import the radio_imaging package from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radio_imaging.audio import LOUDNESS_TARGETS, finish_clip
from radio_imaging.model import generate, load_musicgen, tokens_for

PROMPTS = [
    "calm ambient radio jingle with piano, 70 bpm",
    "energetic rock radio jingle with guitar and drums, 140 bpm",
    "upbeat electronic radio jingle with synth and claps, 128 bpm",
    "dramatic orchestral radio jingle with strings and brass, 90 bpm",
    "warm jazz radio jingle with piano and bass, 100 bpm",
    "upbeat hip hop radio jingle with drums and bass, 95 bpm",
    "short news sting with a rising whoosh and a final hit",
    "soft acoustic guitar bed for a late-night show",
]


def dbfs(samples):
    return 20 * np.log10(np.max(np.abs(samples)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=int, default=5)
    args = parser.parse_args()

    processor, musicgen_model = load_musicgen()
    rows = []
    for i, prompt in enumerate(PROMPTS):
        steps = []
        rate, [clip] = generate(processor, musicgen_model, prompt, seconds=args.seconds, seed=i + 1,
                                on_step=lambda done, total: steps.append(done))
        meter = pyloudnorm.Meter(rate)
        row = {
            "prompt": prompt,
            "progress_calls": len(steps),
            "expected_steps": tokens_for(args.seconds, musicgen_model),
            "clip_s": len(clip) / rate,
            "raw_lufs": round(meter.integrated_loudness(clip), 2),
            "raw_peak_dbfs": round(dbfs(clip), 2),
        }
        for target in LOUDNESS_TARGETS.values():
            finished = finish_clip(clip, rate, target)
            row[f"lufs_for_{target:g}"] = round(meter.integrated_loudness(finished), 2)
            row[f"peak_dbfs_for_{target:g}"] = round(dbfs(finished), 2)
        rows.append(row)
        print(json.dumps(row), flush=True)

    raw = [row["raw_lufs"] for row in rows]
    summary = {"clips": len(rows), "raw_lufs_min": min(raw), "raw_lufs_max": max(raw),
               "raw_spread_lu": round(max(raw) - min(raw), 2)}
    for target in LOUDNESS_TARGETS.values():
        summary[f"max_miss_lu_for_{target:g}"] = round(max(abs(row[f"lufs_for_{target:g}"] - target) for row in rows), 2)
        summary[f"max_peak_dbfs_for_{target:g}"] = max(row[f"peak_dbfs_for_{target:g}"] for row in rows)
    print(json.dumps({"summary": summary}))


if __name__ == "__main__":
    main()
