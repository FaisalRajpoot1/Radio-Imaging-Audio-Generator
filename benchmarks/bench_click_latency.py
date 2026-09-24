"""Measure what a user waits for in the app, using the real MusicGen model.

The Streamlit script runs inside Streamlit's AppTest harness (no browser), so
the numbers cover exactly the app's own code: imports, model loading, audio
generation and WAV encoding. Nothing is mocked.

Usage, from the repo root, in an environment with requirements-dev.txt installed:

    python benchmarks/bench_click_latency.py radio_imaging_app.py --clicks 3

Run it once per app version; each run is a fresh Python process, so the first
render includes the cost of importing the app's libraries.
"""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import psutil
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.local_script_runner import LocalScriptRunner

GENERATE_AUDIO = "▶ Generate Audio"
PROMPT = (
    "Energetic radio station ID: punchy drums, bright synth stabs, "
    "a rising whoosh and a short final hit, 120 BPM."
)


def rss_mb():
    return round(psutil.Process().memory_info().rss / 2**20)


def wait_for_script_thread():
    """Works around a race in Streamlit 1.28.2's AppTest: it can read the
    runner's SHUTDOWN event before the runner thread has sent it
    (KeyError: 'client_state'). Waiting for the thread removes the race."""
    original_run = LocalScriptRunner.run

    def run_then_wait(self, *args, **kwargs):
        tree = original_run(self, *args, **kwargs)
        self._script_thread.join(timeout=60)
        return tree

    LocalScriptRunner.run = run_then_wait


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("app", help="path to the Streamlit script to measure")
    parser.add_argument("--clicks", type=int, default=3, help="Generate Audio clicks in one session")
    args = parser.parse_args()
    app = str(Path(args.app).resolve())

    # Older versions of the app write their WAV into the working directory.
    os.chdir(tempfile.mkdtemp(prefix="radio-bench-"))
    wait_for_script_thread()

    result = {"app": app, "rss_mb_before_render": rss_mb()}

    at = AppTest.from_file(app, default_timeout=1800)
    start = time.perf_counter()
    at.run()
    result["first_render_s"] = round(time.perf_counter() - start, 2)
    result["rss_mb_after_render"] = rss_mb()
    assert not at.exception, at.exception

    clicks = []
    for _ in range(args.clicks):
        at.session_state["generated_prompt"] = PROMPT
        button = next(b for b in at.button if b.label == GENERATE_AUDIO)
        start = time.perf_counter()
        button.click().run(timeout=1800)
        clicks.append(round(time.perf_counter() - start, 2))
        # The app catches errors and shows st.error, so check for both kinds.
        assert not at.exception, at.exception
        assert not at.error, [e.value for e in at.error]
        assert at.get("audio"), "the click produced no audio"
    result["click_s"] = clicks

    memory = psutil.Process().memory_info()
    result["rss_mb_after_clicks"] = round(memory.rss / 2**20)
    if hasattr(memory, "peak_wset"):  # Windows only
        result["peak_mb"] = round(memory.peak_wset / 2**20)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
