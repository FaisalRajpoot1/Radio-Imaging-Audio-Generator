"""Behaviour tests for radio_imaging_app.py.

Only the two slow or external parts are faked: loading MusicGen (2.4 GB of
weights) and the OpenAI chat call. The Streamlit script itself runs for real
inside Streamlit's AppTest harness.

Run from the repo root:  python -m pytest tests -q
"""
import io
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest
import scipy.io.wavfile
import streamlit as st
import streamlit.testing.v1.app_test as app_test_module
import torch
import transformers
from openai.openai_object import OpenAIObject
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.local_script_runner import LocalScriptRunner

APP = str(Path(__file__).resolve().parents[1] / "radio_imaging_app.py")
MODEL_ID = "facebook/musicgen-small"
SAMPLE_RATE = 32_000  # MusicGen's EnCodec rate
GENERATE_AUDIO = "▶ Generate Audio"
GENERATE_PROMPT = "📄 Generate Prompt"


@pytest.fixture(autouse=True)
def run_in_temp_dir(monkeypatch, tmp_path):
    """Run every test in its own empty folder. Older versions of the app write
    their WAV into the working directory, and must never touch the repo."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def wait_for_script_thread(monkeypatch):
    """Works around a race in Streamlit 1.28.2's AppTest.

    AppTest reads the runner's SHUTDOWN event as soon as the script stops,
    sometimes before the runner thread has sent it (KeyError: 'client_state').
    Waiting for the thread to finish removes the race.
    """
    original_run = LocalScriptRunner.run

    def run_then_wait(self, *args, **kwargs):
        tree = original_run(self, *args, **kwargs)
        self._script_thread.join(timeout=30)
        return tree

    monkeypatch.setattr(LocalScriptRunner, "run", run_then_wait)


class FakeProcessor:
    """Stands in for the MusicGen processor and records every text it gets."""

    def __init__(self):
        self.texts = []

    def __call__(self, text, padding, return_tensors):
        self.texts.append(list(text))
        rows = len(text)
        return {
            "input_ids": torch.ones((rows, 4), dtype=torch.long),
            "attention_mask": torch.ones((rows, 4), dtype=torch.long),
        }


class FakeMusicgen:
    """Returns one second of a 440 Hz tone, shaped [batch, channels, samples]
    like the real model's output."""

    config = SimpleNamespace(audio_encoder=SimpleNamespace(sampling_rate=SAMPLE_RATE))

    def generate(self, input_ids, attention_mask, max_new_tokens):
        t = torch.arange(SAMPLE_RATE) / SAMPLE_RATE
        return torch.sin(2 * torch.pi * 440 * t).reshape(1, 1, -1)


@pytest.fixture
def musicgen(monkeypatch):
    loads = {"processor": 0, "model": 0}
    processor = FakeProcessor()
    model = FakeMusicgen()

    def load_processor(name, *args, **kwargs):
        assert name == MODEL_ID
        loads["processor"] += 1
        return processor

    def load_model(name, *args, **kwargs):
        assert name == MODEL_ID
        loads["model"] += 1
        return model

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(transformers.MusicgenForConditionalGeneration, "from_pretrained", load_model)
    st.cache_resource.clear()
    yield SimpleNamespace(loads=loads, processor=processor)
    st.cache_resource.clear()


@pytest.fixture
def fake_gpt(monkeypatch):
    """Replaces the OpenAI call with a reply shaped like a real v0.28 response."""
    reply = "Bright brass hit with a whoosh, 120 BPM"

    def create(model, messages, api_key):
        return OpenAIObject.construct_from({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21},
        })

    monkeypatch.setattr(openai.ChatCompletion, "create", create)
    return reply


@pytest.fixture
def served_media(monkeypatch):
    """Returns a reader for the bytes behind st.audio / st.download_button.

    AppTest gives every run a fresh in-memory media store and drops it when the
    run ends, so this keeps each store and reads from the latest run's one.
    """
    stores = []

    class KeptStorage(MemoryMediaFileStorage):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            stores.append(self)

    monkeypatch.setattr(app_test_module, "MemoryMediaFileStorage", KeptStorage)

    def read(element):
        return stores[-1].get_file(element.proto.url.rsplit("/", 1)[-1]).content

    return read


def open_app():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def click(at, label):
    button = next(b for b in at.button if b.label == label)
    button.click().run(timeout=60)
    assert not at.exception, at.exception
    return at


def generate_audio(at, prompt):
    at.session_state["generated_prompt"] = prompt
    return click(at, GENERATE_AUDIO)


def test_app_starts_without_tensorflow_installed(monkeypatch, musicgen):
    # Break caught: an unused TensorFlow import coming back. requirements.txt
    # no longer installs TensorFlow, so the app must start without it.
    monkeypatch.setitem(sys.modules, "tensorflow", None)  # makes `import tensorflow` fail

    at = AppTest.from_file(APP, default_timeout=30)
    at.run()

    assert not at.exception


def test_model_loads_once_per_server_not_per_click(musicgen):
    # Break caught: loading MusicGen inside the click handler, which reloads
    # 2.4 GB of weights on every click, for every user.
    first = open_app()
    generate_audio(first, "Upbeat synth sting for a morning show")
    generate_audio(first, "Upbeat synth sting for a morning show")
    second = open_app()
    generate_audio(second, "Warm acoustic station ID")

    assert musicgen.loads == {"processor": 1, "model": 1}


def test_generate_audio_adds_no_artificial_wait(musicgen):
    # Break caught: the old fake progress bar that slept 100 x 0.1 s = 10 s
    # before any real work started.
    at = open_app()
    at.session_state["generated_prompt"] = "Short drum fill"

    start = time.perf_counter()
    click(at, GENERATE_AUDIO)
    elapsed = time.perf_counter() - start

    assert elapsed < 3.0, f"one click took {elapsed:.1f} s with an instant fake model"


def test_audio_is_served_from_memory_not_a_shared_file(musicgen, served_media, tmp_path):
    # Break caught: every session writing its clip to one fixed WAV path on the
    # server, where users can overwrite each other's audio. (Tests run inside
    # tmp_path, see run_in_temp_dir.)
    at = open_app()

    generate_audio(at, "Airy pad with a soft riser")

    assert list(tmp_path.iterdir()) == []
    [audio] = at.get("audio")
    rate, samples = scipy.io.wavfile.read(io.BytesIO(served_media(audio)))
    assert rate == 32_000
    assert samples.shape == (32_000,)


def test_wav_download_matches_the_generated_clip(musicgen, served_media):
    # Break caught: the download button missing, or handing out a different
    # file than the clip the user just heard.
    at = open_app()

    generate_audio(at, "Punchy news bumper")

    [audio] = at.get("audio")
    [download] = at.get("download_button")
    assert served_media(download) == served_media(audio)


def test_music_model_gets_only_the_gpt_description(musicgen, fake_gpt):
    # Break caught: the credit line added for people being fed into the music
    # model's text prompt, where it is noise.
    at = open_app()
    at.text_input[0].input("sk-test")
    at.text_area[0].input("brass hit for a sports show")

    click(at, GENERATE_PROMPT)
    click(at, GENERATE_AUDIO)

    assert musicgen.processor.texts == [["Bright brass hit with a whoosh, 120 BPM"]]


def test_prompt_shown_to_user_keeps_the_original_credit(musicgen, fake_gpt):
    # Break caught: dropping the original author's credit from the prompt the
    # user reads and downloads.
    at = open_app()
    at.text_input[0].input("sk-test")
    at.text_area[0].input("brass hit for a sports show")

    click(at, GENERATE_PROMPT)

    shown = [m.value for m in at.markdown if "Bright brass hit with a whoosh, 120 BPM" in m.value]
    assert any("Created through Radio Imaging Audio Generator by Bilsimaging" in text for text in shown)
