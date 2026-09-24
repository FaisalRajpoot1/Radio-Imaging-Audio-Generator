"""Behaviour tests for radio_imaging_app.py (the Streamlit app).

Only the two slow or outside parts are faked (see fakes.py): loading MusicGen
(2.4 GB of weights) and the OpenAI client. The Streamlit script itself runs for
real inside Streamlit's AppTest harness.

Run from the repo root:  python -m pytest tests -q
"""
import io
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx2
import numpy as np
import openai
import pyloudnorm
import pytest
import scipy.io.wavfile
import streamlit as st
import streamlit.testing.v1.app_test as app_test_module
import transformers
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.local_script_runner import LocalScriptRunner

from fakes import GPT_REPLY, FakeMusicgen, FakeOpenAI, FakeProcessor
from radio_imaging.audio import LOUDNESS_TARGETS

APP = str(Path(__file__).resolve().parents[1] / "radio_imaging_app.py")
MODEL_ID = "facebook/musicgen-small"
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
    yield SimpleNamespace(loads=loads, processor=processor, model=model)
    st.cache_resource.clear()


@pytest.fixture
def fake_openai(monkeypatch):
    FakeOpenAI.reset()
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    return FakeOpenAI


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


def played_wav(at, served_media):
    [audio] = at.get("audio")
    return scipy.io.wavfile.read(io.BytesIO(served_media(audio)))


def download(at, label):
    return next(d for d in at.get("download_button") if d.proto.label == label)


# --- Fixes from the first round -------------------------------------------------

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
    rate, samples = played_wav(at, served_media)
    assert rate == 32_000
    assert samples.shape == (320_000,)  # the default 10 s


def test_wav_download_matches_the_generated_clip(musicgen, served_media):
    # Break caught: handing out a different file than the clip the user heard.
    at = open_app()

    generate_audio(at, "Punchy news bumper")

    [audio] = at.get("audio")
    assert served_media(download(at, "Download WAV")) == served_media(audio)


def test_music_model_gets_only_the_gpt_description(musicgen, fake_openai):
    # Break caught: the credit line added for people being fed into the music
    # model's text prompt, where it is noise.
    at = open_app()
    at.text_input(key="openai_api_key").input("sk-test")
    at.text_area(key="idea").input("brass hit for a sports show")

    click(at, GENERATE_PROMPT)
    click(at, GENERATE_AUDIO)

    assert musicgen.processor.texts == [[GPT_REPLY]]


def test_prompt_shown_to_user_keeps_the_original_credit(musicgen, fake_openai):
    # Break caught: dropping the original author's credit from the prompt the
    # user reads and downloads.
    at = open_app()
    at.text_input(key="openai_api_key").input("sk-test")
    at.text_area(key="idea").input("brass hit for a sports show")

    click(at, GENERATE_PROMPT)

    shown = [m.value for m in at.markdown if GPT_REPLY in m.value]
    assert any("Created through Radio Imaging Audio Generator by Bilsimaging" in text for text in shown)


# --- Free mode and the prompt builder ---------------------------------------------

def test_free_mode_makes_audio_without_an_openai_key(musicgen, fake_openai):
    # Break caught: requiring a paid OpenAI key before any audio can be made.
    at = open_app()
    at.text_area(key="idea").input("calm piano bed for a late-night show")

    click(at, GENERATE_AUDIO)

    assert musicgen.processor.texts == [["calm piano bed for a late-night show"]]
    assert fake_openai.requests == []
    assert len(at.get("audio")) == 1


def test_prompt_builder_description_goes_to_musicgen(musicgen):
    # Break caught: the builder's choices not reaching the music model.
    at = open_app()
    at.radio(key="describe_mode").set_value("Use the prompt builder").run()
    at.selectbox(key="genre").select("pop")
    at.selectbox(key="mood").select("upbeat")
    at.multiselect(key="instruments").set_value(["synth", "drums"])
    at.slider(key="bpm").set_value(120)

    click(at, GENERATE_AUDIO)

    assert musicgen.processor.texts == [["upbeat pop radio jingle with synth and drums, 120 bpm"]]


def test_generate_without_a_description_asks_for_one(musicgen):
    # Break caught: running the model on an empty prompt.
    at = open_app()

    click(at, GENERATE_AUDIO)

    assert musicgen.processor.texts == []
    assert any("describe" in e.value.lower() for e in at.error)


# --- Length, progress, seed ------------------------------------------------------

def test_clip_length_setting_changes_the_clip_length(musicgen, served_media):
    # Break caught: the length slider not reaching the model.
    at = open_app()
    at.slider(key="seconds").set_value(5)

    generate_audio(at, "Short sweeper")

    rate, samples = played_wav(at, served_media)
    assert samples.shape == (160_000,)


def test_progress_bar_follows_real_generation_and_finishes(musicgen):
    # Break caught: a progress bar that is not connected to the model's steps
    # (like the old fake one), or that never reaches the end.
    at = open_app()

    generate_audio(at, "Station ID with a rising whoosh")

    assert musicgen.model.got_streamer
    [bar] = at.get("progress")
    assert bar.proto.value == 100
    assert bar.proto.text.startswith("Done")


def test_same_seed_gives_the_same_clip(musicgen, served_media):
    # Break caught: the seed not reaching the model, so a good take can never
    # be made again.
    at = open_app()
    at.number_input(key="seed").set_value(1234)

    generate_audio(at, "Retro synth ID")
    first = served_media(at.get("audio")[0])
    generate_audio(at, "Retro synth ID")
    again = served_media(at.get("audio")[0])
    at.number_input(key="seed").set_value(99)
    generate_audio(at, "Retro synth ID")
    other = served_media(at.get("audio")[0])

    assert first == again
    assert first != other


def test_random_seed_is_shown_so_a_take_can_be_repeated(musicgen, served_media):
    # Break caught: a random take that can't be made again because its seed
    # is never shown.
    at = open_app()

    generate_audio(at, "Jazzy weekend promo")
    random_take = served_media(at.get("audio")[0])
    shown_seed = int(re.search(r"seed (\d+)", at.get("progress")[0].proto.text).group(1))
    at.number_input(key="seed").set_value(shown_seed)
    generate_audio(at, "Jazzy weekend promo")

    assert served_media(at.get("audio")[0]) == random_take


# --- Broadcast-ready output ------------------------------------------------------

@pytest.mark.parametrize("target", [-23.0, -16.0])
def test_clip_meets_the_chosen_loudness_target(musicgen, served_media, target):
    # Break caught: the loudness step being skipped, or the wrong target used.
    at = open_app()
    at.radio(key="loudness").set_value(next(label for label, lufs in LOUDNESS_TARGETS.items() if lufs == target))

    generate_audio(at, "Morning show bed")

    rate, samples = played_wav(at, served_media)
    assert pyloudnorm.Meter(rate).integrated_loudness(samples) == pytest.approx(target, abs=0.1)


def test_mp3_download_is_offered(musicgen, served_media):
    # Break caught: no MP3, or a broken one. Radio tools often want MP3.
    at = open_app()

    generate_audio(at, "Drive-time jingle")

    mp3 = served_media(download(at, "Download MP3"))
    assert mp3[0] == 0xFF and mp3[1] & 0xE0 == 0xE0   # MPEG audio frame sync


# --- The GPT step -----------------------------------------------------------------

def test_gpt_step_without_a_key_explains_what_to_do(musicgen, fake_openai):
    # Break caught: a confusing failure for people without an OpenAI key.
    at = open_app()
    at.text_area(key="idea").input("calm piano")

    click(at, GENERATE_PROMPT)

    assert fake_openai.requests == []
    assert any("API key" in e.value for e in at.error)


def test_retired_gpt_model_shows_a_clear_message(musicgen, fake_openai):
    # Break caught: a raw 404 error when OpenAI retires a model.
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    fake_openai.fail_with = openai.NotFoundError(
        "Error code: 404 - model not found", response=httpx2.Response(404, request=request), body=None,
    )
    at = open_app()
    at.text_input(key="openai_api_key").input("sk-test")
    at.text_area(key="idea").input("calm piano")

    click(at, GENERATE_PROMPT)

    assert any("not available" in e.value for e in at.error)


def test_other_model_name_is_used_for_the_gpt_step(musicgen, fake_openai):
    # Break caught: ignoring the model the user typed, so the app can't follow
    # OpenAI's next model change.
    at = open_app()
    at.text_input(key="openai_api_key").input("sk-test")
    at.text_area(key="idea").input("calm piano")
    at.selectbox(key="gpt_model").select("Other…").run()
    at.text_input(key="gpt_model_other").input("gpt-7-nova")

    click(at, GENERATE_PROMPT)

    assert fake_openai.requests[-1]["model"] == "gpt-7-nova"
