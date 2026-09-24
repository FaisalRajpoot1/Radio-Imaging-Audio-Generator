"""Tests for space/app.py, the Gradio app for the Hugging Face GPU Space.

Skipped where Gradio is not installed. Only loading MusicGen and the OpenAI
client are faked (see fakes.py); the app's own code runs for real.
"""
import importlib.util
import io
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

gr = pytest.importorskip("gradio")
import openai  # noqa: E402
import pyloudnorm  # noqa: E402
import scipy.io.wavfile  # noqa: E402
import spaces  # noqa: E402,F401  (imported first, so its own ZeroGPU settings stay off in tests)
import transformers  # noqa: E402

from fakes import GPT_REPLY, FakeMusicgen, FakeOpenAI, FakeProcessor  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "space" / "app.py"
BROADCAST = "Broadcast (EBU R128, -23 LUFS)"


def load_app(monkeypatch, zero_gpu=False):
    processor, model = FakeProcessor(), FakeMusicgen()
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda name, *a, **k: processor)
    monkeypatch.setattr(transformers.MusicgenForConditionalGeneration, "from_pretrained", lambda name, *a, **k: model)
    if zero_gpu:
        monkeypatch.setenv("SPACES_ZERO_GPU", "true")
    else:
        monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    spec = importlib.util.spec_from_file_location("space_app", APP_PATH)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return SimpleNamespace(app=app, processor=processor, model=model)


@pytest.fixture
def space(monkeypatch, tmp_path):
    import tempfile

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # download files land in pytest's folder
    return load_app(monkeypatch)


@pytest.fixture
def fake_openai(monkeypatch):
    FakeOpenAI.reset()
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    return FakeOpenAI


def test_model_goes_on_the_gpu_on_zerogpu(monkeypatch):
    # Break caught: loading the model on the CPU on ZeroGPU, which must place
    # models on cuda when the app starts.
    loaded = load_app(monkeypatch, zero_gpu=True)

    assert loaded.model.moved_to == ["cuda"]


def test_model_stays_on_the_cpu_without_a_gpu(monkeypatch):
    # Break caught: asking for cuda on a machine without a GPU, which crashes.
    loaded = load_app(monkeypatch)

    assert loaded.model.moved_to == ["cpu"]


def test_each_take_has_the_requested_length_and_loudness(space):
    # Break caught: takes at the wrong length, or skipping the loudness step.
    status, *players, files = space.app.create_audio("calm piano bed", 5, 0, 3, BROADCAST)

    assert len(players) == 3
    for rate, samples in players:
        assert rate == 32_000
        assert samples.shape == (160_000,)
        assert pyloudnorm.Meter(rate).integrated_loudness(samples) == pytest.approx(-23.0, abs=0.1)


def test_every_take_can_be_downloaded_as_wav_and_mp3(space):
    # Break caught: missing or broken download files.
    status, *players, files = space.app.create_audio("news sting", 3, 0, 2, BROADCAST)

    formats_per_take = {}
    for f in files:
        take = Path(f).name.split("_seed")[0]
        formats_per_take.setdefault(take, set()).add(Path(f).suffix)
    assert formats_per_take == {"radio_imaging_take1": {".wav", ".mp3"}, "radio_imaging_take2": {".wav", ".mp3"}}
    wavs = [f for f in files if f.endswith(".wav")]
    for wav, (rate, samples) in zip(sorted(wavs), players[:2]):
        file_rate, file_samples = scipy.io.wavfile.read(wav)
        assert file_rate == 32_000 and np.array_equal(file_samples, samples)
    for mp3 in (f for f in files if f.endswith(".mp3")):
        data = Path(mp3).read_bytes()
        assert data[0] == 0xFF and data[1] & 0xE0 == 0xE0   # MPEG audio frame sync


def test_fewer_takes_leave_the_other_players_empty(space):
    # Break caught: showing stale or duplicate audio in unused players.
    status, *players, files = space.app.create_audio("drum fill", 3, 0, 1, BROADCAST)

    assert players[0] is not None
    assert players[1:] == [None, None]


def test_the_shown_seed_makes_the_same_takes_again(space):
    # Break caught: a random batch that can't be made again.
    status, *players, files = space.app.create_audio("jazzy promo", 3, 0, 2, BROADCAST)
    seed = int(re.search(r"seed (\d+)", status).group(1))

    again = space.app.create_audio("jazzy promo", 3, seed, 2, BROADCAST)[1:3]

    for (rate, first), (_, second) in zip(players[:2], again):
        assert np.array_equal(first, second)


def test_empty_description_asks_for_one(space):
    # Break caught: spending GPU time on an empty prompt.
    with pytest.raises(gr.Error, match="describe"):
        space.app.create_audio("   ", 5, 0, 1, BROADCAST)

    assert space.processor.texts == []


def test_prompt_builder_fills_the_description(space):
    # Break caught: the builder's choices not reaching the description.
    text = space.app.use_builder("pop", "upbeat", ["synth", "drums"], 120, "")

    assert text == "upbeat pop radio jingle with synth and drums, 120 bpm"


def test_gpt_without_a_key_explains_what_to_do(space, fake_openai):
    # Break caught: a confusing failure for visitors without an OpenAI key.
    with pytest.raises(gr.Error, match="API key"):
        space.app.improve("calm piano", "", "gpt-6-luna", "")

    assert fake_openai.requests == []


def test_gpt_text_replaces_the_description_and_the_credit_stays_apart(space, fake_openai):
    # Break caught: the credit notice leaking into the music prompt, or the
    # typed "Other" model name being ignored.
    description, notice = space.app.improve("calm piano", "sk-test", "Other…", "gpt-7-nova")

    assert description == GPT_REPLY
    assert "Created through Radio Imaging Audio Generator by Bilsimaging" in notice
    assert fake_openai.requests[-1]["model"] == "gpt-7-nova"


def test_retired_gpt_model_shows_a_clear_message(space, fake_openai):
    # Break caught: a raw 404 error when OpenAI retires a model.
    import httpx2

    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    fake_openai.fail_with = openai.NotFoundError(
        "Error code: 404 - model not found", response=httpx2.Response(404, request=request), body=None,
    )

    with pytest.raises(gr.Error, match="not available"):
        space.app.improve("calm piano", "sk-test", "gpt-6-luna", "")


def test_gpu_time_budget_grows_with_the_work(space):
    # Break caught: a fixed GPU budget that cuts long or multi-take requests short.
    # Same arguments as the GPU function: description, seconds, seed, takes.
    small = space.app.gpu_duration("x", 5, 0, 1)
    large = space.app.gpu_duration("x", 30, 0, 3)

    assert large > small > 0
