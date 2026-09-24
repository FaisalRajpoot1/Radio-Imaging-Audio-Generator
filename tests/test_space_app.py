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


def take_files(files):
    """{take number: {".wav": path, ".mp3": path}} from the download list."""
    takes = {}
    for f in files:
        number = int(Path(f).name.split("_take")[1].split("_")[0])  # radio_imaging_take1_seed7.wav
        takes.setdefault(number, {})[Path(f).suffix] = f
    return takes


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

    takes = take_files(files)
    assert sorted(takes) == [1, 2, 3]
    for number in (1, 2, 3):
        rate, samples = scipy.io.wavfile.read(takes[number][".wav"])
        assert rate == 32_000
        assert samples.shape == (160_000,)
        assert pyloudnorm.Meter(rate).integrated_loudness(samples) == pytest.approx(-23.0, abs=0.1)


def test_every_take_can_be_downloaded_as_wav_and_mp3(space):
    # Break caught: missing or broken download files.
    status, *players, files = space.app.create_audio("news sting", 3, 0, 2, BROADCAST)

    takes = take_files(files)
    assert {number: set(formats) for number, formats in takes.items()} == {1: {".wav", ".mp3"}, 2: {".wav", ".mp3"}}
    for formats in takes.values():
        data = Path(formats[".mp3"]).read_bytes()
        assert data[0] == 0xFF and data[1] & 0xE0 == 0xE0   # MPEG audio frame sync


def test_players_play_the_small_mp3(space):
    # Break caught: players loading a big WAV. From some networks the Space's
    # downloads crawl at about 12 KB/s, and the 192 kbit/s MP3 is 62% smaller
    # than the 16-bit WAV a player would otherwise load. 10 s of MP3 at
    # 192 kbit/s is about 192_000 / 8 * 10 = 240_000 bytes.
    status, *players, files = space.app.create_audio("drive-time jingle", 10, 0, 2, BROADCAST)

    takes = take_files(files)
    assert players[:2] == [takes[1][".mp3"], takes[2][".mp3"]]
    for player in players[:2]:
        assert Path(player).stat().st_size == pytest.approx(240_000, rel=0.05)


def test_fewer_takes_leave_the_other_players_empty(space):
    # Break caught: showing stale or duplicate audio in unused players.
    status, *players, files = space.app.create_audio("drum fill", 3, 0, 1, BROADCAST)

    assert players[0] is not None
    assert players[1:] == [None, None]


def test_the_shown_seed_makes_the_same_takes_again(space):
    # Break caught: a random batch that can't be made again.
    status, *players, files = space.app.create_audio("jazzy promo", 3, 0, 2, BROADCAST)
    seed = int(re.search(r"seed (\d+)", status).group(1))

    again = space.app.create_audio("jazzy promo", 3, seed, 2, BROADCAST)[-1]

    first, second = take_files(files), take_files(again)
    for number in (1, 2):
        assert Path(first[number][".wav"]).read_bytes() == Path(second[number][".wav"]).read_bytes()


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
