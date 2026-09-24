"""Tests for radio_imaging.prompts.

Only the OpenAI client is faked (an outside, paid service): see FakeOpenAI in
fakes.py. Its replies are the openai library's real ChatCompletion objects,
and its failures here are the library's real error classes.
"""
import httpx2
import openai
import pytest

from fakes import GPT_REPLY, FakeOpenAI


@pytest.fixture
def fake_openai(monkeypatch):
    FakeOpenAI.reset()
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    return FakeOpenAI


def test_prompt_builder_writes_a_musicgen_style_description():
    # Break caught: a broken free prompt builder, which people without an
    # OpenAI key depend on.
    from radio_imaging.prompts import build_prompt

    assert (build_prompt("pop", "upbeat", ["synth", "drums", "claps"], 120)
            == "upbeat pop radio jingle with synth, drums and claps, 120 bpm")
    assert build_prompt("ambient", "calm", [], 70) == "calm ambient radio jingle, 70 bpm"
    assert (build_prompt("rock", "energetic", ["guitar"], 140, extra=" ends on a big hit ")
            == "energetic rock radio jingle with guitar, 140 bpm. ends on a big hit")


def test_gpt_rewrites_the_idea_with_the_chosen_model_and_key(fake_openai):
    # Break caught: sending the wrong model, dropping the user's idea, using
    # someone else's key, or returning GPT's text with stray spaces.
    from radio_imaging.prompts import improve_with_gpt

    text = improve_with_gpt("sk-user-key", "gpt-6-luna", "brass hit for a sports show")

    assert text == GPT_REPLY
    assert fake_openai.keys == ["sk-user-key"]
    assert fake_openai.requests == [{
        "model": "gpt-6-luna",
        "messages": [{"role": "user",
                      "content": "Describe a radio imaging audio piece based on: brass hit for a sports show"}],
    }]


def test_retired_model_gives_a_clear_message(fake_openai):
    # Break caught: a raw 404 error reaching the user when OpenAI retires a
    # model, as happened to gpt-3.5-turbo-16k in 2024.
    from radio_imaging.prompts import ModelUnavailableError, improve_with_gpt

    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    fake_openai.fail_with = openai.NotFoundError(
        "Error code: 404 - The model `gpt-3.5-turbo-16k` does not exist",
        response=httpx2.Response(404, request=request),
        body={"error": {"code": "model_not_found"}},
    )

    with pytest.raises(ModelUnavailableError, match="gpt-3.5-turbo-16k"):
        improve_with_gpt("sk-user-key", "gpt-3.5-turbo-16k", "calm piano")
