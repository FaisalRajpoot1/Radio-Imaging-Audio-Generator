"""Tests for radio_imaging.model and radio_imaging.progress.

MusicGen itself is replaced by FakeMusicgen (see fakes.py), which has the same
interface as the real model.
"""
import numpy as np

from fakes import FakeMusicgen, FakeProcessor


def test_clip_length_maps_to_musicgen_tokens():
    # Break caught: a wrong seconds-to-tokens formula, so clips come out too
    # short or too long. 50 frames per second plus 3 delay steps: the original
    # app's 512 tokens gave the committed 10.18 s sample.
    from radio_imaging.model import tokens_for

    model = FakeMusicgen()

    assert tokens_for(10, model) == 503
    assert tokens_for(5, model) == 253
    assert tokens_for(10.18, model) == 512


def test_generate_returns_a_clip_of_the_requested_length():
    # Break caught: the clip length setting not reaching the model.
    from radio_imaging.model import generate

    rate, clips = generate(FakeProcessor(), FakeMusicgen(), "calm piano", seconds=5)

    assert rate == 32_000
    assert [clip.shape for clip in clips] == [(160_000,)]


def test_generate_makes_the_requested_number_of_variations():
    # Break caught: variations collapsing into one clip, or the prompt not
    # being repeated for each variation.
    from radio_imaging.model import generate

    processor = FakeProcessor()
    rate, clips = generate(processor, FakeMusicgen(), "news sting", seconds=3, variations=3)

    assert processor.texts == [["news sting", "news sting", "news sting"]]
    assert [clip.shape for clip in clips] == [(96_000,), (96_000,), (96_000,)]


def test_same_seed_gives_the_same_audio():
    # Break caught: the seed not being applied before sampling, so a good take
    # can never be made again.
    from radio_imaging.model import generate

    first = generate(FakeProcessor(), FakeMusicgen(), "x", seconds=3, seed=42)[1][0]
    again = generate(FakeProcessor(), FakeMusicgen(), "x", seconds=3, seed=42)[1][0]
    other = generate(FakeProcessor(), FakeMusicgen(), "x", seconds=3, seed=7)[1][0]

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_generate_sends_inputs_to_the_models_device():
    # Break caught: CPU tensors reaching a GPU model, which crashes on the
    # Hugging Face GPU Space. The "meta" device stands in for a GPU here.
    import torch

    from radio_imaging.model import generate

    model = FakeMusicgen(device=torch.device("meta"))

    rate, [clip] = generate(FakeProcessor(), model, "x", seconds=3)

    assert model.input_device == torch.device("meta")
    assert clip.shape == (96_000,)


def test_progress_still_finishes_when_the_last_step_is_not_reported():
    # Break caught: a progress bar stuck one step short on transformers 4.x,
    # whose any() skips custom stopping criteria on the final step.
    from radio_imaging.model import generate

    seen = []
    generate(FakeProcessor(), FakeMusicgen(skips_last_step_report=True), "x", seconds=3,
             on_step=lambda done, total: seen.append((done, total)))

    assert seen == [(step, 153) for step in range(1, 154)]


def test_progress_never_goes_past_100_percent():
    # Break caught: reporting more steps than exist. On a GPU, transformers 5.x
    # may run one extra step and undo it later, and a value over 100% crashes
    # Streamlit's progress bar.
    from radio_imaging.progress import StepProgress

    seen = []
    progress = StepProgress(3, lambda done, total: seen.append((done, total)))

    for _ in range(4):
        progress(None, None)

    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_progress_reports_each_generation_step_once():
    # Break caught: counting the prompt as a step (off by one), skipping steps,
    # or reporting "done" twice.
    from radio_imaging.model import generate

    seen = []
    generate(FakeProcessor(), FakeMusicgen(), "x", seconds=3, on_step=lambda done, total: seen.append((done, total)))

    assert seen == [(step, 153) for step in range(1, 154)]
