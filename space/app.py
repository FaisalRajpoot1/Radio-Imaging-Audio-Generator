"""Radio Imaging Audio Generator: the Gradio app for the Hugging Face GPU Space.

Based on Radio Imaging Audio Generator by Bilel Aroua (Bilsimaging), MIT License.
It shares its audio code (the radio_imaging package) with the Streamlit app,
and runs MusicGen on a free ZeroGPU GPU.
"""
import os
import random
import tempfile
import time
from pathlib import Path

import gradio as gr
import spaces
import torch

from radio_imaging.audio import LOUDNESS_TARGETS, finish_clip, to_mp3_bytes, to_wav_bytes
from radio_imaging.model import generate, load_musicgen
from radio_imaging.prompts import (COPYRIGHT_NOTICE, GENRES, GPT_MODELS, INSTRUMENTS, MOODS,
                                   ModelUnavailableError, build_prompt, improve_with_gpt)

MAX_TAKES = 3
SOURCE_URL = "https://github.com/FaisalRajpoot1/Radio-Imaging-Audio-Generator/tree/fork-improvements"

# ZeroGPU wants models on cuda when the app starts; a real GPU is attached only
# while a @spaces.GPU function runs. Elsewhere, use the CPU unless there is a GPU.
ON_ZERO_GPU = os.getenv("SPACES_ZERO_GPU", "").lower() in ("1", "t", "true")
DEVICE = "cuda" if ON_ZERO_GPU or torch.cuda.is_available() else "cpu"
processor, musicgen_model = load_musicgen()
musicgen_model.to(DEVICE)


def gpu_duration(description, seconds, seed, takes):
    """Seconds of GPU time to reserve: more for longer clips and more takes."""
    return 15 + int(seconds) * int(takes)


@spaces.GPU(duration=gpu_duration)
def make_clips(description, seconds, seed, takes):
    return generate(processor, musicgen_model, description, seconds=seconds, seed=seed, variations=takes)


def create_audio(description, seconds, seed, takes, loudness):
    if not description.strip():
        raise gr.Error("Please describe your audio first: write it, or use the prompt builder.")
    seconds, takes = int(seconds), int(takes)
    take_seed = int(seed or 0) or random.randint(1, 2**31 - 1)

    start = time.perf_counter()
    rate, clips = make_clips(description, seconds, take_seed, takes)
    elapsed = time.perf_counter() - start

    # Loudness and encoding run here, on the CPU, so they use no GPU time.
    folder = Path(tempfile.mkdtemp(prefix="radio-imaging-"))
    players, files = [], []
    for number, clip in enumerate(clips, start=1):
        finished = finish_clip(clip, rate, LOUDNESS_TARGETS[loudness])
        players.append((rate, finished))
        for suffix, data in (("wav", to_wav_bytes(finished, rate)), ("mp3", to_mp3_bytes(finished, rate))):
            path = folder / f"radio_imaging_take{number}_seed{take_seed}.{suffix}"
            path.write_bytes(data)
            files.append(str(path))
    players += [None] * (MAX_TAKES - len(players))

    status = f"Done: {takes} take(s) of {seconds} s, generated in {elapsed:.1f} s (seed {take_seed})."
    return (status, *players, files)


def use_builder(genre, mood, instruments, bpm, extra):
    return build_prompt(genre, mood, list(instruments or []), int(bpm), extra or "")


def improve(idea, api_key, model_choice, other_model):
    if not (api_key or "").strip():
        raise gr.Error("This step needs your OpenAI API key. Or skip it: 'Generate Audio' works without one.")
    if not idea.strip():
        raise gr.Error("Please describe your audio first.")
    model = (other_model or "").strip() if model_choice == "Other…" else model_choice
    try:
        text = improve_with_gpt(api_key.strip(), model, idea)
    except ModelUnavailableError as error:
        raise gr.Error(str(error)) from error
    # GPT's text goes to MusicGen; the credit notice is shown apart from it.
    return text, COPYRIGHT_NOTICE.strip()


HEADER = f"""
# 📻 Radio Imaging Audio Generator
Make radio jingles, station IDs and sweepers from a text description, with Meta's MusicGen on a free GPU.
No OpenAI key needed: write the description yourself, or use the prompt builder.

Based on [Radio Imaging Audio Generator](https://github.com/bilsimaging/Radio-Imaging-Audio-Generator)
by **Bilel Aroua** ([Bilsimaging](https://bilsimaging.com)), MIT License.
This version by **Muhammad Faisal** ([source code]({SOURCE_URL})): free mode, prompt builder,
several takes per click, broadcast-ready loudness, and WAV/MP3 downloads.
"""

with gr.Blocks(title="Radio Imaging Audio Generator", delete_cache=(3600, 3600)) as demo:
    gr.Markdown(HEADER)
    with gr.Row():
        with gr.Column():
            description = gr.Textbox(label="Describe your audio", lines=4,
                                     placeholder="A calm, soothing melody with soft piano for a morning show")
            with gr.Accordion("Prompt builder (no key needed)", open=False):
                with gr.Row():
                    genre = gr.Dropdown(GENRES, value=GENRES[0], label="Genre")
                    mood = gr.Dropdown(MOODS, value=MOODS[0], label="Mood")
                instruments = gr.CheckboxGroup(INSTRUMENTS, label="Instruments")
                bpm = gr.Slider(60, 180, value=120, step=1, label="Tempo (BPM)")
                extra = gr.Textbox(label="Anything else? (optional)")
                build_button = gr.Button("Use this prompt")
            with gr.Accordion("Improve it with GPT (optional, needs your own OpenAI key)", open=False):
                api_key = gr.Textbox(label="OpenAI API key (used once, never stored)", type="password")
                model_choice = gr.Dropdown(GPT_MODELS + ["Other…"], value=GPT_MODELS[0], label="OpenAI model")
                other_model = gr.Textbox(label="Model name, if you chose Other…")
                improve_button = gr.Button("📄 Generate Prompt")
                notice = gr.Markdown()
            with gr.Row():
                seconds = gr.Slider(3, 30, value=10, step=1, label="Clip length (seconds)")
                takes = gr.Slider(1, MAX_TAKES, value=MAX_TAKES, step=1, label="Takes per click")
            with gr.Row():
                seed = gr.Number(value=0, precision=0, label="Seed (0 = random)")
                loudness = gr.Radio(list(LOUDNESS_TARGETS), value=next(iter(LOUDNESS_TARGETS)), label="Loudness")
            generate_button = gr.Button("▶ Generate Audio", variant="primary")
        with gr.Column():
            status = gr.Markdown()
            players = [gr.Audio(label=f"Take {number}", type="numpy", interactive=False)
                       for number in range(1, MAX_TAKES + 1)]
            files = gr.File(label="Downloads: WAV and MP3", file_count="multiple")

    build_button.click(use_builder, [genre, mood, instruments, bpm, extra], description)
    improve_button.click(improve, [description, api_key, model_choice, other_model], [description, notice])
    generate_button.click(create_audio, [description, seconds, seed, takes, loudness], [status, *players, files])

if __name__ == "__main__":
    demo.launch()
