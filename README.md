
# 🌟Radio Imaging Audio Generator

[![tests](https://github.com/FaisalRajpoot1/Radio-Imaging-Audio-Generator/actions/workflows/tests.yml/badge.svg?branch=fork-improvements)](https://github.com/FaisalRajpoot1/Radio-Imaging-Audio-Generator/actions/workflows/tests.yml)

> **This is a fork.** The original app is [Radio Imaging Audio Generator](https://github.com/bilsimaging/Radio-Imaging-Audio-Generator) by **Bilel Aroua** ([Bilsimaging](https://bilsimaging.com)), MIT License. The idea, the app and its design are his. This fork keeps the app working after OpenAI's GPT-3.5 shutdown, adds a free mode, a real progress bar, clip length control and broadcast-ready audio, fixes speed problems, and measures every result. See [What this fork changes](#what-this-fork-changes).

## 📜Description
The Radio Imaging Audio Generator is a Streamlit-based application designed for radio producers and music creators. It combines OpenAI's GPT models with Facebook's MusicGen technology, enabling the generation of unique audio pieces from user-provided prompts.

### 🌐 Project Continuation and User Involvement
This app is the next step in our project, following the Custom GPT Radio Imaging and MusicGen AI. It's tailored for radio producers and music creators, offering new levels of creativity and efficiency.
[GPT] (https://chat.openai.com/g/g-65x53n87E-radio-imaging-musicgen-ai)

## 🚀 Features
- Text-prompt-based audio generation for radio imaging.
- Integration with OpenAI's GPT and Facebook's MusicGen models.
- User-friendly interface for inputting prompts and API keys.
- Direct audio playback and download options within the app.

## What this fork changes

Two rounds of work, each one measured.

### Round 1: speed and reliability

The app works the same for users. This round fixes five problems in how it makes audio.

| Problem in the original | Fix in this fork |
|---|---|
| The 2.4 GB MusicGen model was loaded again on every click, for every user. | The model loads once per server (`st.cache_resource`). Every click reuses it. |
| A fake progress bar added 10 seconds of waiting to every click. | Removed. A spinner shows while the real work runs. |
| TensorFlow was installed and imported, but never used. | Removed from the code and from `requirements.txt`. |
| Every user's audio was written to one fixed WAV file on the server. | The audio stays in memory, one copy per user. Added a **Download Audio** button. |
| The copyright notice was sent to MusicGen as part of the music prompt. | MusicGen gets only GPT's description. The notice is still shown and saved with the prompt. |

#### Measured results (round 1)

| Measure | Original | This fork | Change |
|---|---|---|---|
| Install size (`site-packages`) | 2.72 GB | 1.51 GB | −1.21 GB (−44.5%) |
| First page load after the server starts (median of 5) | 5.97 s | 2.95 s | −3.0 s (−51%) |
| Memory after the first page load | 468 MiB | 288 MiB | −180 MiB (−38%) |
| Wait per "Generate Audio" click, after the first (median) | 124.3 s | 102.3 s | −22.0 s (−18%) |
| Peak memory over 3 clicks | 6,702 MiB | 5,015 MiB | −1,687 MiB (−25%) |
| Model loads | 1 per click | 1 per server | — |

How to read this:
- The click saving (22 s) matches its two causes: the 10 s fake progress bar, plus an 11 s model reload (median of 3 timed loads).
- The first click after a server starts still loads the model once. It took 115.6 s (median of 2), against 121.9 s in the original (median of 3).
- Most of each click, about 100 s, is MusicGen making 10 seconds of audio on a laptop CPU. This fork does not change that part. A GPU would.
- Page load and memory drop because `transformers` imports TensorFlow at start-up whenever TensorFlow is installed.

How it was measured: Intel Core i7-8665U laptop (4 cores), 16 GB RAM, no GPU, Windows 11, Python 3.11. The original ran with its own `requirements.txt` (with TensorFlow). The fork ran in a clean install from the new `requirements.txt`. Both used the same prompt, with the model already on disk and the network off. Runs alternated between the two versions: 3 runs of 3 clicks for the original, 2 for the fork. Memory is the process's resident memory (RSS). Script: `benchmarks/bench_click_latency.py`.

### Round 2: new features

| What was missing or broken | What this fork adds |
|---|---|
| The GPT step used `gpt-3.5-turbo-16k` (shut down on 13 Sep 2024) and `gpt-3.5-turbo` (shuts down on 23 Oct 2026), through the old OpenAI library (0.28). | The current OpenAI library and models (default `gpt-6-luna`), an "Other…" field for future models, and a clear message if a model is retired. |
| A paid OpenAI key was needed before any audio could be made. | **Free mode**: write the description yourself, or use the new **prompt builder** (genre, mood, instruments, tempo). |
| No progress bar after round 1 (the old one was fake). | A **real progress bar** that follows MusicGen's generation steps, with the time left. |
| Every clip was about 10 s long, and every take was random. | **Clip length** from 3 to 30 s, and a **seed** shown after every take, so a good take can be made again. |
| Clip loudness varied a lot between prompts, and only WAV was offered. | **Broadcast-ready audio**: loudness set to EBU R128 (-23 LUFS) or online (-16 LUFS) with a look-ahead peak limiter, so no sample goes above -1 dBFS; fades; WAV and MP3 downloads. |

The audio logic now lives in a small `radio_imaging/` package (model, progress, audio, prompts), so other interfaces can reuse it.

#### Measured results (round 2)

Wait per click by clip length (median of the clicks after the first, which also loads the model):

| Clip length | Wait per click | Work per second of audio |
|---|---|---|
| 5 s | 46.9 s | 9.4 s |
| 10 s | 96.7 s | 9.7 s |
| 20 s | 226.0 s | 11.3 s |

Loudness of 8 real clips from 8 different prompts (5 s each, fixed seeds):

| | Peak cap only (first try) | With the peak limiter |
|---|---|---|
| Raw loudness before processing | -28.7 to -16.5 LUFS (a 12.1 LU spread) | same clips |
| Worst miss at -23 LUFS | 0.48 LU | 0.00 LU |
| Worst miss at -16 LUFS | 7.48 LU (6 of 8 clips fell short) | 0.49 LU |
| Highest sample | -1.00 dBFS | -1.00 dBFS |

How to read this:
- A 5 s station ID is ready in about half the time of a 10 s clip.
- My first version only lowered the gain when peaks were too high, so peaky clips could not reach -16 LUFS. The measurement caught it, and the look-ahead limiter fixed it.
- Checked on the real model: the progress bar gets exactly one update per generation step (153 steps for a 3 s clip, 253 for 5 s). A 10 s clip is now exactly 10.00 s (503 tokens); the original's 512 tokens gave 10.18 s.

How it was measured: the same laptop, the round-2 app in a clean install, the model on disk and the network off. Scripts: `benchmarks/bench_click_latency.py --seconds N` and `benchmarks/bench_loudness.py`.

### Tests

`tests/` has 38 fast tests and 2 real-model tests:
- **18 core tests** for `radio_imaging/`: clip length, seed, progress, loudness, limiter, fades, WAV, MP3, the prompt builder and the GPT step.
- **20 app tests**: they run the real Streamlit script with Streamlit's AppTest.
- Only the slow or outside parts are faked: loading the model and the OpenAI client. So these tests need no API key and no model download, and they run in about 20 seconds. GitHub Actions runs them on every push.
- **2 real-model tests** prove that the fakes behave like the real MusicGen. They are skipped unless `RUN_REAL_MODEL=1` is set.

Each feature test was written first and seen failing before the code existed.

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
RUN_REAL_MODEL=1 python -m pytest tests/test_real_model.py   # downloads the 2.4 GB model
```

To run the benchmarks (the first run downloads the 2.4 GB model):

```bash
python benchmarks/bench_click_latency.py radio_imaging_app.py --clicks 3 --seconds 10
python benchmarks/bench_loudness.py --seconds 5
```


#### 👥 How You Can Contribute
- **Feedback**: Share your experiences and improvement suggestions.
- **Use Cases**: Tell us about your process using the app.
- **Spread the Word**: Help others discover and use this tool.

## 🛠 Installation

### Requirements
- Python 3.11 (the pinned `torch==2.1.1` has no builds for Python 3.12 or newer)
- Streamlit, Transformers, PyTorch, SciPy, pyloudnorm, lameenc and openai (see `requirements.txt`)
- An OpenAI API key is optional: it is only needed for the GPT step

### Setup
1. Clone the repository.
2. Install required packages: `pip install -r requirements.txt`
3. Run the app: `streamlit run radio_imaging_app.py`

## Usage
1. Launch the Streamlit app.
2. Describe your audio: write it yourself, or use the prompt builder.
3. (Optional) Add an OpenAI API key in the sidebar and click 'Generate Prompt' for a richer description.
4. Choose the clip length, a seed (0 = random) and the loudness.
5. Click 'Generate Audio' and watch the progress bar.
6. Listen, then download WAV or MP3.

### 🌐 Access the Application
The original app by Bilsimaging (without this fork's changes) runs here:
https://radio-imaging-audio-generator.streamlit.app/

### 💖 Support
To support further development, consider donating at [Ko-fi](https://ko-fi.com/bilsimaging).

Thank you for your interest!
- Bilel Aroua

### License
[MIT License](LICENSE)

### 💬 Contact
Email: [contact@bilsimaging.com](mailto:contact@bilsimaging.com)  
More Info: [Bilsimaging](https://bilsimaging.com)
