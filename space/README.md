---
title: Radio Imaging Audio Generator
emoji: 📻
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Radio jingles and station IDs from text, with MusicGen
models:
  - facebook/musicgen-small
---

# 📻 Radio Imaging Audio Generator

Make radio jingles, station IDs and sweepers from a text description, with Meta's MusicGen running on a free GPU (ZeroGPU).

- **No OpenAI key needed**: write the description yourself, or use the prompt builder.
- **Up to 3 takes per click**, 3 to 30 seconds each, with a seed to make a take again.
- **Broadcast-ready loudness**: EBU R128 (-23 LUFS) or online (-16 LUFS), with a look-ahead peak limiter so no sample goes above -1 dBFS.
- **WAV and MP3 downloads.**
- Optional: let GPT write a richer description, with your own OpenAI key (used once, never stored).

Based on [Radio Imaging Audio Generator](https://github.com/bilsimaging/Radio-Imaging-Audio-Generator) by **Bilel Aroua** ([Bilsimaging](https://bilsimaging.com)), MIT License.
This version by **Muhammad Faisal**. Source code, tests and measurements: [GitHub](https://github.com/FaisalRajpoot1/Radio-Imaging-Audio-Generator/tree/fork-improvements).
