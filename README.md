
# 🌟Radio Imaging Audio Generator

> **This is a fork.** The original app is [Radio Imaging Audio Generator](https://github.com/bilsimaging/Radio-Imaging-Audio-Generator) by **Bilel Aroua** ([Bilsimaging](https://bilsimaging.com)), MIT License. The idea, the app and its design are his. This fork fixes speed and reliability problems and measures the result. See [What this fork changes](#what-this-fork-changes).

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

The app works the same for users. This fork fixes five problems in how it makes audio, and measures the result.

| Problem in the original | Fix in this fork |
|---|---|
| The 2.4 GB MusicGen model was loaded again on every click, for every user. | The model loads once per server (`st.cache_resource`). Every click reuses it. |
| A fake progress bar added 10 seconds of waiting to every click. | Removed. A spinner shows while the real work runs. |
| TensorFlow was installed and imported, but never used. | Removed from the code and from `requirements.txt`. |
| Every user's audio was written to one fixed WAV file on the server. | The audio stays in memory, one copy per user. Added a **Download Audio** button. |
| The copyright notice was sent to MusicGen as part of the music prompt. | MusicGen gets only GPT's description. The notice is still shown and saved with the prompt. |

### Measured results

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

### Tests

`tests/` has 7 tests. They run the real app script with Streamlit's AppTest. Only the slow or outside parts are faked: loading the model and the OpenAI call. So they need no API key and no model download, and they run in about 7 seconds.

Each test was written first and seen failing on the original code: 6 failed, and the 7th, which checks that the original copyright notice is still shown, passed on both.

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

To run the benchmark (the first run downloads the 2.4 GB model):

```bash
python benchmarks/bench_click_latency.py radio_imaging_app.py --clicks 3
```


#### 👥 How You Can Contribute
- **Feedback**: Share your experiences and improvement suggestions.
- **Use Cases**: Tell us about your process using the app.
- **Spread the Word**: Help others discover and use this tool.

## 🛠 Installation

### Requirements
- Python 3.11 (the pinned `torch==2.1.1` has no builds for Python 3.12 or newer)
- Streamlit
- Transformers
- SciPy
- PyTorch
- OpenAI API key

### Setup
1. Clone the repository.
2. Install required packages: `pip install -r requirements.txt`
3. Run the app: `streamlit run radio_imaging_app.py`

## Usage
1. Launch the Streamlit app.
2. Enter your OpenAI API key.
3. Select an OpenAI chat model.
4. Input a description for the audio piece.
5. Click 'Generate Audio'.
6. Listen and download the audio directly in the app.

### 🌐 Access the Application
Experience the Radio Imaging Audio Generator now: Access the Streamlit App here.
https://radio-imaging-audio-generator.streamlit.app/

### How to Use This Web App?
To get started with creating your unique audio pieces, follow these simple steps:
 **1. Enter OpenAI API Key**
 - In the sidebar, input your **OpenAI API key**. This is essential to access the GPT model for 
generating audio descriptions.
 - Don't have an API key? Get one for free [here](https://platform.openai.com/account/apikeys).
 **2. Select GPT Model**
 - Choose the desired GPT model from the dropdown in the sidebar. We recommend using 
**'gpt-3.5-turbo-16k'** for more detailed and rich descriptions.
 **3. Input Your Detailed Description**
 - Describe your audio idea in the text area provided. Be as detailed as possible to guide the AI 
effectively. This could include the mood, style, specific instruments, or any other relevant details.
 **4. Generate and Review the Prompt**
 - Click on **' Generate Prompt'** to create a descriptive prompt for your audio. Review it to 
ensure it aligns with your vision.
 **5. Generate Your Audio**
 - If you're satisfied with the prompt, hit **'▶ Generate Audio'**. This will process your request 
and create the audio piece based on the AI-generated description.
 **6. Playback and Download**
 - Once generated, you can play the audio directly within the app. If it meets your needs, feel 
free to download and use it in your projects.


### 💖 Support
To support further development, consider donating at [Ko-fi](https://ko-fi.com/bilsimaging).

Thank you for your interest!
- Bilel Aroua

### License
[MIT License](LICENSE)

### 💬 Contact
Email: [contact@bilsimaging.com](mailto:contact@bilsimaging.com)  
More Info: [Bilsimaging](https://bilsimaging.com)
