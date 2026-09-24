import random
import time
import streamlit as st

from radio_imaging.audio import LOUDNESS_TARGETS, finish_clip, to_mp3_bytes, to_wav_bytes
from radio_imaging.model import generate, load_musicgen as load_musicgen_uncached
from radio_imaging.prompts import (GENRES, GPT_MODELS, INSTRUMENTS, MOODS, ModelUnavailableError,
                                   build_prompt, improve_with_gpt)


# Copyright notice shown to people with every prompt. It is not sent to MusicGen.
copyright_notice = "\n\n© Created through Radio Imaging Audio Generator by Bilsimaging [WEBSITE](https://bilsimaging.com)"


# Load MusicGen once per server process. Every click, from every user, reuses it
# instead of reloading 2.4 GB of weights.
@st.cache_resource(show_spinner="Loading the MusicGen model (first run only)...")
def load_musicgen():
    return load_musicgen_uncached()


# Streamlit app setup
st.set_page_config(
    page_icon="https://soundboard.bilsimaging.com/faviconbilsimaging.png", 
    layout="wide",
    page_title='Radio Imaging Audio Generator Beta 0.1',
    initial_sidebar_state="expanded"
)

# Main Description and Header
st.markdown("""
    <h1 style=''>Radio Imaging Audio Generator 
    <span style='font-size: 24px; color: #FDC74A;'>Beta 0.1</span></h1>
    """, unsafe_allow_html=True)
st.write("Welcome to the Radio Imaging & MusicGen Ai audio generator. This web application allows you to easily create unique audio for your radio imaging projects or any music creators using AI technology.")
st.markdown("---")

# How to Use the App - Instructions
with st.expander('📘 How to Use This Web App?'):
    st.markdown('''
        To get started with creating your unique audio pieces, follow these simple steps:

        **1. Describe Your Audio**
        - Write your idea in the text area, or use the **prompt builder** (genre, mood, instruments, tempo). No OpenAI key is needed.

        **2. Improve It with GPT (optional)**
        - Add your **OpenAI API key** in the sidebar and click **'📄 Generate Prompt'** for a richer description. Review it to ensure it aligns with your vision.
        - Don't have an API key? Skip this step, or get one [here](https://platform.openai.com/account/api-keys).

        **3. Choose the Clip Settings**
        - Clip length (3–30 seconds), a seed (to make the same take again), and the loudness: broadcast (EBU R128, -23 LUFS) or online/podcast (-16 LUFS).

        **4. Generate Your Audio**
        - Hit **'▶ Generate Audio'**. The progress bar follows the real generation steps.

        **5. Playback and Download**
        - Play the audio directly in the app, and download it as WAV or MP3 for your projects.
    ''')


# Sidebar for user inputs
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API key (optional)", type="password", key="openai_api_key",
                                   help="Only needed for the optional GPT step.")
    st.caption("*No key? You can still make audio: write the description yourself or use the prompt builder. Get a key [here](https://platform.openai.com/account/api-keys).*")
    model = st.selectbox("OpenAI chat model", GPT_MODELS + ["Other…"], key="gpt_model",
                         help="The old gpt-3.5 models are retired or being retired by OpenAI.")
    if model == "Other…":
        model = st.text_input("Model name", key="gpt_model_other", help="Any current OpenAI chat model.")
    st.markdown("Check out our video tutorials on [YouTube](https://www.youtube.com/playlist?list=PLwEbW4bdYBSDe6qAJRFiWGyHSW-JR-B0_) for helpful guides on using this app!")
    st.markdown('''Made with ❤️ by [Bilsimaging](https://bilsimaging.com)''', unsafe_allow_html=True)
    st.caption("Fork with free mode, real progress and broadcast-ready audio by [Muhammad Faisal](https://github.com/FaisalRajpoot1/Radio-Imaging-Audio-Generator).")

# Guidelines for generating prompt and audio
st.markdown("""
    ### 💡 Steps to Generate Your Audio:
    1. **Describe your audio**: write it yourself, or use the prompt builder.
    2. **Generate Prompt (optional)**: let GPT write a richer description. This needs an OpenAI API key.
    3. **Choose the clip settings**: length, seed and loudness.
    4. **Generate Audio**: then play it, or download it as WAV or MP3.
""")

# Prompt input: write it yourself, or build it from a few choices (no OpenAI key needed)
st.markdown("## ✍🏻Write your Description")
describe_mode = st.radio("How do you want to describe it?", ["Write it myself", "Use the prompt builder"],
                         key="describe_mode", horizontal=True)
if describe_mode == "Write it myself":
    prompt = st.text_area("Enter your radio imaging draft idea prompt here", key="idea",
                          help="Describe the audio piece you want to create.")

    # Instructions for users
    st.info("👉🏻 Provide a detailed description of the audio you need, such as mood, instruments, and style. Example: A calm, soothing melody with soft piano for a morning show.")
else:
    genre_column, mood_column = st.columns(2)
    genre = genre_column.selectbox("Genre", GENRES, key="genre")
    mood = mood_column.selectbox("Mood", MOODS, key="mood")
    instruments = st.multiselect("Instruments", INSTRUMENTS, key="instruments")
    bpm = st.slider("Tempo (BPM)", 60, 180, 120, key="bpm")
    extra = st.text_input("Anything else? (optional)", key="extra")
    prompt = build_prompt(genre, mood, instruments, bpm, extra)
    st.info(f"👉🏻 Prompt for MusicGen: {prompt}")

# Generate Prompt Button with user confirmation and patience message
st.markdown("## 📝 Generate Prompt (optional)")
st.info("🚨 This optional step uses GPT to write a richer description. It needs an OpenAI API key; without one, skip it.")
if st.button("📄 Generate Prompt"):
    if not openai_api_key.strip():
        st.error("This step needs an OpenAI API key (add it in the sidebar). Or skip it: click 'Generate Audio' to use your own description.")
    elif not prompt.strip():
        st.error("Please describe your audio first.")
    else:
        with st.spinner("Generating your prompt... Please wait, this might take a few moments."):
            try:
                descriptive_text = improve_with_gpt(openai_api_key, model, prompt)

                # MusicGen gets only GPT's description; the copyright notice is for people
                st.session_state['generated_prompt'] = descriptive_text
                prompt_with_notice = descriptive_text + copyright_notice
                st.success("Your prompt has been successfully generated! Review the prompt below:")
                st.write(prompt_with_notice)

                # Download Button for the generated prompt
                st.download_button(
                    label="Download Prompt",
                    data=prompt_with_notice,
                    file_name="generated_prompt.txt",
                    mime="text/plain"
                )

            except ModelUnavailableError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"An error occurred: {e}")

st.markdown("---")



# Generate Audio Button with Load Management
st.markdown("## 🎶 Generate Audio")
st.info("🚨 Please be patient as generating audio can take some time. This might take a moment due to resource limits. Feel free to notify me if you encounter any issues.")   

# Clip settings
length_column, seed_column = st.columns(2)
seconds = length_column.slider("Clip length (seconds)", 3, 30, 10, key="seconds")
seed = seed_column.number_input("Seed (0 = random)", min_value=0, max_value=2**31 - 1, value=0, step=1, key="seed",
                                help="Use the seed shown after a take to make the same clip again.")
target_lufs = LOUDNESS_TARGETS[st.radio("Loudness", list(LOUDNESS_TARGETS), key="loudness", horizontal=True)]
if seconds > 15:
    st.caption("Long clips take much longer on a CPU.")

if st.button("▶ Generate Audio"):
    # GPT's description if the optional step was used, otherwise the user's own
    descriptive_text = st.session_state.get('generated_prompt') or prompt
    if not descriptive_text.strip():
        st.error("Please describe your audio first: write it, or use the prompt builder.")
    else:
        # Placeholder for server load check
        server_ready_for_audio_generation = True  # Replace with actual server load check logic

        if server_ready_for_audio_generation:
            take_seed = int(seed) or random.randint(1, 2**31 - 1)
            progress_bar = st.progress(0, text="Starting...")
            try:
                processor, musicgen_model = load_musicgen()
                start = time.perf_counter()

                def show_progress(done, total):
                    left = (time.perf_counter() - start) / done * (total - done)
                    progress_bar.progress(done / total, text=f"Generating: step {done} of {total}, about {left:.0f} s left")

                sampling_rate, [clip] = generate(processor, musicgen_model, descriptive_text,
                                                 seconds=seconds, seed=take_seed, on_step=show_progress)
                clip = finish_clip(clip, sampling_rate, target_lufs)
                progress_bar.progress(1.0, text=f"Done: {seconds} s of audio in {time.perf_counter() - start:.0f} s (seed {take_seed})")

                # Keep the audio in memory, so each user gets their own clip and nothing is left on the server
                wav_bytes = to_wav_bytes(clip, sampling_rate)
                st.success("Your audio has been successfully created! It was made from this description:")
                st.write(descriptive_text + copyright_notice)
                st.audio(wav_bytes, format="audio/wav")
                wav_column, mp3_column = st.columns(2)
                wav_column.download_button("Download WAV", data=wav_bytes,
                                           file_name="Bilsimaging_radio_imaging_output.wav", mime="audio/wav")
                mp3_column.download_button("Download MP3", data=to_mp3_bytes(clip, sampling_rate),
                                           file_name="Bilsimaging_radio_imaging_output.mp3", mime="audio/mpeg")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("The server is currently busy. Please try generating your audio again later.")



# Footer and Support Section
st.markdown("---")
st.markdown("## 🌐 Project Continuation and User Involvement")
st.markdown("✔️ This app is the next step in our project, following the Custom GPTs Radio Imaging and MusicGen AI. <br>It's tailored for radio producers and music creators, offering new levels of creativity and efficiency by Bilsimaging. [Try our GPTs](https://chat.openai.com/g/g-65x53n87E-radio-imaging-musicgen-ai).", unsafe_allow_html=True)
st.markdown("If you appreciate my deployment and wish to support me, please consider a donation. Your support helps me continue providing value. Thank you for joining me on this journey! - Bilel Aroua")
st.markdown("For support ☕ [Buy me a Coffee](https://ko-fi.com/bilsimaging).")
st.image('https://storage.ko-fi.com/cdn/brandasset/kofi_button_dark.png', width=300, caption="Project Bilsimaigng")

# Hide Streamlit branding
st.markdown("<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>", unsafe_allow_html=True)
