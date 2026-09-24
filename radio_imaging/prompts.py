import openai

# Checked on OpenAI's models page in September 2026. gpt-3.5-turbo-16k was shut
# down on 13 Sep 2024 and gpt-3.5-turbo shuts down on 23 Oct 2026.
GPT_MODELS = ["gpt-6-luna", "gpt-6-astra"]

# Choices for the free prompt builder.
GENRES = ["pop", "rock", "electronic", "hip hop", "jazz", "orchestral", "ambient"]
MOODS = ["upbeat", "energetic", "calm", "dramatic", "warm", "dark"]
INSTRUMENTS = ["synth", "drums", "piano", "guitar", "bass", "brass", "strings", "claps", "whoosh effects"]


class ModelUnavailableError(Exception):
    pass


def build_prompt(genre, mood, instruments, bpm, extra=""):
    """A short, concrete description, the style MusicGen works best with."""
    text = f"{mood} {genre} radio jingle"
    if instruments:
        listed = instruments[0] if len(instruments) == 1 else ", ".join(instruments[:-1]) + " and " + instruments[-1]
        text += f" with {listed}"
    text += f", {bpm} bpm"
    if extra.strip():
        text += f". {extra.strip()}"
    return text


def improve_with_gpt(api_key, model, idea):
    """Ask GPT to turn a rough idea into a detailed audio description."""
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Describe a radio imaging audio piece based on: {idea}"}],
        )
    except openai.NotFoundError as error:
        raise ModelUnavailableError(
            f"The OpenAI model '{model}' is not available. It may have been retired. Please pick another model."
        ) from error
    return response.choices[0].message.content.strip()
