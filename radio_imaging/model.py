import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration, StoppingCriteriaList

from radio_imaging.progress import StepProgress

MODEL_ID = "facebook/musicgen-small"


def load_musicgen():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    musicgen_model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)
    return processor, musicgen_model


def tokens_for(seconds, musicgen_model):
    """Number of MusicGen tokens for a clip of the given length.

    MusicGen makes frame_rate audio frames per second (50), and its codebook
    delay pattern needs num_codebooks - 1 extra steps (3). The original app's
    512 tokens therefore gave a 10.18 s clip.
    """
    frame_rate = musicgen_model.config.audio_encoder.frame_rate
    delay_steps = musicgen_model.config.decoder.num_codebooks - 1
    return round(seconds * frame_rate) + delay_steps


def generate(processor, musicgen_model, text, seconds=10, seed=None, on_step=None, variations=1):
    """Make `variations` clips from one text prompt.

    Returns (sampling_rate, clips), where each clip is a 1-D float32 numpy array.
    on_step(done, total) is called once per generation step, for progress bars.
    """
    if seed is not None:
        torch.manual_seed(seed)
    inputs = processor(text=[text] * variations, padding=True, return_tensors="pt")
    # The model may sit on a GPU; its inputs must be on the same device.
    inputs = {name: tensor.to(musicgen_model.device) for name, tensor in inputs.items()}
    max_new_tokens = tokens_for(seconds, musicgen_model)
    progress = StepProgress(max_new_tokens, on_step) if on_step else None
    audio_values = musicgen_model.generate(**inputs, max_new_tokens=max_new_tokens,
                                           stopping_criteria=StoppingCriteriaList([progress]) if progress else None)
    if progress:
        progress.finish()
    sampling_rate = musicgen_model.config.audio_encoder.sampling_rate
    return sampling_rate, [audio_values[i, 0].cpu().numpy() for i in range(variations)]
