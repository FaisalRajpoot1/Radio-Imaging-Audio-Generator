"""Test doubles shared by the test files.

FakeMusicgen mirrors the real MusicgenForConditionalGeneration: generate() gets
the processor's tensors, max_new_tokens, and optional stopping criteria and
streamer. Like transformers' generate + sample(), it calls the stopping
criteria once per generated step, and it samples with torch's random
generator. Like transformers 5.x, it hands the streamer only the prompt ids:
5.x's MusicGen generate() no longer passes the streamer to its step loop.
With skips_last_step_report=True it behaves like transformers 4.x, whose any()
skips custom stopping criteria on the final step, once max length is reached.
It returns [batch, channels, samples] like the real model.

FakeOpenAI stands in for openai.OpenAI (an outside, paid service). Its replies
are the openai library's real ChatCompletion objects.
"""
from types import SimpleNamespace

import torch
from openai.types.chat import ChatCompletion

SAMPLE_RATE = 32_000
FRAME_RATE = 50
CODEBOOKS = 4
SAMPLES_PER_FRAME = SAMPLE_RATE // FRAME_RATE  # 640
GPT_REPLY = "Bright brass hit with a whoosh, 120 BPM"


class FakeProcessor:
    """Stands in for the MusicGen processor and records every text it gets."""

    def __init__(self):
        self.texts = []

    def __call__(self, text, padding, return_tensors):
        self.texts.append(list(text))
        rows = len(text)
        return {
            "input_ids": torch.ones((rows, 4), dtype=torch.long),
            "attention_mask": torch.ones((rows, 4), dtype=torch.long),
        }


class FakeMusicgen:
    config = SimpleNamespace(
        audio_encoder=SimpleNamespace(sampling_rate=SAMPLE_RATE, frame_rate=FRAME_RATE),
        decoder=SimpleNamespace(num_codebooks=CODEBOOKS),
    )

    def __init__(self, device=torch.device("cpu"), skips_last_step_report=False):
        self.device = device
        self.skips_last_step_report = skips_last_step_report
        self.moved_to = []
        self.got_progress_hook = False
        self.input_device = None

    def to(self, device):
        self.moved_to.append(str(device))
        return self

    def generate(self, input_ids, attention_mask, max_new_tokens, stopping_criteria=None, streamer=None):
        self.input_device = input_ids.device
        rows = input_ids.shape[0]
        if streamer is not None:
            streamer.put(torch.zeros((rows * CODEBOOKS, 1), dtype=torch.long))  # prompt ids only
        if stopping_criteria is not None:
            self.got_progress_hook = True
            ids = torch.zeros((rows * CODEBOOKS, 1), dtype=torch.long)
            for _ in range(max_new_tokens - 1 if self.skips_last_step_report else max_new_tokens):
                stopping_criteria(ids, None)
        frames = max_new_tokens - (CODEBOOKS - 1)
        return (torch.rand(rows, 1, frames * SAMPLES_PER_FRAME) * 2 - 1) * 0.3


def chat_completion(text, model):
    return ChatCompletion.model_validate({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    })


class FakeOpenAI:
    """Stands in for openai.OpenAI and records what the app sends."""

    requests = []
    keys = []
    fail_with = None

    @classmethod
    def reset(cls):
        cls.requests, cls.keys, cls.fail_with = [], [], None

    def __init__(self, api_key):
        FakeOpenAI.keys.append(api_key)
        self.chat = self
        self.completions = self

    def create(self, model, messages):
        FakeOpenAI.requests.append({"model": model, "messages": messages})
        if FakeOpenAI.fail_with:
            raise FakeOpenAI.fail_with
        return chat_completion(f"  {GPT_REPLY}\n", model)
