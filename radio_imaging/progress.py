from transformers.generation.streamers import BaseStreamer


class StepProgress(BaseStreamer):
    """Calls on_step(done, total) once per MusicGen generation step.

    generate() first passes the prompt ids to put(), then one put() per new
    step, then end(). Only the new steps are counted.
    """

    def __init__(self, total_steps, on_step):
        self.total_steps = total_steps
        self.on_step = on_step
        self.done = 0
        self._prompt_seen = False

    def put(self, value):
        if not self._prompt_seen:
            self._prompt_seen = True
            return
        self.done += 1
        self.on_step(self.done, self.total_steps)

    def end(self):
        pass
