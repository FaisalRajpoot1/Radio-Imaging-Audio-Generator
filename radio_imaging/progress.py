from transformers import StoppingCriteria


class StepProgress(StoppingCriteria):
    """Calls on_step(done, total) once per MusicGen generation step. It never
    stops generation.

    It is a stopping criterion, not a streamer: transformers calls stopping
    criteria at every step, but MusicGen's generate() in transformers 5.x no
    longer passes its streamer to the step loop.
    """

    def __init__(self, total_steps, on_step):
        self.total_steps = total_steps
        self.on_step = on_step
        self.done = 0

    def __call__(self, input_ids, scores, **kwargs):
        # On a GPU, transformers 5.x may run one step more and undo it later.
        if self.done < self.total_steps:
            self.done += 1
            self.on_step(self.done, self.total_steps)
        # A plain False works with both old (any()) and new (tensor |) transformers.
        return False

    def finish(self):
        """Report the last step if generation ended without reporting it.

        transformers 4.x checks stopping criteria with any(), which skips this
        one on the final step, once the max length criterion is met.
        """
        if self.done < self.total_steps:
            self.done = self.total_steps
            self.on_step(self.done, self.total_steps)
