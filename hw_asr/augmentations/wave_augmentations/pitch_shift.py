import librosa as lr
from torch import Tensor
import random
from hw_asr.augmentations.base import AugmentationBase


class PitchShift(AugmentationBase):
    def __init__(self, **kwargs):
        self.steps = kwargs.get("steps")
        self.sampling_rate = kwargs.get("sampling_rate", 16000)

    def __call__(self, data: Tensor):
        n_steps = float(random.randint(-self.steps, self.steps))
        return lr.effects.pitch_shift(data, self.sampling_rate, n_steps=n_steps)
