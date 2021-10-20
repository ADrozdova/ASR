import random

import librosa as lr
import torch
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class PitchShift(AugmentationBase):
    def __init__(self, **kwargs):
        self.steps = kwargs.get("steps")
        self.sampling_rate = kwargs.get("sampling_rate", 16000)

    def __call__(self, data: Tensor):
        n_steps = float(random.randint(-self.steps, self.steps))
        data = data.squeeze(0).numpy()
        return torch.from_numpy(lr.effects.pitch_shift(data, self.sampling_rate, n_steps=n_steps)).unsqueeze(0)
