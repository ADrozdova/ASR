import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class Gain(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.sample_rate = kwargs.get("sample_rate", 16000)
        self._aug = torch_audiomentations.Gain(p=kwargs.get("p"))

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x, self.sample_rate).squeeze(1)
