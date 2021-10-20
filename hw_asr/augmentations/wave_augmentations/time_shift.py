import numpy as np
import torch_audiomentations
from torch import Tensor
import torch

from hw_asr.augmentations.base import AugmentationBase


class TimeShift(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.start = kwargs.get("start")
        self.stop = kwargs.get("stop")

    def __call__(self, data: Tensor):
        start_ = int(np.random.uniform(self.start, self.stop))
        data = data.squeeze(0)
        if start_ >= 0:
            wav_time_shift = np.r_[data[start_:], np.random.uniform(-0.001, 0.001, start_)]
        else:
            wav_time_shift = np.r_[np.random.uniform(-0.001, 0.001, -start_), data[:start_]]
        if not torch.is_tensor(wav_time_shift):
            wav_time_shift = torch.from_numpy(wav_time_shift)
        return wav_time_shift.unsqueeze(0)
