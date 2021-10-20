from torch import Tensor
from torchaudio.transforms import TimeStretch, TimeMasking, FrequencyMasking

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.sequential import SequentialAugmentation


class SpecAugment(AugmentationBase):
    def __init__(self, **kwargs):
        fixed_rate = kwargs.get("fixed_rate")
        time_mask_param = kwargs.get("time_mask_param")
        freq_mask_param = kwargs.get("freq_mask_param")
        self._aug = SequentialAugmentation([TimeMasking(time_mask_param=time_mask_param),
                                            FrequencyMasking(freq_mask_param=freq_mask_param)])

    def __call__(self, data: Tensor):
        return self._aug(data)
