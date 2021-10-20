import os

import numpy as np
import torch
import torchaudio
from google_drive_downloader import GoogleDriveDownloader as gdd
from numpy.random import randint
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase

GOOGLE_DRIVE_FILE_ID = "1Fufd3QVrTFLZOHXf0LuURCHHRMyMCs0x"


class AddNoise(AugmentationBase):
    def __init__(self, **kwargs):
        self.dir = "data/noises"
        if not os.path.exists(self.dir):
            zip_file = './data/noises.zip'
            gdd.download_file_from_google_drive(file_id=GOOGLE_DRIVE_FILE_ID,
                                                dest_path=zip_file,
                                                unzip=True)
            os.remove(zip_file)
        self.bg_files = os.listdir(self.dir)
        self.sr = kwargs.get("sr", 16000)

    def __call__(self, data: Tensor):
        # source: https://www.kaggle.com/haqishen/augmentation-methods-for-audio/notebook
        data = data.squeeze(0)
        chosen_bg_file = self.bg_files[randint(len(self.bg_files))]

        filepath = os.path.join(self.dir, chosen_bg_file)
        bg, sr = torchaudio.load(filepath)
        bg = bg.squeeze(0)
        if bg.shape[0] >= data.shape[0]:
            noise = bg[:len(data)]
        else:
            noise = torch.zeros(data.shape)
            start_ = np.random.randint(noise.shape[0] - bg.shape[0])
            noise[start_:start_ + bg.shape[0]] = bg

        wav_with_bg: Tensor = data * np.random.uniform(0.8, 1.2) + noise * np.random.uniform(0, 0.1)
        return wav_with_bg.unsqueeze(0)
