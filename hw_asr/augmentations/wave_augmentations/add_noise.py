import os

import librosa
import numpy as np
import torch
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
        chosen_bg_file = self.bg_files[randint(len(self.bg_files))]

        filepath = os.path.join(self.dir, chosen_bg_file)
        bg, sr = librosa.load(filepath, sr=self.sr)

        start_ = np.random.randint(bg.shape[0] - 16000)
        bg_slice = bg[start_: start_ + 16000]
        wav_with_bg: Tensor = data * torch.from_numpy(np.random.uniform(0.8, 1.2)) + \
                              torch.from_numpy(bg_slice * np.random.uniform(0, 0.1))
        return wav_with_bg
