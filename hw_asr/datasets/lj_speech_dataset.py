import json
import logging
import os
import shutil
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_asr.base.base_dataset import BaseDataset
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)

URL_LINKS = {
    "LJSpeech": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
}


class LJSpeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        assert part in URL_LINKS

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading {part}")
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        wav_dir = Path(self._data_dir / "wavs")
        meta_file = open(self._data_dir / "metadata.csv", "r")
        for line in meta_file:
            idx, text, norm_text = line.strip().split("|")
            wav_path = wav_dir / f"{idx}.wav"
            info = torchaudio.info(str(wav_path))
            index.append({
                "path": str(wav_path.absolute().resolve()),
                "text": norm_text.lower(),
                "audio_len": info.num_frames / info.sample_rate,
            })
        return index


if __name__ == "__main__":
    text_encoder = CTCCharTextEncoder.get_simple_alphabet()
    config_parser = ConfigParser.get_default_configs()

    ds = LJSpeechDataset(
        "LJSpeech", text_encoder=text_encoder, config_parser=config_parser
    )
    item = ds[0]
    print(item)
