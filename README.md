# ASR project barebones

## Installation guide
Install requirements.
```shell
pip install -r ./requirements.txt -q
```
Download model checkpoint and config in `default_test_model/checkpoint.pth`
   and `default_test_model/config.json`.
```shell
python3 download_checkpoint.py
```
Noises for augmentation are downloaded in `hw_asr/augmentations/wave_augmentations/add_noise.py` line `20`.

Pretrained KenLM and librispeech vocabulary are downloaded in `hw_asr/text_encoder/ctc_char_text_encoder.py` line `37`, `63`.

## Credits

this repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
