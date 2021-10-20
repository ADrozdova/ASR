import logging
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

DATASET_KEYS = ['spectrogram', 'text', 'text_encoded', 'text_encoded_length', 'spectrogram_length']


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    if len(dataset_items) == 0:
        return result_batch
    for key in DATASET_KEYS:
        result_batch[key] = []
    for item in dataset_items:
        result_batch['spectrogram'].append(item['spectrogram'].squeeze(0).T)
        result_batch['text_encoded'].append(item['text_encoded'].squeeze(0))
        result_batch['text'].append(item['text'])
        result_batch['text_encoded_length'].append(item['text_encoded'].shape[1])
        result_batch['spectrogram_length'].append(item['spectrogram'].shape[2])
    result_batch['spectrogram'] = pad_sequence(result_batch['spectrogram'], batch_first=True)
    result_batch['text_encoded'] = pad_sequence(result_batch['text_encoded'], batch_first=True)
    result_batch['text_encoded_length'] = torch.tensor(result_batch['text_encoded_length'])
    result_batch['spectrogram_length'] = torch.tensor(result_batch['spectrogram_length'])
    return result_batch
