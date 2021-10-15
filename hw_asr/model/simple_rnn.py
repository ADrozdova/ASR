from torch import nn
from torch.nn import Sequential
from torch import zeros, unsqueeze

from hw_asr.base import BaseModel


class SimpleRnnModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, n_layers=2, dropout=0.25, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.n_layers = n_layers
        self.fc_hidden = fc_hidden
        self.rnn = nn.LSTM(
            n_feats, fc_hidden, num_layers=n_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )

        self.fc = nn.Linear(2 * fc_hidden, n_class)
        self.layers = Sequential(
            self.rnn,
            self.fc
        )

    def forward(self, spectrogram, *args, **kwargs):
        output, _ = self.rnn(spectrogram)
        output = output.view(spectrogram.size(0), -1, 2 * self.fc_hidden)
        return {"logits": self.fc(output)}

    def init_hidden(self, batch_size):
        hidden = zeros(self.n_layers, batch_size, self.fc_hidden)
        return hidden

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
