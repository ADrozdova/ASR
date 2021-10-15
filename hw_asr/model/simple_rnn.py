from torch import nn
from torch.nn import Sequential
from torch import zeros, unsqueeze

from hw_asr.base import BaseModel


class SimpleRnnModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, n_layers=2, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.n_layers = n_layers
        self.fc_hidden = fc_hidden
        self.rnn = nn.RNN(n_feats, fc_hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(fc_hidden, n_class)
        self.layers = Sequential(
            self.rnn,
            self.fc
        )

    def forward(self, spectrogram, *args, **kwargs):
        batch_sz = spectrogram.size(0)
        hidden = self.init_hidden(batch_sz)
        output, hidden = self.rnn(spectrogram, hidden)
        # output = output.contiguous().view(batch_sz, -1, self.fc_hidden)
        output = output.view(batch_sz, -1, self.fc_hidden)
        output = self.fc(output)
        return {"logits": output}

    def init_hidden(self, batch_size):
        hidden = zeros(self.n_layers, batch_size, self.fc_hidden)
        return hidden

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
