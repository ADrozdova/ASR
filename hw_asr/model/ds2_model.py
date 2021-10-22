from torch import nn
import torch.nn.functional as F

from hw_asr.base import BaseModel


class ResidualCNN(nn.Module):
    """Residual CNN https://arxiv.org/pdf/1603.05027.pdf
        with layer norm instead of batch norm """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()
        self.rcnn = nn.Sequential(
            nn.LayerNorm(n_feats),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2),
            nn.LayerNorm(n_feats),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        )

    def forward(self, x):
        return x + self.rcnn(x)


class BGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BGRU, self).__init__()

        self.bgru = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.bgru(x)
        x = self.dropout(x)
        return x


class DS2(BaseModel):
    """Model Inspired by DeepSpeech 2"""
    def __init__(self, n_feats, n_class, n_cnn, n_rnn, hidden, stride=2, dropout=0.1, *args, **kwargs):
        super(DS2, self).__init__(n_feats, n_class, *args, **kwargs)
        n_feats = n_feats // 2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3 // 2)

        layers = []
        for i in range(n_cnn):
            layers.append(ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats))
        self.rcnn = nn.Sequential(*layers)

        self.fc = nn.Linear(n_feats * 32, hidden)

        layers = []
        for i in range(n_rnn):
            layers.append(BGRU(rnn_dim=hidden if i == 0 else hidden * 2, hidden_size=hidden,
                               dropout=dropout, batch_first=i == 0))
        self.bgru_layers = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        spectrogram = spectrogram.unsqueeze(1)
        x = self.cnn(spectrogram)
        x = self.rcnn(x)
        x = x.transpose(2, 3).contiguous()
        sz = x.size()
        x = x.view(sz[0], sz[1] * sz[2], sz[3])
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = self.bgru_layers(x)
        return self.classifier(x)

    def transform_input_lengths(self, input_lengths):
        return (input_lengths + 1) // 2