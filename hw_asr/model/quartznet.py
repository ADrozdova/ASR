from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, bias=False,
                 separable=False, activation="relu"):
        super().__init__()
        if separable:
            layers = [
                nn.Conv1d(in_channels, in_channels, kernel_size,
                          stride=stride, dilation=dilation, padding=padding, bias=bias,
                          groups=in_channels),
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=1, dilation=1, padding=0, bias=bias)
            ]
        else:
            layers = [
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          stride=stride, dilation=dilation, padding=padding, bias=bias)
            ]
        layers.append(nn.BatchNorm1d(out_channels))
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class QuartzNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_count=3, kernel_size=11, stride=1, residual=True, dilation=1,
                 activation='relu', separable=False):
        super().__init__()
        padding_val = (dilation * kernel_size) // 2 - 1 if dilation > 1 else kernel_size // 2
        layers = []
        channels = in_channels
        for i in range(block_count):
            if i + 1 == block_count:
                activation = ""
            layers.append(
                ConvBNReLU(
                    channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding_val,
                    separable=separable,
                    activation=activation
                ))
            channels = out_channels
        self.residual = residual
        if residual:
            self.res_layer = ConvBNReLU(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                dilation=dilation,
                padding=padding_val,
                separable=False,
                activation=activation
            )
        self.out = nn.ReLU(inplace=True)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        output = self.net(x)
        if self.residual:
            res = self.res_layer(x)
            output += res
        return self.out(output)


class QuartzNet(BaseModel):
    def __init__(self, n_feats, n_class, n_blocks, repeat, channels, kernel_sz, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        layers = [ConvBNReLU(n_feats, channels[0], kernel_sz[0], stride=2)]  # C_1
        for i in range(n_blocks):
            layers.append(QuartzNetBlock(channels[i], channels[i + 1], repeat, kernel_sz[i + 1], separable=True))  # B
        layers.append(ConvBNReLU(channels[-3], channels[-2], kernel_sz[-3]))  # C_2
        layers.append(ConvBNReLU(channels[-2], channels[-1], kernel_sz[-2]))  # C_3
        layers.append(ConvBNReLU(channels[-1], n_class, kernel_sz[-1], dilation=2, bias=True))  # C_4
        self.net = Sequential(*layers)

    def forward(self, spectrogram, *args, **kwargs):
        output = self.net(spectrogram)
        return {"logits": output}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
