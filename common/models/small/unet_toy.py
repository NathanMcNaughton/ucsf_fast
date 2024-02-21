import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class ToyUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ToyUNet, self).__init__()
        self.inc = ConvBlock(n_channels, 16)
        self.down = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(16, 32)
        self.up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec = ConvBlock(32, 16)
        self.outc = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)

        # Downsample
        x2 = self.down(x1)

        # Bottleneck
        x3 = self.bottleneck(x2)

        # Upsample
        x4 = self.up(x3)

        # Decoder (with skip connection)
        x5 = torch.cat([x4, x1], dim=1)
        x6 = self.dec(x5)

        # Output
        logits = self.outc(x6)
        return logits