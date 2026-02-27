"""Model architecture class."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Basic convolution block."""

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize convolution block."""
        super().__init__()
        # Conv -> ReLU -> Conv -> ReLU
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward pass."""
        return self.block(x)


class SeismicUNet(nn.Module):
    """Unet architecture for out seismic dataset."""

    def __init__(self):
        """Initialize UNet architecture."""
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(70, 64)
        self.enc2 = ConvBlock(64, 128)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))

        # Bottleneck layer
        self.bottleneck = ConvBlock(128, 256)

        # Decoder
        self.up1 = nn.ConvTranspose2d(
            256, 128, kernel_size=(1, 2), stride=(1, 2)
        )
        self.dec1 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128, 64, kernel_size=(1, 2), stride=(1, 2)
        )
        self.dec2 = ConvBlock(128, 64)

        # Final layer to project velocity map
        # Collapse time & source information into spatial velocity grid
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((70, 70)),  # force output spatial size
            nn.Conv2d(32, 1, kernel_size=1),  # final velocity map
        )

    def forward(self, x):
        """Model forward pass."""
        # Encoder (Input shape) -> (batch,70,5,1000)
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        u1 = self.up1(b)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))

        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))

        # output
        out = self.final_conv(d2)  # (B, 1, 70, 70)

        return out
