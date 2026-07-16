from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConv3d(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv3d(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = self.up(x)

        dz = skip.shape[-3] - x.shape[-3]
        dy = skip.shape[-2] - x.shape[-2]
        dx = skip.shape[-1] - x.shape[-1]

        # Если размеры отличаются на один-два вокселя,
        # сначала дополняем меньший тензор.
        if dz < 0 or dy < 0 or dx < 0:
            z_start = max((-dz) // 2, 0)
            y_start = max((-dy) // 2, 0)
            x_start = max((-dx) // 2, 0)

            z_end = z_start + min(x.shape[-3], skip.shape[-3])
            y_end = y_start + min(x.shape[-2], skip.shape[-2])
            x_end = x_start + min(x.shape[-1], skip.shape[-1])

            x = x[
                :,
                :,
                z_start:z_end,
                y_start:y_end,
                x_start:x_end,
            ]

            dz = skip.shape[-3] - x.shape[-3]
            dy = skip.shape[-2] - x.shape[-2]
            dx = skip.shape[-1] - x.shape[-1]

        x = F.pad(
            x,
            [
                max(dx // 2, 0),
                max(dx - dx // 2, 0),
                max(dy // 2, 0),
                max(dy - dy // 2, 0),
                max(dz // 2, 0),
                max(dz - dz // 2, 0),
            ],
        )

        if x.shape[-3:] != skip.shape[-3:]:
            raise RuntimeError(
                f"Не удалось согласовать размеры: "
                f"x={x.shape}, skip={skip.shape}"
            )

        return self.conv(torch.cat([skip, x], dim=1))


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_features: int = 8,
    ):
        super().__init__()

        self.in_conv = DoubleConv3d(
            in_channels,
            base_features,
        )

        self.down1 = Down3d(
            base_features,
            base_features * 2,
        )

        self.down2 = Down3d(
            base_features * 2,
            base_features * 4,
        )

        self.down3 = Down3d(
            base_features * 4,
            base_features * 8,
        )

        self.bottleneck = Down3d(
            base_features * 8,
            base_features * 16,
        )

        self.up1 = Up3d(
            base_features * 16,
            base_features * 8,
            base_features * 8,
        )

        self.up2 = Up3d(
            base_features * 8,
            base_features * 4,
            base_features * 4,
        )

        self.up3 = Up3d(
            base_features * 4,
            base_features * 2,
            base_features * 2,
        )

        self.up4 = Up3d(
            base_features * 2,
            base_features,
            base_features,
        )

        self.out_conv = nn.Conv3d(
            base_features,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out_conv(x)


if __name__ == "__main__":
    model = UNet3D()
    sample = torch.randn(1, 4, 96, 96, 96)
    output = model(sample)
    print("Input:", sample.shape)
    print("Output:", output.shape)
