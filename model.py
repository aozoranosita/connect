import torch
import torch.nn as nn
from typing import Optional, List

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        residual = out
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ResidualSymmetricUNet3D(nn.Module):
    def __init__(self, in_channels: int = 1,
                 out_channels: int = 3, #afinity map of 3d
                filters: List[int] = [16, 32, 64, 128, 256]):
        super(ResidualSymmetricUNet3D, self).__init__()
        self.down1 = ResidualBlock(in_channels, filters[0])
        self.down2 = DownSample(filters[0], filters[1])
        self.down3 = ResidualBlock(filters[1], filters[1])
        self.down4 = DownSample(filters[1],filters[2])
        self.down5 = ResidualBlock(filters[2], filters[2])
        self.down6 = DownSample(filters[2], filters[3])
        self.down7 = ResidualBlock(filters[3], filters[3])
        self.down8 = DownSample(filters[3], filters[4])
        self.center = ResidualBlock(filters[4], filters[4])
        self.up1 = UpSample(filters[4], filters[3])
        self.up2 = ResidualBlock(filters[4], filters[3])
        self.up3 = UpSample(filters[3], filters[2])
        self.up4 = ResidualBlock(filters[3], filters[2])
        self.up5 = UpSample(filters[2], filters[1])
        self.up6 = ResidualBlock(filters[2], filters[1])
        self.up7 = UpSample(filters[1], filters[0])
        self.up8 = ResidualBlock(filters[1], filters[0])
        self.out = nn.Conv3d(filters[0], out_channels, kernel_size=1)
        # for output, should I use bn and activatoin?

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        down6 = self.down6(down5)
        down7 = self.down7(down6)
        down8 = self.down8(down7)
        center = self.center(down8)
        up1 = self.up1(center)
        up2 = self.up2(torch.cat([up1, down7], dim=1))
        up3 = self.up3(up2)
        up4 = self.up4(torch.cat([up3, down5], dim=1))
        up5 = self.up5(up4)
        up6 = self.up6(torch.cat([up5, down3], dim=1))
        up7 = self.up7(up6)
        up8 = self.up8(torch.cat([up7, down1], dim=1))
        out = self.out(up8)
        return out