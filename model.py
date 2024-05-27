import torch
import torch.nn as nn

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
    def __init__(self, in_channels, out_channels):
        super(ResidualSymmetricUNet3D, self).__init__()
        self.down1 = ResidualBlock(in_channels, 32)
        self.down2 = DownSample(32, 64)
        self.down3 = ResidualBlock(64, 64)
        self.down4 = DownSample(64, 128)
        self.down5 = ResidualBlock(128, 128)
        self.down6 = DownSample(128, 256)
        self.down7 = ResidualBlock(256, 256)
        self.down8 = DownSample(256, 512)
        self.center = ResidualBlock(512, 512)
        self.up1 = UpSample(512, 256)
        self.up2 = ResidualBlock(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = ResidualBlock(256, 128)
        self.up5 = UpSample(128, 64)
        self.up6 = ResidualBlock(128, 64)
        self.up7 = UpSample(64, 32)
        self.up8 = ResidualBlock(64, 32)
        self.out = nn.Conv3d(32, out_channels, kernel_size=1)

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