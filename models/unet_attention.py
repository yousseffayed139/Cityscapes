import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNetWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetWithAttention, self).__init__()

        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.middle = self.conv_block(512, 1024)

        self.att4 = AttentionBlock(512, 256, 128)
        self.up4 = self.upconv_block(1024, 512)

        self.att3 = AttentionBlock(256, 128, 64)
        self.up3 = self.upconv_block(512, 256)

        self.att2 = AttentionBlock(128, 64, 32)
        self.up2 = self.upconv_block(256, 128)

        self.att1 = AttentionBlock(64, in_channels, 16)
        self.up1 = self.upconv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))

        middle = self.middle(F.max_pool2d(enc4, kernel_size=2))

        dec4 = self.up4(middle)
        dec4 = self.att4(dec4, enc4)
        dec4 = torch.cat((dec4, enc4), dim=1)

        dec3 = self.up3(dec4)
        dec3 = self.att3(dec3, enc3)
        dec3 = torch.cat((dec3, enc3), dim=1)

        dec2 = self.up2(dec3)
        dec2 = self.att2(dec2, enc2)
        dec2 = torch.cat((dec2, enc2), dim=1)

        dec1 = self.up1(dec2)
        dec1 = self.att1(dec1, enc1)
        dec1 = torch.cat((dec1, enc1), dim=1)

        out = self.final(dec1)
        return out
