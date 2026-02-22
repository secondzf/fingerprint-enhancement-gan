import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

# ---------------------------
# 注意力模块：CBAM（通道+空间注意力）
# ---------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 确保 reduction 不会导致通道数为 0
        mid_channels = max(in_channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class CBAMBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ---------------------------
# UNet生成器
# ---------------------------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=32, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        # Encoder (下采样)
        self.enc1 = self._conv_block(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._conv_block(features, features * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._conv_block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._conv_block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # 中间层
        self.bottleneck = self._conv_block(features * 8, features * 16)

        # Decoder (上采样)
        self.up4 = nn.ConvTranspose2d(features * 16, features * 8, 2, stride=2)
        self.dec4 = self._conv_block(features * 16, features * 8)
        
        self.up3 = nn.ConvTranspose2d(features * 8, features * 4, 2, stride=2)
        self.dec3 = self._conv_block(features * 8, features * 4)
        
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, 2, stride=2)
        self.dec2 = self._conv_block(features * 4, features * 2)
        
        self.up1 = nn.ConvTranspose2d(features * 2, features, 2, stride=2)
        self.dec1 = self._conv_block(features * 2, features)

        self.out_conv = nn.Conv2d(features, out_channels, 1)
        self.tanh = nn.Tanh()

        # 注意力模块挂载在 Decoder 的每个阶段之后
        if self.use_attention:
            self.att4 = CBAMBlock(features * 8)
            self.att3 = CBAMBlock(features * 4)
            self.att2 = CBAMBlock(features * 2)
            self.att1 = CBAMBlock(features)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def center_crop(self, x, target_shape):
        """对 Skip Connection 的特征图进行裁剪以对齐尺寸"""
        _, _, h, w = x.shape
        th, tw = target_shape
        if h == th and w == tw:
            return x
        i = (h - th) // 2
        j = (w - tw) // 2
        return x[:, :, i:i + th, j:j + tw]

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder + Skip Connections + Attention
        d4 = self.up4(bottleneck)
        d4 = torch.cat([d4, self.center_crop(enc4, d4.shape[2:])], 1)
        d4 = self.dec4(d4)
        if self.use_attention: d4 = self.att4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, self.center_crop(enc3, d3.shape[2:])], 1)
        d3 = self.dec3(d3)
        if self.use_attention: d3 = self.att3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, self.center_crop(enc2, d2.shape[2:])], 1)
        d2 = self.dec2(d2)
        if self.use_attention: d2 = self.att2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, self.center_crop(enc1, d1.shape[2:])], 1)
        d1 = self.dec1(d1)
        if self.use_attention: d1 = self.att1(d1)

        out = self.out_conv(d1)
        
        # 最后的插值确保输出尺寸严格符合 Config 定义
        if out.shape[2:] != Config.img_size:
            out = F.interpolate(out, size=Config.img_size, mode='bilinear', align_corners=True)
            
        return self.tanh(out)

# ---------------------------
# 实例化模型
# ---------------------------
def get_generator():
    return UNetGenerator(use_attention=True).to(Config.device)

# 快速验证尺寸
if __name__ == "__main__":
    net = get_generator()
    test_data = torch.randn(1, 1, 768, 768).to(Config.device)
    res = net(test_data)
    print(f"输入: {test_data.shape} -> 输出: {res.shape}")