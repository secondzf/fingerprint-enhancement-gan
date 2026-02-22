import torch
import torch.nn as nn
from config import Config

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1, features=64):
        super().__init__()
        # 这里的 in_channels 默认为 1，适配 Config 中的指纹灰度图设定
        self.net = nn.Sequential(
            # 第一层：下采样 [B, 1, 768, 768] -> [B, 64, 384, 384]
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第二层：下采样 [B, 64, 384, 384] -> [B, 128, 192, 192]
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第三层：下采样 [B, 128, 192, 192] -> [B, 256, 96, 96]
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第四层：保持尺寸 [B, 256, 96, 96] -> [B, 512, 96, 96]
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 最终映射层：输出单通道的判别矩阵
            # 输出大小约为 [B, 1, 95, 95]
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        """
        x 可能是生成图，也可能是真实图
        返回的是一个 Patch 矩阵，每个像素代表对应输入区域的“真实度”
        """
        return self.net(x)

# 快速验证脚本
if __name__ == "__main__":
    # 模拟一张 768x768 的指纹图
    model = PatchDiscriminator().to(Config.device)
    test_input = torch.randn(1, 1, 768, 768).to(Config.device)
    output = model(test_input)
    print(f"输入尺寸: {test_input.shape}")
    print(f"判别器输出尺寸: {output.shape}")