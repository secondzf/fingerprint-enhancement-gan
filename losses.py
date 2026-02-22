import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import Config

# --------------------
# 1. 基础 L1 损失 (用于像素级还原)
# --------------------
criterion_l1 = nn.L1Loss()

# --------------------
# 2. SSIM 结构相似度 (维持脊线连贯性)
# --------------------
def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    """
    img: Tensor range [-1, 1], shape [B, 1, H, W]
    """
    # 归一化到 [0, 1] 用于计算
    img1 = (img1 + 1) / 2.0
    img2 = (img2 + 1) / 2.0

    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2

    ssim_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = ssim_n / (ssim_d + 1e-8)
    return ssim_map.mean()

# --------------------
# 3. 轻量化感知损失 (VGG11)
# --------------------
class LitePerceptualLoss(nn.Module):
    def __init__(self, device=Config.device):
        super().__init__()
        try:
            # 提取 VGG11 前几层特征
            vgg = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1).features[:12]
        except:
            vgg = models.vgg11(pretrained=True).features[:12]
            
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.to(device)
        self.device = device

    def forward(self, x, y):
        # 确保输入是 float 并归一化
        x = (x + 1) / 2.0
        y = (y + 1) / 2.0
        
        # 灰度转三通道以匹配 VGG
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
            
        fx = self.vgg(x.to(self.device))
        fy = self.vgg(y.to(self.device))
        return F.l1_loss(fx, fy)

# --------------------
# 4. 方向场损失 (指纹任务核心：约束脊线流向)
# --------------------
def orientation_field(img, kernel_size=9):
    """
    计算指纹方向场向量
    """
    img = img.float()
    if img.min() < -0.1:
        img = (img + 1) / 2.0

    device = img.device
    dtype = img.dtype

    # 使用 dtype=dtype 解决 LongTensor 与 FloatTensor 不匹配的错误
    kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=device, dtype=dtype)
    ky = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=device, dtype=dtype)

    gx = F.conv2d(img, kx, padding=1)
    gy = F.conv2d(img, ky, padding=1)

    # 局部平滑
    pad = kernel_size // 2
    avg_pool = nn.AvgPool2d(kernel_size, stride=1, padding=pad)

    gxx = avg_pool(gx * gx)
    gyy = avg_pool(gy * gy)
    gxy = avg_pool(gx * gy)

    # 计算方向角 theta
    theta = 0.5 * torch.atan2(2 * gxy, gxx - gyy + 1e-8)
    
    # 转换为单位向量图
    vx = torch.cos(theta)
    vy = torch.sin(theta)

    return torch.cat([vx, vy], dim=1)

def orientation_loss(fake, real):
    vf = orientation_field(fake)
    vr = orientation_field(real)
    return F.l1_loss(vf, vr)

# --------------------
# 5. 对抗损失 (GAN)
# --------------------
class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, pred, target_is_real):
        if target_is_real:
            target = torch.ones_like(pred)
        else:
            target = torch.zeros_like(pred)
        return self.criterion(pred, target)

# 预实例化常用损失函数供 train_stage.py 调用
criterion_gan = GANLoss()