import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from losses import (
    criterion_l1, ssim, LitePerceptualLoss, orientation_loss, criterion_gan
)
from metrics import save_samples, save_model

def train_single_stage(generator, discriminator, stage_idx, stage_config, train_dataset, device):
    """
    核心训练函数
    注意：参数名 stage_idx 必须与 train.py 中的调用严格一致
    """
    generator.train()
    discriminator.train()

    # 1. 准备数据加载器
    dataloader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        num_workers=Config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 2. 优化器设置 (Adam 针对 GAN 的标准配置)
    lr_g = stage_config.get('lr_gen', Config.lr_gen)
    lr_d = stage_config.get('lr_dis', Config.lr_dis)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # 3. 损失函数实例化
    # LitePerceptualLoss 内部持有 VGG 权重，只需在每个 stage 开始时初始化一次
    perc_loss_func = LitePerceptualLoss(device=device)
    
    # 损失权重映射
    weights = {
        'l1': Config.lambda_l1,
        'perceptual': Config.lambda_perceptual,
        'ssim': 10.0,
        'orientation': 50.0,
        'adv': 2.0
    }

    # 4. 训练循环
    epochs = stage_config['epochs']
    for epoch in range(1, epochs + 1):
        total_G_loss = 0.0
        total_D_loss = 0.0
        
        # 使用 tqdm 显示进度条
        pbar = tqdm(dataloader, desc=f"Stage {stage_idx} | Epoch {epoch}/{epochs}")
        
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            # ---------------------------
            # (A) 更新生成器 (Generator)
            # ---------------------------
            optimizer_G.zero_grad()
            fake = generator(inputs)
            
            # 计算各项 Loss
            l_l1 = criterion_l1(fake, targets) * weights['l1']
            l_perc = perc_loss_func(fake, targets) * weights['perceptual']
            l_ssim = (1.0 - ssim(fake, targets)) * weights['ssim']
            l_orient = orientation_loss(fake, targets) * weights['orientation']
            
            # 对抗损失：希望判别器把 fake 判为真(True)
            pred_fake = discriminator(fake)
            l_adv = criterion_gan(pred_fake, True) * weights['adv']
            
            loss_G = l_l1 + l_perc + l_ssim + l_orient + l_adv
            loss_G.backward()
            optimizer_G.step()

            # ---------------------------
            # (B) 更新判别器 (Discriminator)
            # ---------------------------
            optimizer_D.zero_grad()
            
            # 判别真实图像
            pred_real = discriminator(targets)
            loss_D_real = criterion_gan(pred_real, True)
            
            # 判别生成图像 (使用 detach 避免更新 G)
            pred_fake_d = discriminator(fake.detach())
            loss_D_fake = criterion_gan(pred_fake_d, False)
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # 累加损失用于统计
            total_G_loss += loss_G.item()
            total_D_loss += loss_D.item()
            
            # 动态更新进度条显示
            pbar.set_postfix({
                "G_Loss": f"{loss_G.item():.3f}", 
                "D_Loss": f"{loss_D.item():.3f}"
            })

        # 5. 每个 Epoch 结束后的数据持久化
        # 建立对应 stage 的子目录
        current_model_dir = os.path.join(Config.save_models_dir, f"stage_{stage_idx}")
        current_sample_dir = os.path.join(Config.save_samples_dir, f"stage_{stage_idx}")
        
        # 如果是该阶段最后一个 Epoch，标记为 best 以供下一阶段加载
        is_best = (epoch == epochs)
        
        save_model(generator, current_model_dir, stage_idx, epoch, is_best=is_best)
        save_samples(inputs, fake, targets, current_sample_dir, epoch, stage_idx)

    # 清理显存缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()