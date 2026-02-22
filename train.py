import torch
import os
from config import Config
from dataset import ProgressiveDataset
from generator import get_generator # 使用之前定义的 get_generator 确保获取带 CBAM 的模型
from discriminator import PatchDiscriminator
from train_stage import train_single_stage

# 1. 初始化设备
device = Config.device

# 2. 实例化模型
# get_generator() 内部已包含 UNetGenerator(use_attention=True)
generator = get_generator().to(device)
discriminator = PatchDiscriminator().to(device)

print(f"--- 指纹增强 GAN 训练系统启动 ---")
print(f"设备: {device} | 图像目标尺寸: {Config.img_size}")
print("使用随机初始化模型，开启渐进式多阶段训练策略。")

# 3. 定义训练计划
# 前三个是预训练阶段(数字 ID)，最后一个是后训练阶段(字符串 ID)
training_plan = [1, 2, 3, 'aftrain']

for stage_id in training_plan:
    # 获取当前阶段配置
    cfg = Config.get_stage_config(stage_id)
    if cfg is None:
        print(f"跳过阶段 {stage_id}：在 Config 中未找到配置。")
        continue

    print(f"\n" + "="*50)
    print(f"▶️ 启动阶段 {stage_id}: {cfg['name']}")
    print(f"预期 Epochs: {cfg['epochs']} | 是否开启旋转增强: {cfg.get('rotation_aug', False)}")
    print("="*50)

    # 4. 实例化当前阶段的数据集
    # 之前修改过的 ProgressiveDataset 只需要 stage_id 即可自动识别路径
    train_dataset = ProgressiveDataset(stage_id=stage_id)
    
    # 5. 调用核心训练函数
    # 注意：train_single_stage 内部会自动处理 optimizer 的创建和 loss 实例化
    try:
        train_single_stage(
            generator=generator, 
            discriminator=discriminator, 
            stage_idx=stage_id, 
            stage_config=cfg, 
            train_dataset=train_dataset, 
            device=device
        )
        print(f"✅ 阶段 {stage_id} 训练圆满完成。")
    except Exception as e:
        print(f"❌ 阶段 {stage_id} 训练期间发生错误: {e}")
        # 如果某阶段失败，通常建议停止，防止后续阶段在错误的权重上浪费时间
        break

print("\n🎉 所有计划阶段已执行完毕！请在结果目录查看生成的样本和模型。")