import os
import torch
from torchvision.utils import save_image
from config import Config

def save_samples(inputs, outputs, targets, save_dir, epoch, stage_idx, max_samples=4):
    """
    保存训练样本为拼接图片: 输入(退化图) + 生成(修复图) + 标签(清晰图)
    :param stage_idx: 1, 2, 3 或 'aftrain'
    :param max_samples: 每个 epoch 最多保存的样本数，防止磁盘溢出
    """
    os.makedirs(save_dir, exist_ok=True)

    # 确定保存数量，不大于当前 batch 也不大于最大限制
    num_to_save = min(len(inputs), max_samples)

    for i in range(num_to_save):
        # 统一处理归一化范围：从 [-1, 1] 映射到 [0, 1]
        def to_img(t):
            t = t.detach().cpu()
            if t.min() < 0:
                t = t * 0.5 + 0.5
            return torch.clamp(t, 0, 1)

        inp = to_img(inputs[i])
        out = to_img(outputs[i])
        tgt = to_img(targets[i])

        # 横向拼接: [C, H, W*3]
        concat = torch.cat([inp, out, tgt], dim=2)
        
        # 命名格式：stage_X_epoch_X_idx_X.png
        file_name = f'stage_{stage_idx}_epoch_{epoch:03d}_sample_{i}.png'
        save_path = os.path.join(save_dir, file_name)
        
        save_image(concat, save_path)

def save_model(model, save_dir, stage_idx, epoch, is_best=False):
    """
    保存模型权重
    :param is_best: 如果为 True，额外保存一份名为 best 的权重
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 基础文件名
    model_name = f'generator_stage_{stage_idx}_epoch_{epoch:03d}.pth'
    save_path = os.path.join(save_dir, model_name)
    
    # 保存当前权重
    torch.save(model.state_dict(), save_path)
    
    # 如果是最佳模型，复制一份
    if is_best:
        best_path = os.path.join(save_dir, f'generator_stage_{stage_idx}_best.pth')
        torch.save(model.state_dict(), best_path)
        print(f"⭐ 已更新阶段 {stage_idx} 的最佳模型权重")

def load_checkpoint(model, path):
    """
    辅助函数：加载断点/预训练权重
    """
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=Config.device)
        model.load_state_dict(state_dict, strict=False)
        print(f"📖 成功从 {path} 加载权重")
    else:
        print(f"⚠️ 未找到路径: {path}，跳过加载")
    return model