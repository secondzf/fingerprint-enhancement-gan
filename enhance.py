import os
import torch
import numpy as np
from PIL import Image
from generator import get_generator  # 确保加载带 CBAM 注意力的模型
from config import Config

def main():
    # -------------------- 1. 路径与设备配置 --------------------
    device = Config.device
    
    # 输入：预处理后的 Prelatent 图 (由 preprocess.py 生成)
    input_dir = Config.aftrain_latent_root 
    # 输出：GAN 增强后的最终成品
    output_dir = Config.enhanced_output_dir
    
    # 模型路径：自动指向 aftrain 阶段生成的最佳模型
    model_path = os.path.join(Config.save_models_dir, "stage_aftrain", "generator_stage_aftrain_best.pth")
    
    os.makedirs(output_dir, exist_ok=True)

    # -------------------- 2. 加载模型 --------------------
    # get_generator() 会返回带有 use_attention=True 的 UNet
    generator = get_generator().to(device)

    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 成功加载增强模型: {model_path}")
    else:
        # 如果找不到 aftrain 模型，尝试加载 stage3 的模型作为兜底
        alt_path = os.path.join(Config.save_models_dir, "stage3", "generator_stage3_best.pth")
        if os.path.exists(alt_path):
            generator.load_state_dict(torch.load(alt_path, map_location=device))
            print(f"⚠️ 未找到 aftrain 模型，已加载 Stage 3 模型: {alt_path}")
        else:
            print(f"❌ 错误：未找到任何模型权重文件。")
            return

    generator.eval()

    # -------------------- 3. 核心处理工具 --------------------

    def fingerprint_normalization(img_np):
        """指纹特征标准化：确保推理时的对比度分布与训练时 (norm_M0, norm_VAR0) 一致"""
        img_np = img_np.astype(np.float32)
        M = np.mean(img_np)
        VAR = max(np.var(img_np), 1e-6)
        diff = img_np - M
        # 使用 Config 中定义的 100 和 50
        term = np.sqrt(Config.norm_VAR0 * (diff**2) / VAR)
        normalized = np.where(img_np > M, Config.norm_M0 + term, Config.norm_M0 - term)
        return np.clip(normalized, 0, 255).astype(np.uint8)

    def preprocess_tensor(img_pil):
        """Resize -> Normalization -> Tensor Mapping"""
        # 尺寸对齐 (768, 768)
        img = img_pil.resize(Config.img_size, Image.Resampling.LANCZOS)
        
        # 特征标准化
        img_np = np.array(img)
        norm_np = fingerprint_normalization(img_np)

        # 归一化到 [-1, 1] 供 Generator 使用
        t = torch.from_numpy(norm_np).float().unsqueeze(0).unsqueeze(0)
        t = (t / 127.5) - 1.0 
        return t.to(device)

    def postprocess_img(tensor):
        """Tensor [-1, 1] -> PIL Image [0, 255]"""
        t = tensor.detach().cpu().squeeze()
        t = (t + 1.0) / 2.0  # 映射回 [0, 1]
        t = torch.clamp(t, 0, 1)
        img_np = (t.numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np)

    # -------------------- 4. 执行批量推理 --------------------
    files = sorted([f for f in os.listdir(input_dir)
                    if f.lower().endswith(("png","jpg","bmp","jpeg"))])

    if not files:
        print(f"⚠️ 在 {input_dir} 中未找到待处理图像，请先运行 preprocess.py")
        return

    print(f"🚀 正在增强推理，共 {len(files)} 张图片...")

    with torch.inference_mode(): 
        for name in files:
            try:
                # 1. 读取（此时应该是 Gabor 处理过的灰度图）
                raw_img = Image.open(os.path.join(input_dir, name)).convert("L")
                
                # 2. 预处理与模型前向传播
                input_tensor = preprocess_tensor(raw_img)
                enhanced_tensor = generator(input_tensor)
                
                # 3. 后处理与保存
                result_img = postprocess_img(enhanced_tensor)
                result_img.save(os.path.join(output_dir, name))
                
                # 可选：保存对比图以便观察 GAN 的修复效果
                # compare_w = Config.img_size[0] * 2
                # compare_img = Image.new('L', (compare_w, Config.img_size[1]))
                # compare_img.paste(raw_img.resize(Config.img_size), (0, 0))
                # compare_img.paste(result_img, (Config.img_size[0], 0))
                # compare_img.save(os.path.join(output_dir, f"cmp_{name}"))

                print(f"成功增强并保存: {name}")
            except Exception as e:
                print(f"❌ 处理失败 {name}: {e}")

    print(f"\n✨ 增强任务全部完成！")
    print(f"📂 结果目录: {output_dir}")

if __name__ == "__main__":
    main()