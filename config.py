import os
import torch

class Config:
    # ================== 1. 设备与基础路径 ==================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 工程与结果根目录
    base_dir = "/root/finger/finger"
    save_dir = "/root/finger/result"
    
    # ================== 2. 核心数据路径 (修复 AttributeError) ==================
    # 合成数据路径 (Stage 1-3 使用)
    # 这一部分存放的是用于基础模型训练的大规模指纹库
    train_root = os.path.join(base_dir, "data/train")  # 原始清晰图
    test_root = os.path.join(base_dir, "data/test")    # 验证集

    # 后训练 (aftrain) 阶段在训练时需要读取的成对数据路径
    # latent_root 存放的是后训练用的输入，label_root 存放的是对应的清晰目标
    aftrain_latent_root = os.path.join(base_dir, "data/latent")
    aftrain_label_root = os.path.join(base_dir, "data/label")
    
    # 辅助工具路径
    aftrain_raw_latent_root = "/root/autodl-tmp/1"      # 原始未处理图路径
    enhanced_output_dir = "/root/autodl-tmp/enhanced_results1"

    # ================== 3. 模型与样本保存 ==================
    save_samples_dir = os.path.join(save_dir, "samples")
    save_models_dir = os.path.join(save_dir, "models")

    # ================== 4. 图像参数与归一化 ==================
    img_size = (768, 768)
    norm_M0 = 100         # 均值基准
    norm_VAR0 = 50        # 方差基准

    # ================== 5. 训练全局基础参数 ==================
    batch_size = 1
    num_workers = 2
    
    # 默认优化器参数
    lr_gen = 3e-5
    lr_dis = 1.5e-6
    
    # 默认损失函数权重 (对应 train_single_stage 中的 weights 逻辑)
    lambda_l1 = 50.0
    lambda_perceptual = 25.0

# ================== 6. 渐进式 Stage 配置 (补全参数 + 旋转全开) ==================
    stage_configs = {
        1: {
            'stage': 1,
            'name': '轻度模糊',
            'epochs': 10,
            'rotation_aug': True,         # 开启旋转
            'blur_kernel_range': (3, 5),
            'blur_sigma_range': (0.3, 0.8),
            'noise_std': 0.03,
            'downsample_scales': [0.85, 0.9, 1.0], # 修复 KeyError
            'num_blur_steps': 1
        },
        2: {
            'stage': 2,
            'name': '中度模糊',
            'epochs': 10,
            'rotation_aug': True,         # 开启旋转
            'blur_kernel_range': (5, 7),
            'blur_sigma_range': (0.8, 1.5),
            'noise_std': 0.06,
            'downsample_scales': [0.75, 0.85, 0.95], # 修复 KeyError
            'num_blur_steps': 1
        },
        3: {
            'stage': 3,
            'name': '复杂退化',
            'epochs': 10,
            'rotation_aug': True,         # 开启旋转
            'blur_kernel_range': (5, 7),
            'blur_sigma_range': (0.8, 1.5),
            'noise_std': 0.06,
            'downsample_scales': [0.75, 0.85, 0.95],
            'num_blur_steps': 1,
            'brightness_range': (0.95, 1.05),
            'contrast_range': (0.95, 1.05),
            'motion_blur_kernel': 3,
            'random_erase_prob': 0.2,
            'random_erase_size': (3, 16)
        },
        'aftrain': {
            'stage': 4,
            'name': '真实Latent后训练',
            'epochs': 30,
            'lr_gen': 1e-5,
            'lr_dis': 5e-7,
            'rotation_aug': True,         # 开启旋转
            'is_paired': True,
            'weights': {
                'l1': 50.0,
                'perceptual': 25.0,
                'ssim': 10.0,
                'orientation': 50.0,
                'adv': 2.0
            }
        }
    }

    @staticmethod
    def get_stage_config(stage_id):
        return Config.stage_configs.get(stage_id)

    @classmethod
    def init_dirs(cls):
        paths = [cls.save_samples_dir, cls.save_models_dir, 
                 cls.aftrain_latent_root, cls.enhanced_output_dir]
        for s in [1, 2, 3, 'aftrain']:
            paths.append(os.path.join(cls.save_models_dir, f"stage_{s}"))
            paths.append(os.path.join(cls.save_samples_dir, f"stage_{s}"))
        for p in paths:
            os.makedirs(p, exist_ok=True)

# 初始化目录
Config.init_dirs()