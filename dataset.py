import os
import numpy as np
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset
from scipy.signal import convolve2d
from config import Config

class ProgressiveDataset(Dataset):
    def __init__(self, stage_id):
        """
        :param stage_id: 可以是 1, 2, 3 (预训练) 或 'aftrain' (后训练)
        """
        # 1. 获取阶段配置
        self.stage_config = Config.get_stage_config(stage_id)
        if self.stage_config is None:
            raise ValueError(f"无效的 stage_id: {stage_id}，请检查 Config.stage_configs")
            
        self.stage_id = stage_id
        self.is_aftrain = (stage_id == 'aftrain')

        # 2. 根据阶段选择路径
        if self.is_aftrain:
            self.input_dir = Config.aftrain_latent_root
            self.target_dir = Config.aftrain_label_root
        else:
            self.input_dir = Config.train_root 
            self.target_dir = Config.test_root

        # 3. 加载文件列表
        self.file_names = sorted([f for f in os.listdir(self.input_dir)
                                 if f.lower().endswith(('png', 'bmp', 'jpg', 'jpeg'))])

        if len(self.file_names) == 0:
            raise RuntimeError(f"路径下无图像文件：{self.input_dir}")

        # 4. 标准 Tensor 转换逻辑 (用于 Target 和已经处理好的 Input)
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    # ---------------- 指纹归一化 (Hong's Algorithm 核心步) ----------------
    def _fingerprint_normalization(self, img_np):
        img_np = img_np.astype(np.float32)
        M = np.mean(img_np)
        VAR = max(np.var(img_np), 1e-6)
        diff = img_np - M
        term = np.sqrt(Config.norm_VAR0 * (diff ** 2) / VAR)
        normalized_img = np.where(img_np > M, Config.norm_M0 + term, Config.norm_M0 - term)
        return np.clip(normalized_img, 0, 255).astype(np.uint8)

    # ---------------- 基础退化算子 ----------------
    def _apply_blur(self, img, sigma_range=(0.5, 1.5), steps=1):
        blurred = img.copy()
        for _ in range(steps):
            sigma = np.random.uniform(*sigma_range)
            blurred = blurred.filter(ImageFilter.GaussianBlur(radius=sigma))
        return blurred

    def _apply_motion_blur(self, img, kernel_size=3):
        img_np = np.array(img, dtype=np.float32)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = 1.0 / kernel_size
        img_np = convolve2d(img_np, kernel, mode='same', boundary='symm')
        return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

    def _apply_brightness_contrast(self, img, brightness_range=(0.9, 1.1), contrast_range=(0.9, 1.1)):
        b = np.random.uniform(*brightness_range)
        c = np.random.uniform(*contrast_range)
        img = ImageEnhance.Brightness(img).enhance(b)
        img = ImageEnhance.Contrast(img).enhance(c)
        return img

    def _apply_random_erase(self, img, erase_prob=0.3, erase_size=(3, 32)):
        if np.random.rand() > erase_prob:
            return img
        img_np = np.array(img)
        h, w = img_np.shape
        eh = np.random.randint(*erase_size)
        ew = np.random.randint(*erase_size)
        ex = np.random.randint(0, max(1, w - ew))
        ey = np.random.randint(0, max(1, h - eh))
        img_np[ey:ey + eh, ex:ex + ew] = 127 
        return Image.fromarray(img_np)

    def _apply_rotation(self, img, label_img):
        if not self.stage_config.get('rotation_aug', False):
            return img, label_img
        angle = random.choice([0, 90, 180, 270])
        return img.rotate(angle), label_img.rotate(angle)

    # ---------------- 动态退化生成逻辑 (针对 Stage 1-3) ----------------
    def _generate_degraded_input(self, input_img):
        w0, h0 = input_img.size
        img = input_img.copy()

        if self.stage_id == 3:
            scale = np.random.choice(self.stage_config['downsample_scales'])
            img = img.resize((int(w0 * scale), int(h0 * scale)), Image.BILINEAR)
            img = img.resize((w0, h0), Image.BILINEAR)
            img = self._apply_blur(img, sigma_range=(0.5, 1.0), steps=1)
            img = self._apply_brightness_contrast(img)
            img = self._apply_motion_blur(img, kernel_size=3)
            img = self._apply_random_erase(img)
        else:
            scale = np.random.choice(self.stage_config['downsample_scales'])
            img = img.resize((int(w0 * scale), int(h0 * scale)), Image.BILINEAR)
            img = img.resize((w0, h0), Image.BILINEAR)
            img = self._apply_blur(img, 
                                   sigma_range=self.stage_config['blur_sigma_range'], 
                                   steps=self.stage_config['num_blur_steps'])

        # 核心：无论预训练还是后训练，都经过 Hong's Normalization
        img_np = self._fingerprint_normalization(np.array(img))
        img_tensor = self.base_transform(Image.fromarray(img_np))

        if self.stage_config.get('noise_std', 0) > 0:
            img_tensor += torch.randn_like(img_tensor) * self.stage_config['noise_std']

        return img_tensor

    # ---------------- 核心获取方法 ----------------
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        
        # 1. 加载原始图像
        raw_input = Image.open(os.path.join(self.input_dir, file_name)).convert("L")
        raw_target = Image.open(os.path.join(self.target_dir, file_name)).convert("L")
        
        # 2. 统一 Resize 到 Config 尺寸 (重要：防止 latent 大小不一导致无法 batch)
        raw_input = raw_input.resize(Config.img_size, Image.Resampling.LANCZOS)
        raw_target = raw_target.resize(Config.img_size, Image.Resampling.LANCZOS)

        # 3. 同步数据增强
        raw_input, raw_target = self._apply_rotation(raw_input, raw_target)

        if self.is_aftrain:
            # 后训练：直接归一化真实 latent 数据，不加人为退化
            norm_input_np = self._fingerprint_normalization(np.array(raw_input))
            input_tensor = self.base_transform(Image.fromarray(norm_input_np))
            target_tensor = self.base_transform(raw_target)
        else:
            # 预训练：动态退化
            input_tensor = self._generate_degraded_input(raw_input)
            target_tensor = self.base_transform(raw_target)

        return input_tensor, target_tensor