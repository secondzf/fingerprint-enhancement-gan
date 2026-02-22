import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from config import Config  # 仅引入配置类

# -------------------------- 参数 (保持不变) --------------------------
MEDIAN_KERNEL = 3
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

BLOCK_SIZE = 16
BLOCK_OVERLAP = 0.5
GAUSSIAN_SMOOTH_SIGMA = 1.0

SIGMA_MIN = 1.0
SIGMA_MAX = 3.0
LAMBDA_LIST = [5, 7, 9]
GAMMA = 0.5
PSI = 0

INPUT_SHAPE = (256, 256)
TEXTURE_WEIGHT = 0.3  # Gabor 纹理增强权重

DEVICE = "cuda"  # 如果需要可改成 "cuda"

# -------------------------- 1. 预处理 (逻辑完全不动) --------------------------
def preprocess_image(image_path):
    # 修改点：增加对 16bit tif 的兼容性读取
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图像：{image_path}")
    
    # 逻辑不动：处理颜色通道和位深，确保后续逻辑一致
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.medianBlur(img, MEDIAN_KERNEL)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    img = clahe.apply(img)
    return img

# -------------------------- 2. 局部质量评估 (逻辑完全不动) --------------------------
def calculate_local_quality(img):
    h, w = img.shape
    step = int(BLOCK_SIZE*(1-BLOCK_OVERLAP))
    quality_map = np.zeros_like(img, dtype=np.float32)
    for i in range(0, h-BLOCK_SIZE+1, step):
        for j in range(0, w-BLOCK_SIZE+1, step):
            block = img[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            dx = cv2.Sobel(block, cv2.CV_64F,1,0,ksize=3)
            dy = cv2.Sobel(block, cv2.CV_64F,0,1,ksize=3)
            cov_xx = np.mean(dx**2)
            cov_yy = np.mean(dy**2)
            cov_xy = np.mean(dx*dy)
            lambda1 = (cov_xx+cov_yy+np.sqrt((cov_xx-cov_yy)**2+4*cov_xy**2))/2
            lambda2 = (cov_xx+cov_yy-np.sqrt((cov_xx-cov_yy)**2+4*cov_xy**2))/2
            coherence = (lambda1+lambda2)/(lambda1-lambda2+1e-6)
            vertical_profiles = block.mean(axis=1)
            peaks = np.where(np.r_[True, vertical_profiles[1:] > vertical_profiles[:-1]] & 
                             np.r_[vertical_profiles[:-1] > vertical_profiles[1:], True])[0]
            freq_stability = 1.0 if len(peaks)>=2 and (1/25 <= 1/np.mean(np.diff(peaks)) <= 1/3) else 0.0
            var = np.var(block)/255
            quality_map[i:i+BLOCK_SIZE,j:j+BLOCK_SIZE] = 0.4*coherence+0.3*freq_stability+0.3*var
    quality_map = gaussian_filter(quality_map, GAUSSIAN_SMOOTH_SIGMA)
    return cv2.normalize(quality_map, None, 0.0, 1.0, cv2.NORM_MINMAX)

# -------------------------- 3. Gabor滤波增强纹理 (逻辑完全不动) --------------------------
def create_gabor_kernel(theta, sigma, lambd, gamma=GAMMA, psi=PSI):
    sigma_x = sigma
    sigma_y = sigma/gamma
    theta = np.deg2rad(theta)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    ksize = int(6*sigma+1) | 1
    half = ksize//2
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    for x in range(-half, half+1):
        for y in range(-half, half+1):
            x_rot = x*cos_t + y*sin_t
            y_rot = -x*sin_t + y*cos_t
            g = np.exp(-(x_rot**2/(2*sigma_x**2)+y_rot**2/(2*sigma_y**2))) * np.cos(2*np.pi*x_rot/lambd + psi)
            kernel[x+half, y+half] = g
    kernel -= np.mean(kernel)
    kernel /= np.linalg.norm(kernel)+1e-6
    return kernel

def gabor_texture_enhance(img, quality_map):
    h, w = img.shape
    enhanced = np.zeros_like(img, dtype=np.float32)
    step = int(BLOCK_SIZE*(1-BLOCK_OVERLAP))
    Qmax, Qmin = np.max(quality_map), np.min(quality_map)
    for i in range(0, h-BLOCK_SIZE+1, step):
        for j in range(0, w-BLOCK_SIZE+1, step):
            block = img[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE].astype(np.float32)
            q_mean = np.mean(quality_map[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE])
            dx = cv2.Sobel(block, cv2.CV_64F,1,0,ksize=3)
            dy = cv2.Sobel(block, cv2.CV_64F,0,1,ksize=3)
            theta = np.rad2deg(0.5*np.arctan2(2*np.mean(dx*dy), np.mean(dx**2-dy**2)))
            if theta < 0:
                theta += 180
            sigma = SIGMA_MIN + (SIGMA_MAX-SIGMA_MIN)*(Qmax-q_mean)/(Qmax-Qmin+1e-6)
            responses=[]
            for lambd in LAMBDA_LIST:
                k = create_gabor_kernel(theta, sigma, lambd)
                responses.append(cv2.filter2D(block, -1, k))
            texture = np.mean(responses, axis=0)
            enhanced[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = block + TEXTURE_WEIGHT * texture
    return cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# -------------------------- 主流程 (仅适配路径和后缀) --------------------------
def preprocess_and_save(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # 修改点：增加 .tiff 格式支持
    valid_extensions = (".png", ".jpg", ".tif", ".tiff", ".jpeg")
    
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(valid_extensions):
            continue
        input_path = os.path.join(input_dir, fname)
        
        # 依次调用原逻辑函数
        img = preprocess_image(input_path)
        q_map = calculate_local_quality(img)
        img_enhanced = gabor_texture_enhance(img, q_map)
        
        # 修改点：剥离原后缀，统一存为 .png
        base_name = os.path.splitext(fname)[0]
        save_path = os.path.join(output_dir, f"{base_name}.png")
        
        cv2.imwrite(save_path, img_enhanced)
        print(f"保存增强图像: {save_path}")

# -------------------------- 执行 --------------------------
if __name__ == "__main__":
    # 使用你 Config 类中定义的路径
    input_dir = Config.aftrain_raw_latent_root
    output_dir = Config.aftrain_latent_root
    preprocess_and_save(input_dir, output_dir)