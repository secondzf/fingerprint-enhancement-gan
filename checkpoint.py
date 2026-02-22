import torch
from config import Config  # 引入你刚才定义的配置类

def load_pretrained(model, path, device=None):
    # 如果没传 device，自动走 Config 里的设置
    if device is None:
        device = Config.device
        
    if not os.path.exists(path):
        print(f"⚠️ 警告: 未找到预训练权重文件 {path}，将从随机初始化开始训练。")
        return model

    state = torch.load(path, map_location=device)
    
    # 兼容多种保存格式 (EMA 或 普通 state_dict)
    if 'params_ema' in state:
        state_dict = state['params_ema']
    elif 'state_dict' in state:
        state_dict = state['state_dict']
    else:
        state_dict = state
        
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ 成功加载权重: {path}")
    return model