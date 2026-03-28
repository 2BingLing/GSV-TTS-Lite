import torch

def get_cuda_device_info(idx: int):
    """获取 CUDA 设备信息"""
    if not torch.cuda.is_available() or idx >= torch.cuda.device_count():
        return None

    try:
        props = torch.cuda.get_device_properties(idx)
    except Exception:
        return None

    name = props.name
    major, minor = props.major, props.minor
    sm_version = major + minor / 10.0
    mem_gb = props.total_memory / (1024**3)

    # 算力太低（Pascal 架构之前的旧卡），返回 None
    if sm_version < 5.3:
        return None

    device = torch.device(f"cuda:{idx}")

    # 针对 GTX 16 系列 (Turing 架构但不带 Tensor Cores) 或 Pascal (如 GTX 1080)
    # 这些卡虽然支持 fp16 计算，但速度可能不如 fp32，或者容易溢出，所以使用 fp32
    is_16_series = (major == 7 and minor == 5) and ("16" in name)
    if sm_version == 6.1 or is_16_series:
        return device, torch.float32, sm_version, mem_gb

    # 其他较新的卡（Volta, Turing RTX, Ampere, Hopper 等）使用 fp16
    if sm_version > 6.1:
        return device, torch.float16, sm_version, mem_gb

    return None


def get_mps_device_info():
    """获取 Apple Silicon MPS 设备信息"""
    if not torch.backends.mps.is_available():
        return None

    try:
        # MPS 设备
        device = torch.device("mps")
        # Apple Silicon 上 MPS 使用 float32 更稳定
        # 虽然 MPS 支持 float16，但在某些模型上可能有精度问题
        return device, torch.float32, 0.0, 0.0  # sm_version 和 mem_gb 对 MPS 不适用
    except Exception:
        return None


# 检测设备类型和配置
infer_device = None
is_half = False
device_type = "cpu"

# 优先尝试 CUDA
if torch.cuda.is_available():
    GPU_COUNT = torch.cuda.device_count()
    available_devices = []
    for i in range(GPU_COUNT):
        info = get_cuda_device_info(i)
        if info is not None:
            available_devices.append(info)

    if available_devices:
        best_info = max(available_devices, key=lambda x: (x[2], x[3]))
        infer_device = best_info[0]
        is_half = (best_info[1] == torch.float16)
        device_type = "cuda"

# 如果没有 CUDA，尝试 MPS (Apple Silicon)
if infer_device is None:
    mps_info = get_mps_device_info()
    if mps_info is not None:
        infer_device = mps_info[0]
        is_half = False  # MPS 使用 float32
        device_type = "mps"

# 如果没有可用的 GPU，使用 CPU
if infer_device is None:
    infer_device = torch.device("cpu")
    is_half = False  # CPU 使用 float32
    device_type = "cpu"


class Config:
    def __init__(self):
        self.is_half = is_half
        self.dtype = torch.float16 if is_half else torch.float32
        self.device = infer_device
        self.device_type = device_type  # 'cuda', 'mps', 或 'cpu'

        self.use_flash_attn = False

        self.gpt_cache = None

        self.compile_mode = None

        self.cnroberta = None


class GlobalConfig:
    def __init__(self):
        self.models_dir = None

        self.use_jieba_fast = None

        self.chinese_g2p = None
        self.japanese_g2p = None
        self.english_g2p = None

global_config = GlobalConfig()