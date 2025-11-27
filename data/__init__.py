import torch.nn as nn
import numpy as np
"""
LWE 数据模块

这个模块包含用于decisional-LWE问题的数据集类和数据加载工具。
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# 导入主要类和方法
from .dataset import (
    LWEDataset,
    LWEIterativeDataset,
    LWEBatchSampler,
    create_data_loaders,
    random_shift,
    add_noise,
    create_composite_transform
)

# 定义公共API
__all__ = [
    "LWEDataset",
    "LWEIterativeDataset", 
    "LWEBatchSampler",
    "create_data_loaders",
    "random_shift",
    "add_noise",
    "create_composite_transform"
]

# 数据配置常量
DEFAULT_DATA_CONFIG = {
    "batch_size": 32,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "balanced_sampling": True,
    "normalize": False
}

# 预定义的数据增强管道
DATA_AUGMENTATION_PIPELINES = {
    "basic": create_composite_transform(
        lambda A, b: (A, b)  # 无操作
    ),
    "noise": create_composite_transform(
        add_noise
    ),
    "shift": create_composite_transform(
        random_shift
    ),
    "full": create_composite_transform(
        random_shift,
        add_noise
    )
}

def get_data_config(config_name: str = "default") -> dict:
    """
    获取预定义的数据配置
    
    Args:
        config_name: 配置名称
        
    Returns:
        数据配置字典
    """
    configs = {
        "default": DEFAULT_DATA_CONFIG,
        "large_batch": {
            "batch_size": 64,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "balanced_sampling": True,
            "normalize": False
        },
        "imbalanced": {
            "batch_size": 32,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "balanced_sampling": False,
            "normalize": True
        }
    }
    
    return configs.get(config_name, DEFAULT_DATA_CONFIG)

def create_dataset_from_generator(
    n: int,
    q: int,
    secret: np.ndarray,
    num_samples: int,
    error_std: float = 0.5,
    use_iterative: bool = False,
    **kwargs
):
    """
    从生成器创建数据集
    
    Args:
        n: LWE维度
        q: 模数
        secret: 秘密向量
        num_samples: 样本数量
        error_std: 误差标准差
        use_iterative: 是否使用迭代式数据集
        **kwargs: 其他数据集参数
        
    Returns:
        创建的数据集
    """
    if use_iterative:
        return LWEIterativeDataset(
            n=n,
            q=q,
            secret=secret,
            num_samples=num_samples,
            error_std=error_std,
            **kwargs
        )
    else:
        # 需要先导入数据生成函数
        from utils.data_generator import generate_lwe_samples, generate_uniform_samples
        
        lwe_samples = generate_lwe_samples(n, q, secret, num_samples // 2, error_std)
        uniform_samples = generate_uniform_samples(n, q, num_samples // 2)
        
        return LWEDataset(lwe_samples, uniform_samples, **kwargs)

def print_dataset_info(dataset: LWEDataset, name: str = "Dataset") -> None:
    """
    打印数据集信息
    
    Args:
        dataset: 数据集实例
        name: 数据集名称
    """
    stats = dataset.get_statistics()
    
    print(f"\n{name} Information:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  LWE samples: {stats['lwe_samples']}")
    print(f"  Uniform samples: {stats['uniform_samples']}")
    print(f"  Class ratio (LWE/Total): {stats['class_ratio']:.4f}")
    
    if 'A_mean' in stats:
        print(f"  Normalization: Enabled")
        print(f"  A mean: {stats['A_mean'][:3]}...")  # 只显示前3个维度
        print(f"  A std: {stats['A_std'][:3]}...")
        print(f"  b mean: {stats['b_mean']:.4f}")
        print(f"  b std: {stats['b_std']:.4f}")

# 包初始化信息
print(f"LWE Data Module v{__version__} loaded successfully.")