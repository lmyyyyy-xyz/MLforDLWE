"""
LWE 工具模块

这个模块包含用于LWE Transformer项目的各种工具函数。
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# 导入主要类和方法
from .data_generator import (
    generate_lwe_samples,
    generate_uniform_samples,
    generate_secret,
    generate_lwe_problem,
    generate_structured_lwe_samples,
    calculate_lwe_parameters,
    validate_lwe_parameters,
    add_label_noise,
    generate_parameter_sweep_datasets,
    save_lwe_dataset,
    load_lwe_dataset
)

from .evaluator import (
    LWEEvaluator,
    compare_models,
    print_comparison_results,
    plot_model_comparison,
    calculate_statistical_significance
)

from .logger import (
    setup_logger,
    ExperimentLogger,
    ProgressTracker
)

# 定义公共API
__all__ = [
    # 数据生成
    "generate_lwe_samples",
    "generate_uniform_samples", 
    "generate_secret",
    "generate_lwe_problem",
    "generate_structured_lwe_samples",
    "calculate_lwe_parameters",
    "validate_lwe_parameters",
    "add_label_noise",
    "generate_parameter_sweep_datasets",
    "save_lwe_dataset",
    "load_lwe_dataset",
    
    # 评估
    "LWEEvaluator",
    "compare_models",
    "print_comparison_results", 
    "plot_model_comparison",
    "calculate_statistical_significance",
    
    # 日志记录
    "setup_logger",
    "ExperimentLogger",
    "ProgressTracker"
]

# 工具配置
DEFAULT_GENERATOR_CONFIG = {
    "n": 10,
    "q": 97,
    "error_std": 0.5,
    "secret_distribution": "uniform",
    "error_distribution": "gaussian"
}

DEFAULT_EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1_score", "advantage"],
    "confidence_level": 0.95,
    "n_bootstraps": 1000
}

def get_generator_config(config_type: str = "default") -> dict:
    """
    获取预定义的数据生成器配置
    
    Args:
        config_type: 配置类型
        
    Returns:
        生成器配置字典
    """
    configs = {
        "default": DEFAULT_GENERATOR_CONFIG,
        "easy": {
            "n": 5,
            "q": 97,
            "error_std": 0.1,
            "secret_distribution": "uniform",
            "error_distribution": "gaussian"
        },
        "hard": {
            "n": 20,
            "q": 97,
            "error_std": 1.0,
            "secret_distribution": "uniform", 
            "error_distribution": "gaussian"
        },
        "binary_secret": {
            "n": 10,
            "q": 97,
            "error_std": 0.5,
            "secret_distribution": "binary",
            "error_distribution": "gaussian"
        },
        "structured": {
            "n": 10,
            "q": 97,
            "error_std": 0.5,
            "secret_distribution": "uniform",
            "error_distribution": "gaussian",
            "structure_type": "toeplitz"
        }
    }
    
    return configs.get(config_type, DEFAULT_GENERATOR_CONFIG)

def get_evaluation_config(config_type: str = "default") -> dict:
    """
    获取预定义的评估配置
    
    Args:
        config_type: 配置类型
        
    Returns:
        评估配置字典
    """
    configs = {
        "default": DEFAULT_EVALUATION_CONFIG,
        "basic": {
            "metrics": ["accuracy", "advantage"],
            "confidence_level": 0.95,
            "n_bootstraps": 500
        },
        "comprehensive": {
            "metrics": ["accuracy", "precision", "recall", "f1_score", "advantage", "auc_roc"],
            "confidence_level": 0.99,
            "n_bootstraps": 2000
        }
    }
    
    return configs.get(config_type, DEFAULT_EVALUATION_CONFIG)

def create_standard_evaluator(model, device=None):
    """
    创建标准评估器
    
    Args:
        model: 要评估的模型
        device: 计算设备
        
    Returns:
        配置好的评估器
    """
    return LWEEvaluator(model, device)

def setup_experiment_logging(experiment_name, config=None, log_dir="experiments"):
    """
    设置实验日志记录
    
    Args:
        experiment_name: 实验名称
        config: 实验配置
        log_dir: 日志目录
        
    Returns:
        实验日志记录器
    """
    return ExperimentLogger(experiment_name, log_dir, config)

# 工具函数
def set_random_seed(seed: int = 42) -> None:
    """
    设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子
    """
    import random
    import torch
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 为了确定性，但可能会影响性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")

def print_system_info() -> None:
    """打印系统信息"""
    import torch
    import platform
    import psutil
    
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print(f"RAM: {psutil.virtual_memory().total // (1024**3)} GB")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")
    print("="*50)

def estimate_training_time(model, data_loader, epochs=10, device=None):
    """
    估计训练时间
    
    Args:
        model: 模型
        data_loader: 数据加载器
        epochs: 周期数
        device: 计算设备
        
    Returns:
        估计的训练时间（秒）
    """
    import time
    import torch
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)
    model.train()
    
    # 预热
    for i, (A, b, labels) in enumerate(data_loader):
        if i >= 2:  # 2个批次预热
            break
        A, b, labels = A.to(device), b.to(device), labels.to(device)
        outputs = model(A, b)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
    
    # 测量时间
    start_time = time.time()
    for i, (A, b, labels) in enumerate(data_loader):
        if i >= 10:  # 测量10个批次
            break
        A, b, labels = A.to(device), b.to(device), labels.to(device)
        outputs = model(A, b)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
    
    end_time = time.time()
    avg_batch_time = (end_time - start_time) / 10
    
    # 估计总时间
    total_batches = len(data_loader) * epochs
    estimated_time = avg_batch_time * total_batches
    
    print(f"Estimated training time: {estimated_time/60:.2f} minutes "
          f"for {epochs} epochs ({avg_batch_time:.3f} seconds per batch)")
    
    return estimated_time

# 包初始化信息
print(f"LWE Utils Module v{__version__} loaded successfully.")