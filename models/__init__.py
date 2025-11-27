"""
LWE Transformer Package

基于Transformer的decisional-LWE问题求解模型
包含自适应停止、门控机制等高级特性
"""

from .transformer import (
    # 常量
    N_MAX_POSITIONS,
    
    # 函数
    Embedding,
    create_sinusoidal_embeddings,
    gelu,
    get_masks,
    create_standard_lwe_transformer,
    create_advanced_lwe_transformer,
    create_sparse_lwe_transformer,
    
    # 核心类
    MultiHeadAttention,
    TransformerFFN,
    Gate,
    TransformerLayer,
    AdaptiveHalt,
    PositionalEncoding,
    LWETransformer,
    AdvancedLWETransformer,
)

__version__ = "1.0.0"
__author__ = "LWE Transformer Team"
__email__ = "lwe-transformer@example.com"

# 定义公开的API
__all__ = [
    # 常量
    "N_MAX_POSITIONS",
    
    # 工具函数
    "Embedding",
    "create_sinusoidal_embeddings", 
    "gelu",
    "get_masks",
    
    # 模型创建函数
    "create_standard_lwe_transformer",
    "create_advanced_lwe_transformer",
    "create_sparse_lwe_transformer",
    
    # 核心组件
    "MultiHeadAttention",
    "TransformerFFN", 
    "Gate",
    "TransformerLayer",
    "AdaptiveHalt",
    "PositionalEncoding",
    
    # 主要模型
    "LWETransformer",
    "AdvancedLWETransformer",
]

# 包级别的文档字符串
__doc__ = """
LWE Transformer Package

这个包提供了基于Transformer架构的decisional-LWE问题求解模型。

主要特性：
- 多头注意力机制 (MultiHeadAttention)
- 自适应停止机制 (AdaptiveHalt)
- 门控机制 (Gate)
- 正弦位置编码
- 多种预配置模型

使用示例：
    from lwe_transformer import create_standard_lwe_transformer
    
    model = create_standard_lwe_transformer(
        vocab_size=1000,
        n=100,
        n_classes=2
    )

可用模型：
- LWETransformer: 基础Transformer模型
- AdvancedLWETransformer: 包含高级特性的增强版本

工具函数：
- create_standard_lwe_transformer: 创建标准配置模型
- create_advanced_lwe_transformer: 创建高级配置模型  
- create_sparse_lwe_transformer: 创建针对稀疏LWE的模型
"""
# 包级别的配置
DEFAULT_MODEL_CONFIG = {
    "d_model": 128,
    "nhead": 8,
    "num_layers": 6,
    "dim_feedforward": 256,
    "dropout": 0.1
}

def get_model_config(model_type: str = "standard") -> dict:
    """
    获取预定义的模型配置
    
    Args:
        model_type: 模型类型 ("standard", "large")
        
    Returns:
        模型配置字典
    """
    configs = {
        "standard": {
            "d_model": 128,
            "nhead": 8,
            "num_layers": 6,
            "dim_feedforward": 256,
            "dropout": 0.1
        },
        "large": {
            "d_model": 256,
            "nhead": 8,
            "num_layers": 8,
            "dim_feedforward": 512,
            "dropout": 0.1
        },
        "small": {
            "d_model": 64,
            "nhead": 4,
            "num_layers": 4,
            "dim_feedforward": 128,
            "dropout": 0.1
        }
    }
    
    return configs.get(model_type, configs["standard"])

def create_model_from_config(
    vocab_size: int,
    max_seq_len: int,
    n_classes: int = 2,
    model_type: str = "standard"
) -> LWETransformer:
    """
    根据配置创建模型
    
    Args:
        vocab_size: 词汇表大小
        max_seq_len: 最大序列长度
        n_classes: 分类数
        model_type: 模型类型
        
    Returns:
        创建的模型实例
    """
    config = get_model_config(model_type)
    
    return LWETransformer(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        max_seq_len=max_seq_len,
        n_classes=n_classes,
        dropout=config["dropout"]
    )

# 包初始化信息
print(f"LWE Transformer Models v{__version__} loaded successfully.")

# 包初始化信息
print(f"Loaded LWE Transformer v{__version__}")
print("Available models: LWETransformer, AdvancedLWETransformer")
print("Use create_*_lwe_transformer() for easy model creation.")