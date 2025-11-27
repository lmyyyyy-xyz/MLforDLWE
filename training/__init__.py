"""
LWE 训练模块

这个模块包含用于训练LWE Transformer模型的训练器和相关工具。
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# 导入主要类和方法
from .trainer import (
    LWETrainer,
    CrossValidationTrainer,
    create_trainer,
    create_advanced_trainer
)

# 定义公共API
__all__ = [
    "LWETrainer",
    "CrossValidationTrainer", 
    "create_trainer",
    "create_advanced_trainer"
]

# 训练配置常量
DEFAULT_TRAINING_CONFIG = {
    "epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "grad_clip": 1.0,
    "early_stopping_patience": 10,
    "log_interval": 10,
    "checkpoint_interval": 10
}

ADVANCED_TRAINING_CONFIG = {
    "epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 32,
    "grad_clip": 0.5,
    "early_stopping_patience": 15,
    "log_interval": 5,
    "checkpoint_interval": 5
}

def get_training_config(config_type: str = "default") -> dict:
    """
    获取预定义的训练配置
    
    Args:
        config_type: 配置类型 ("default", "advanced", "fast")
        
    Returns:
        训练配置字典
    """
    configs = {
        "default": DEFAULT_TRAINING_CONFIG,
        "advanced": ADVANCED_TRAINING_CONFIG,
        "fast": {
            "epochs": 20,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "grad_clip": 1.0,
            "early_stopping_patience": 5,
            "log_interval": 20,
            "checkpoint_interval": 5
        },
        "large_batch": {
            "epochs": 50,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "batch_size": 128,
            "grad_clip": 0.5,
            "early_stopping_patience": 10,
            "log_interval": 10,
            "checkpoint_interval": 10
        }
    }
    
    return configs.get(config_type, DEFAULT_TRAINING_CONFIG)

def setup_training(
    model,
    train_loader,
    val_loader,
    config_type: str = "default",
    custom_config: dict = None
) -> LWETrainer:
    """
    快速设置训练
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config_type: 配置类型
        custom_config: 自定义配置
        
    Returns:
        配置好的训练器
    """
    config = get_training_config(config_type)
    
    if custom_config:
        config.update(custom_config)
    
    if config_type == "advanced":
        trainer = create_advanced_trainer(model, train_loader, val_loader, config)
    else:
        trainer = create_trainer(model, train_loader, val_loader, config)
    
    return trainer

def print_training_summary(trainer: LWETrainer) -> None:
    """
    打印训练摘要
    
    Args:
        trainer: 训练器实例
    """
    summary = trainer.get_training_summary()
    
    print("\n=== Training Summary ===")
    print(f"Total Epochs: {summary['total_epochs']}")
    print(f"Best Validation Loss: {summary['best_val_loss']:.4f}")
    print(f"Best Validation Accuracy: {summary['best_val_accuracy']:.4f}")
    print(f"Final Training Loss: {summary['final_train_loss']:.4f}")
    print(f"Final Validation Loss: {summary['final_val_loss']:.4f}")
    print(f"Model Parameters: {summary['parameters_count']:,}")
    print(f"Training Device: {summary['device']}")

# 训练工具函数
def find_optimal_learning_rate(
    model,
    train_loader,
    val_loader,
    lr_range=(1e-5, 1e-1),
    num_iter=100
):
    """
    寻找最优学习率（学习率范围测试）
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        lr_range: 学习率范围
        num_iter: 迭代次数
        
    Returns:
        推荐的学习率
    """
    # 这是一个简化的实现，实际可以使用更复杂的方法
    # 如PyTorch的lr_finder或自定义实现
    
    print("Learning rate range test is a placeholder.")
    print("In practice, you might want to use torch-lr-finder or similar packages.")
    
    # 返回一个合理的学习率
    return 1e-3

def compare_optimizers(
    model_class,
    train_loader,
    val_loader,
    optimizers=['adam', 'sgd', 'adamw']
):
    """
    比较不同优化器的性能
    
    Args:
        model_class: 模型类
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizers: 要比较的优化器列表
        
    Returns:
        比较结果字典
    """
    results = {}
    
    for opt_name in optimizers:
        print(f"\n=== Testing {opt_name.upper()} ===")
        
        # 创建新模型实例
        model = model_class()
        
        # 设置优化器
        if opt_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
        elif opt_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        elif opt_name == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        else:
            continue
        
        # 创建训练器并训练少量周期进行测试
        trainer = LWETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config={'epochs': 10, 'log_interval': 20}
        )
        
        history = trainer.train()
        results[opt_name] = {
            'final_val_loss': history['val_loss'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
            'best_val_loss': trainer.best_val_loss
        }
    
    return results

# 包初始化信息
print(f"LWE Training Module v{__version__} loaded successfully.")