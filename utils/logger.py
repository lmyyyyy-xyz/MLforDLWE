import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time
from datetime import datetime
import torch
from typing import Optional, Dict, Any, List


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 记录器名称
        log_file: 日志文件路径
        level: 日志级别
        format_string: 日志格式字符串
        
    Returns:
        配置好的日志记录器
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除已有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter(format_string)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果提供了日志文件）
    if log_file:
        # 创建日志目录
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 防止日志传递给根记录器
    logger.propagate = False
    
    return logger


class ExperimentLogger:
    """
    实验日志记录器，用于记录训练实验的元数据和结果
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "experiments",
        config: Optional[Dict[str, Any]] = None
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"experiment_{timestamp}.log"
        
        # 设置记录器
        self.logger = setup_logger(
            f"experiment_{experiment_name}",
            str(self.log_file),
            level=logging.INFO
        )
        
        # 保存实验配置
        self.config = config or {}
        self._save_config()
        
        # 实验指标
        self.metrics = {}
        self.start_time = time.time()
        
        self.logger.info(f"Experiment '{experiment_name}' started")
        self.logger.info(f"Log directory: {self.log_dir}")
    
    def _save_config(self) -> None:
        """保存实验配置到JSON文件"""
        config_file = self.log_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        记录实验指标
        
        Args:
            metrics: 指标字典
            step: 步骤编号（如epoch数）
        """
        timestamp = time.time() - self.start_time
        
        if step is not None:
            metric_record = {
                'step': step,
                'timestamp': timestamp,
                'metrics': metrics
            }
            if step not in self.metrics:
                self.metrics[step] = []
            self.metrics[step].append(metric_record)
        else:
            if 'general' not in self.metrics:
                self.metrics['general'] = []
            self.metrics['general'].append({
                'timestamp': timestamp,
                'metrics': metrics
            })
        
        # 记录到日志
        if step is not None:
            self.logger.info(f"Step {step}: {metrics}")
        else:
            self.logger.info(f"Metrics: {metrics}")
    
    def log_message(self, message: str, level: str = "INFO") -> None:
        """
        记录一般消息
        
        Args:
            message: 消息内容
            level: 日志级别
        """
        if level.upper() == "INFO":
            self.logger.info(message)
        elif level.upper() == "WARNING":
            self.logger.warning(message)
        elif level.upper() == "ERROR":
            self.logger.error(message)
        elif level.upper() == "DEBUG":
            self.logger.debug(message)
    
    def save_metrics(self, filename: Optional[str] = None) -> None:
        """
        保存指标到JSON文件
        
        Args:
            filename: 文件名
        """
        if filename is None:
            filename = f"metrics_{int(time.time())}.json"
        
        metrics_file = self.log_dir / filename
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to {metrics_file}")
    
    def log_model_info(self, model: torch.nn.Module) -> None:
        """
        记录模型信息
        
        Args:
            model: PyTorch模型
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_architecture': str(model)
        }
        
        self.logger.info(f"Model parameters: {trainable_params:,} trainable, {total_params:,} total")
        
        # 保存模型信息到文件
        model_info_file = self.log_dir / "model_info.json"
        with open(model_info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
    
    def finish_experiment(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """
        结束实验并保存总结
        
        Args:
            summary: 实验总结
        """
        end_time = time.time()
        duration = end_time - self.start_time
        
        experiment_summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time,
            'end_time': end_time,
            'duration_seconds': duration,
            'config': self.config,
            'final_metrics': self.metrics
        }
        
        if summary:
            experiment_summary.update(summary)
        
        # 保存实验总结
        summary_file = self.log_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(experiment_summary, f, indent=2)
        
        self.logger.info(f"Experiment completed in {duration:.2f} seconds")
        self.logger.info(f"Summary saved to {summary_file}")
        
        # 关闭文件处理器
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()


class ProgressTracker:
    """
    训练进度跟踪器
    """
    
    def __init__(self, total_epochs: int, metrics: List[str] = None):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.metrics = metrics or ['loss', 'accuracy', 'advantage']
        self.history = {metric: [] for metric in self.metrics}
        self.best_metrics = {}
    
    def update(self, metrics: Dict[str, float]) -> None:
        """
        更新进度
        
        Args:
            metrics: 当前epoch的指标
        """
        self.current_epoch += 1
        
        for metric in self.metrics:
            if metric in metrics:
                self.history[metric].append(metrics[metric])
                
                # 更新最佳指标
                if metric not in self.best_metrics:
                    self.best_metrics[metric] = {
                        'value': metrics[metric],
                        'epoch': self.current_epoch
                    }
                else:
                    # 对于损失，我们希望最小值；对于其他指标，我们希望最大值
                    if metric == 'loss':
                        if metrics[metric] < self.best_metrics[metric]['value']:
                            self.best_metrics[metric] = {
                                'value': metrics[metric],
                                'epoch': self.current_epoch
                            }
                    else:
                        if metrics[metric] > self.best_metrics[metric]['value']:
                            self.best_metrics[metric] = {
                                'value': metrics[metric],
                                'epoch': self.current_epoch
                            }
    
    def get_progress_string(self) -> str:
        """获取进度字符串"""
        progress = (self.current_epoch / self.total_epochs) * 100
        return f"Epoch {self.current_epoch}/{self.total_epochs} ({progress:.1f}%)"
    
    def get_best_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取最佳指标"""
        return self.best_metrics
    
    def plot_progress(self, save_path: Optional[str] = None) -> None:
        """绘制训练进度图"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, len(self.metrics), figsize=(5*len(self.metrics), 4))
        if len(self.metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(self.metrics):
            if self.history[metric]:
                axes[i].plot(range(1, len(self.history[metric]) + 1), self.history[metric])
                axes[i].set_title(metric.upper())
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric.upper())
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()