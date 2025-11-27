import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
import json
from typing import Dict, List, Tuple, Optional, Callable, Any
from pathlib import Path
import warnings


class LWETrainer:
    """
    LWE模型训练器
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备
        config: 训练配置字典
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        device: str = None,
        config: Dict[str, Any] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置损失函数和优化器
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=1e-3)
        
        # 训练配置
        self.config = config or {}
        self.epochs = self.config.get('epochs', 50)
        self.early_stopping_patience = self.config.get('early_stopping_patience', 10)
        self.save_best_only = self.config.get('save_best_only', True)
        self.metrics = self.config.get('metrics', ['accuracy', 'loss'])
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.early_stopping_counter = 0
        self.train_history = {
            'train_loss': [], 'val_loss': [],
            'train_accuracy': [], 'val_accuracy': [],
            'learning_rates': []
        }
        
        # 日志和保存路径
        self.log_dir = Path(self.config.get('log_dir', 'logs'))
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard写入器
        self.writer = SummaryWriter(self.log_dir)
        
        # 将模型移动到设备
        self.model.to(self.device)
        
        print(f"Training initialized on device: {self.device}")
        print(f"Model has {self._count_parameters():,} trainable parameters")
    
    def _count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        训练一个epoch
        
        Returns:
            (平均训练损失, 训练准确率)
        """
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (A, b, labels) in enumerate(self.train_loader):
            # 移动数据到设备
            A, b, labels = A.to(self.device), b.to(self.device), labels.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(A, b)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（如果配置了）
            if self.config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            # 更新参数
            self.optimizer.step()
            
            # 统计信息
            running_loss += loss.item() * A.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # 进度打印
            if batch_idx % self.config.get('log_interval', 10) == 0:
                print(f'Train Epoch: {self.current_epoch} [{batch_idx * len(A)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self) -> Tuple[float, float, Dict[str, float]]:
        """
        验证一个epoch
        
        Returns:
            (验证损失, 验证准确率, 详细指标字典)
        """
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # 用于计算额外指标的变量
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        with torch.no_grad():
            for A, b, labels in self.val_loader:
                A, b, labels = A.to(self.device), b.to(self.device), labels.to(self.device)
                
                outputs = self.model(A, b)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * A.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # 计算混淆矩阵元素
                true_positives += ((predicted == 1) & (labels == 1)).sum().item()
                false_positives += ((predicted == 1) & (labels == 0)).sum().item()
                true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
                false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
        
        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        
        # 计算额外指标
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # LWE优势（decisional-LWE问题的关键指标）
        advantage = 2 * (epoch_accuracy - 0.5)
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'advantage': advantage,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
        
        return epoch_loss, epoch_accuracy, metrics
    
    def train(self) -> Dict[str, List[float]]:
        """
        完整训练过程
        
        Returns:
            训练历史记录
        """
        print("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_loss, train_accuracy = self.train_epoch()
            
            # 验证
            val_loss, val_accuracy, val_metrics = self.validate_epoch()
            
            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 更新历史记录
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_accuracy'].append(train_accuracy)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_accuracy)
            self.train_history['learning_rates'].append(current_lr)
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            self.writer.add_scalar('Advantage/val', val_metrics['advantage'], epoch)
            
            # 打印进度
            print(f'Epoch {epoch+1}/{self.epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            print(f'  Val Advantage: {val_metrics["advantage"]:.4f}')
            print(f'  Learning Rate: {current_lr:.6f}')
            
            # 学习率调度（如果配置了）
            if hasattr(self, 'scheduler'):
                # 根据调度器类型调用不同的step方法
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau需要传入验证损失
                    self.scheduler.step(val_loss)
                else:
                    # 其他调度器不需要参数
                    self.scheduler.step()
            
            # 早停和模型保存
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_accuracy
                self.early_stopping_counter = 0
                self.save_checkpoint(best=True)
                print(f'  *** New best model saved (loss: {val_loss:.4f}) ***')
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
            
            # 保存常规检查点
            if epoch % self.config.get('checkpoint_interval', 10) == 0:
                self.save_checkpoint(best=False)
        
        # 训练结束
        training_time = time.time() - start_time
        print(f'Training completed in {training_time:.2f} seconds')
        print(f'Best validation loss: {self.best_val_loss:.4f}')
        print(f'Best validation accuracy: {self.best_val_accuracy:.4f}')
        
        # 关闭TensorBoard写入器
        self.writer.close()
        
        return self.train_history
    
    def save_checkpoint(self, best: bool = False) -> None:
        """
        保存模型检查点
        
        Args:
            best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy
        }
        
        if hasattr(self, 'scheduler'):
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if best:
            filename = self.checkpoint_dir / 'best_model.pth'
        else:
            filename = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
        
        torch.save(checkpoint, filename)
        
        # 同时保存训练历史为JSON
        if best:
            history_file = self.checkpoint_dir / 'training_history.json'
            with open(history_file, 'w') as f:
                json.dump(self.train_history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载模型检查点
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_history = checkpoint['train_history']
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}")
    
    def set_scheduler(self, scheduler: optim.lr_scheduler._LRScheduler) -> None:
        """
        设置学习率调度器
        
        Args:
            scheduler: 学习率调度器
        """
        self.scheduler = scheduler
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        在测试集上评估模型
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            评估指标字典
        """
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for A, b, labels in test_loader:
                A, b, labels = A.to(self.device), b.to(self.device), labels.to(self.device)
                
                outputs = self.model(A, b)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * A.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_loss = running_loss / total_samples
        test_accuracy = correct_predictions / total_samples
        
        # 计算LWE优势
        advantage = 2 * (test_accuracy - 0.5)
        
        # 计算其他统计量
        from sklearn.metrics import classification_report, confusion_matrix
        cm = confusion_matrix(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, output_dict=True)
        
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'advantage': advantage,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        print(f"Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Advantage: {advantage:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        
        return metrics
    
    def predict(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        预测单个样本
        
        Args:
            A: 输入矩阵A
            b: 输入向量b
            
        Returns:
            预测结果
        """
        self.model.eval()
        with torch.no_grad():
            A = A.to(self.device).unsqueeze(0)  # 添加批次维度
            b = b.to(self.device).unsqueeze(0)
            output = self.model(A, b)
            _, predicted = torch.max(output, 1)
            return predicted.cpu().item()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        获取训练摘要
        
        Returns:
            训练摘要字典
        """
        return {
            'total_epochs': self.current_epoch + 1,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'final_train_loss': self.train_history['train_loss'][-1] if self.train_history['train_loss'] else None,
            'final_val_loss': self.train_history['val_loss'][-1] if self.train_history['val_loss'] else None,
            'parameters_count': self._count_parameters(),
            'device': str(self.device)
        }


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any] = None
) -> LWETrainer:
    """
    创建训练器的便捷函数
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 训练配置
        
    Returns:
        配置好的训练器
    """
    # 默认配置
    default_config = {
        'epochs': 50,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'early_stopping_patience': 10,
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints',
        'log_interval': 10,
        'checkpoint_interval': 10
    }
    
    if config:
        default_config.update(config)
    
    config = default_config
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 创建训练器
    trainer = LWETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config
    )
    
    # 学习率调度器 - 使用StepLR而不是ReduceLROnPlateau来避免参数问题
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5
    )
    trainer.set_scheduler(scheduler)
    
    return trainer


def create_advanced_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any] = None
) -> LWETrainer:
    """
    创建高级训练器，带有更复杂的配置
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 训练配置
        
    Returns:
        配置好的高级训练器
    """
    # 高级配置
    advanced_config = {
        'epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'grad_clip': 0.5,
        'early_stopping_patience': 15,
        'log_dir': 'advanced_logs',
        'checkpoint_dir': 'advanced_checkpoints',
        'log_interval': 5,
        'checkpoint_interval': 5,
        'metrics': ['accuracy', 'loss', 'precision', 'recall', 'f1_score', 'advantage']
    }
    
    if config:
        advanced_config.update(config)
    
    config = advanced_config
    
    # 使用带warming up的学习率调度
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # 创建训练器
    trainer = LWETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=config
    )
    
    # 复杂的学习率调度 - 使用CosineAnnealingWarmRestarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    trainer.set_scheduler(scheduler)
    
    return trainer


class CrossValidationTrainer:
    """
    交叉验证训练器
    """
    
    def __init__(self, model_class, dataset, k_folds=5, config=None):
        self.model_class = model_class
        self.dataset = dataset
        self.k_folds = k_folds
        self.config = config or {}
        self.fold_results = []
    
    def run_cross_validation(self):
        """运行k折交叉验证"""
        from sklearn.model_selection import KFold
        
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            print(f"\n=== Fold {fold + 1}/{self.k_folds} ===")
            
            # 创建数据加载器
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(
                self.dataset, batch_size=self.config.get('batch_size', 32),
                sampler=train_subsampler
            )
            val_loader = DataLoader(
                self.dataset, batch_size=self.config.get('batch_size', 32),
                sampler=val_subsampler
            )
            
            # 创建新模型实例
            model = self.model_class(**self.config.get('model_args', {}))
            
            # 创建训练器并训练
            trainer = create_trainer(model, train_loader, val_loader, self.config)
            history = trainer.train()
            
            # 记录结果
            fold_result = {
                'fold': fold + 1,
                'history': history,
                'best_val_loss': trainer.best_val_loss,
                'best_val_accuracy': trainer.best_val_accuracy,
                'summary': trainer.get_training_summary()
            }
            self.fold_results.append(fold_result)
        
        return self._summarize_cross_validation()
    
    def _summarize_cross_validation(self):
        """汇总交叉验证结果"""
        val_losses = [result['best_val_loss'] for result in self.fold_results]
        val_accuracies = [result['best_val_accuracy'] for result in self.fold_results]
        
        summary = {
            'mean_val_loss': np.mean(val_losses),
            'std_val_loss': np.std(val_losses),
            'mean_val_accuracy': np.mean(val_accuracies),
            'std_val_accuracy': np.std(val_accuracies),
            'fold_results': self.fold_results
        }
        
        print(f"\n=== Cross Validation Summary ===")
        print(f"Mean Val Loss: {summary['mean_val_loss']:.4f} ± {summary['std_val_loss']:.4f}")
        print(f"Mean Val Accuracy: {summary['mean_val_accuracy']:.4f} ± {summary['std_val_accuracy']:.4f}")
        
        return summary
    