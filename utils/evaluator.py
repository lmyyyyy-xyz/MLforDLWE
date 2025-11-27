import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class LWEEvaluator:
    """
    LWE模型评估器
    
    Args:
        model: 要评估的模型
        device: 计算设备
    """
    
    def __init__(self, model: torch.nn.Module, device: str = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, float]:
        """
        在数据加载器上评估模型
        
        Args:
            data_loader: 数据加载器
            return_predictions: 是否返回预测结果
            
        Returns:
            评估指标字典
        """
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for A, b, labels in data_loader:
                A, b, labels = A.to(self.device), b.to(self.device), labels.to(self.device)
                
                outputs = self.model(A, b)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # 计算指标
        metrics = self._compute_metrics(all_predictions, all_labels, all_probabilities)
        
        if return_predictions:
            return metrics, all_predictions, all_probabilities
        else:
            return metrics
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            predictions: 预测标签
            labels: 真实标签
            probabilities: 预测概率
            
        Returns:
            指标字典
        """
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        # LWE优势（decisional-LWE问题的关键指标）
        advantage = 2 * (accuracy - 0.5)
        
        # AUC-ROC（如果问题是二分类且概率可用）
        try:
            auc_roc = roc_auc_score(labels, probabilities[:, 1])
        except:
            auc_roc = 0.0
        
        # 混淆矩阵元素
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # 计算误报率和漏报率
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'advantage': advantage,
            'auc_roc': auc_roc,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'confusion_matrix': cm.tolist()
        }
    
    def evaluate_parameter_sensitivity(
        self,
        datasets: Dict[str, Any],
        batch_size: int = 32
    ) -> Dict[str, Dict[str, float]]:
        """
        评估模型对不同LWE参数的敏感性
        
        Args:
            datasets: 参数扫描数据集
            batch_size: 批次大小
            
        Returns:
            敏感性分析结果
        """
        from torch.utils.data import DataLoader
        from data.dataset import LWEDataset
        
        results = {}
        
        for key, dataset_info in datasets.items():
            print(f"Evaluating on {key}...")
            
            # 创建数据集和数据加载器
            dataset = LWEDataset(
                dataset_info['lwe_samples'],
                dataset_info['uniform_samples']
            )
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            # 评估模型
            metrics = self.evaluate(data_loader)
            results[key] = {
                'metrics': metrics,
                'parameters': dataset_info['parameters']
            }
        
        return results
    
    def plot_confusion_matrix(
        self,
        data_loader: torch.utils.data.DataLoader,
        save_path: Optional[str] = None,
        title: str = "Confusion Matrix"
    ) -> None:
        """
        绘制混淆矩阵
        
        Args:
            data_loader: 数据加载器
            save_path: 保存路径
            title: 图表标题
        """
        metrics, predictions, _ = self.evaluate(data_loader, return_predictions=True)
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_parameter_sensitivity(
        self,
        sensitivity_results: Dict[str, Dict[str, Any]],
        parameter: str,  # 'n', 'q', or 'error_std'
        metric: str = 'advantage',
        save_path: Optional[str] = None
    ) -> None:
        """
        绘制参数敏感性分析图
        
        Args:
            sensitivity_results: 敏感性分析结果
            parameter: 要分析的参数
            metric: 要绘制的指标
            save_path: 保存路径
        """
        # 提取数据和参数值
        param_values = []
        metric_values = []
        
        for key, result in sensitivity_results.items():
            param_dict = result['parameters']
            if parameter in param_dict:
                param_values.append(param_dict[parameter])
                metric_values.append(result['metrics'][metric])
        
        # 排序
        sorted_indices = np.argsort(param_values)
        param_values = np.array(param_values)[sorted_indices]
        metric_values = np.array(metric_values)[sorted_indices]
        
        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, metric_values, 'o-', linewidth=2, markersize=8)
        plt.xlabel(parameter.upper())
        plt.ylabel(metric.upper())
        plt.title(f'Model {metric.upper()} vs {parameter.upper()}')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sensitivity plot saved to {save_path}")
        
        plt.show()
    
    def compute_confidence_intervals(
        self,
        data_loader: torch.utils.data.DataLoader,
        n_bootstraps: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        使用自助法计算指标的置信区间
        
        Args:
            data_loader: 数据加载器
            n_bootstraps: 自助法采样次数
            confidence_level: 置信水平
            
        Returns:
            置信区间字典
        """
        # 收集所有预测和标签
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for A, b, labels in data_loader:
                A, b, labels = A.to(self.device), b.to(self.device), labels.to(self.device)
                outputs = self.model(A, b)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # 自助法采样
        n_samples = len(all_labels)
        bootstrap_metrics = []
        
        for _ in range(n_bootstraps):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_pred = all_predictions[indices]
            bootstrap_true = all_labels[indices]
            
            # 计算指标
            accuracy = accuracy_score(bootstrap_true, bootstrap_pred)
            precision = precision_score(bootstrap_true, bootstrap_pred, zero_division=0)
            recall = recall_score(bootstrap_true, bootstrap_pred, zero_division=0)
            f1 = f1_score(bootstrap_true, bootstrap_pred, zero_division=0)
            advantage = 2 * (accuracy - 0.5)
            
            bootstrap_metrics.append({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'advantage': advantage
            })
        
        # 计算置信区间
        confidence_intervals = {}
        alpha = (1 - confidence_level) / 2
        percentiles = [100 * alpha, 100 * (1 - alpha)]
        
        for metric_name in bootstrap_metrics[0].keys():
            values = [m[metric_name] for m in bootstrap_metrics]
            mean_value = np.mean(values)
            ci_low, ci_high = np.percentile(values, percentiles)
            
            confidence_intervals[metric_name] = (mean_value, ci_low, ci_high)
        
        return confidence_intervals


def compare_models(
    models: Dict[str, torch.nn.Module],
    data_loader: torch.utils.data.DataLoader,
    device: str = None
) -> Dict[str, Dict[str, float]]:
    """
    比较多个模型的性能
    
    Args:
        models: 模型字典 {模型名: 模型}
        data_loader: 数据加载器
        device: 计算设备
        
    Returns:
        比较结果字典
    """
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        evaluator = LWEEvaluator(model, device)
        metrics = evaluator.evaluate(data_loader)
        results[name] = metrics
    
    return results


def print_comparison_results(comparison_results: Dict[str, Dict[str, float]]) -> None:
    """
    打印模型比较结果
    
    Args:
        comparison_results: 比较结果
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    # 获取所有指标名称
    metric_names = list(comparison_results[list(comparison_results.keys())[0]].keys())
    metric_names = [m for m in metric_names if not m.startswith('confusion')]
    
    # 打印表头
    header = "Model".ljust(20)
    for metric in metric_names:
        header += f"{metric.upper().ljust(12)}"
    print(header)
    print("-"*80)
    
    # 打印每个模型的结果
    for model_name, metrics in comparison_results.items():
        row = model_name.ljust(20)
        for metric in metric_names:
            value = metrics[metric]
            if isinstance(value, float):
                row += f"{value:.4f}".ljust(12)
            else:
                row += f"{value}".ljust(12)
        print(row)
    
    print("="*80)


def plot_model_comparison(
    comparison_results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    绘制模型比较图
    
    Args:
        comparison_results: 比较结果
        metrics: 要绘制的指标列表
        save_path: 保存路径
    """
    if metrics is None:
        metrics = ['accuracy', 'advantage', 'f1_score', 'auc_roc']
    
    model_names = list(comparison_results.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        values = [comparison_results[model][metric] for model in model_names]
        
        axes[i].bar(model_names, values, color=plt.cm.Set3(np.arange(len(model_names))))
        axes[i].set_title(metric.upper())
        axes[i].set_ylabel(metric.upper())
        axes[i].tick_params(axis='x', rotation=45)
        
        # 在柱状图上添加数值
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


def calculate_statistical_significance(
    model1_metrics: List[float],
    model2_metrics: List[float],
    metric_name: str = 'accuracy'
) -> Dict[str, Any]:
    """
    计算两个模型性能差异的统计显著性
    
    Args:
        model1_metrics: 模型1的指标列表(来自交叉验证或自助法)
        model2_metrics: 模型2的指标列表
        metric_name: 指标名称
        
    Returns:
        统计检验结果
    """
    from scipy import stats
    
    # 执行配对t检验
    t_stat, p_value = stats.ttest_rel(model1_metrics, model2_metrics)
    
    # 计算效应量（Cohen's d）
    diff = np.array(model1_metrics) - np.array(model2_metrics)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05,
        'metric': metric_name,
        'mean_model1': np.mean(model1_metrics),
        'mean_model2': np.mean(model2_metrics),
        'std_model1': np.std(model1_metrics),
        'std_model2': np.std(model2_metrics)
    }