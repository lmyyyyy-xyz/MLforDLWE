import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Union
import pickle
import os


class LWEDataset(Dataset):
    """
    LWE数据集类，用于decisional-LWE问题的训练和评估
    
    Args:
        lwe_samples: LWE样本元组 (A, b)
        uniform_samples: 均匀随机样本元组 (A, b)
        transform: 数据转换函数
        normalize: 是否归一化数据
    """
    
    def __init__(
        self,
        lwe_samples: Tuple[np.ndarray, np.ndarray],
        uniform_samples: Tuple[np.ndarray, np.ndarray],
        transform: Optional[callable] = None,
        normalize: bool = False
    ):
        self.samples = []
        self.labels = []
        self.transform = transform
        self.normalize = normalize
        
        # 处理LWE样本（标签为1）
        A_lwe, b_lwe = lwe_samples
        for i in range(len(A_lwe)):
            self.samples.append((A_lwe[i], b_lwe[i]))
            self.labels.append(1)  # LWE样本标签为1
        
        # 处理均匀随机样本（标签为0）
        A_uni, b_uni = uniform_samples
        for i in range(len(A_uni)):
            self.samples.append((A_uni[i], b_uni[i]))
            self.labels.append(0)  # 均匀样本标签为0
        
        # 转换为numpy数组以便后续处理
        self.samples = np.array(self.samples, dtype=object)
        self.labels = np.array(self.labels)
        
        # 计算归一化参数（如果需要）
        if self.normalize:
            self._compute_normalization_params()
    
    def _compute_normalization_params(self) -> None:
        """计算归一化参数"""
        all_A = np.vstack([sample[0] for sample in self.samples])
        all_b = np.array([sample[1] for sample in self.samples])
        
        self.A_mean = all_A.mean(axis=0)
        self.A_std = all_A.std(axis=0) + 1e-8  # 避免除零
        
        self.b_mean = all_b.mean()
        self.b_std = all_b.std() + 1e-8
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            (A, b, label) 元组
        """
        A, b = self.samples[idx]
        label = self.labels[idx]
        
        # 转换为tensor
        A_tensor = torch.tensor(A, dtype=torch.long)
        b_tensor = torch.tensor(b, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # 归一化（如果需要）
        if self.normalize:
            A_tensor = (A_tensor.float() - torch.tensor(self.A_mean)) / torch.tensor(self.A_std)
            b_tensor = (b_tensor.float() - self.b_mean) / self.b_std
        
        # 应用转换（如果有）
        if self.transform:
            A_tensor, b_tensor = self.transform(A_tensor, b_tensor)
        
        return A_tensor, b_tensor, label_tensor
    
    def get_statistics(self) -> dict:
        """获取数据集统计信息"""
        stats = {
            'total_samples': len(self),
            'lwe_samples': np.sum(self.labels == 1),
            'uniform_samples': np.sum(self.labels == 0),
            'class_ratio': np.sum(self.labels == 1) / len(self.labels)
        }
        
        if self.normalize:
            stats.update({
                'A_mean': self.A_mean.tolist(),
                'A_std': self.A_std.tolist(),
                'b_mean': float(self.b_mean),
                'b_std': float(self.b_std)
            })
        
        return stats
    
    def save(self, filepath: str) -> None:
        """保存数据集到文件"""
        data = {
            'samples': self.samples,
            'labels': self.labels,
            'normalize': self.normalize,
            'transform': self.transform
        }
        
        if self.normalize:
            data.update({
                'A_mean': self.A_mean,
                'A_std': self.A_std,
                'b_mean': self.b_mean,
                'b_std': self.b_std
            })
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'LWEDataset':
        """从文件加载数据集"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # 重新构建数据集
        lwe_mask = data['labels'] == 1
        uniform_mask = data['labels'] == 0
        
        lwe_samples = (
            np.array([sample[0] for sample in data['samples'][lwe_mask]]),
            np.array([sample[1] for sample in data['samples'][lwe_mask]])
        )
        
        uniform_samples = (
            np.array([sample[0] for sample in data['samples'][uniform_mask]]),
            np.array([sample[1] for sample in data['samples'][uniform_mask]])
        )
        
        dataset = cls(lwe_samples, uniform_samples, data.get('transform'), data.get('normalize', False))
        
        # 恢复归一化参数
        if data.get('normalize', False):
            dataset.A_mean = data['A_mean']
            dataset.A_std = data['A_std']
            dataset.b_mean = data['b_mean']
            dataset.b_std = data['b_std']
        
        return dataset


class LWEBatchSampler:
    """
    自定义批次采样器，确保每个批次中正负样本平衡
    """
    
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 处理 Subset 对象的情况
        if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'labels'):
            # 如果是 Subset 对象，通过 indices 获取标签
            self.labels = dataset.dataset.labels[dataset.indices]
        elif hasattr(dataset, 'labels'):
            # 如果是普通 LWEDataset 对象
            self.labels = dataset.labels
        else:
            # 如果无法获取标签，抛出错误
            raise AttributeError("数据集对象没有 labels 属性")
        
        # 分离正负样本索引
        self.positive_indices = np.where(self.labels == 1)[0]
        self.negative_indices = np.where(self.labels == 0)[0]
        
        assert len(self.positive_indices) > 0 and len(self.negative_indices) > 0, \
            "数据集必须包含正负样本"
        
        # 确保批次大小是偶数，以便平衡采样
        if self.batch_size % 2 != 0:
            self.batch_size += 1
            print(f"警告：批次大小调整为偶数: {self.batch_size}")
    
    def __iter__(self):
        """生成平衡的批次"""
        n_batches = len(self) // 2  # 每个批次包含相同数量的正负样本
        
        if self.shuffle:
            np.random.shuffle(self.positive_indices)
            np.random.shuffle(self.negative_indices)
        
        for i in range(n_batches):
            # 从正负样本中各取一半
            batch_positive = self.positive_indices[i * self.batch_size // 2:(i + 1) * self.batch_size // 2]
            batch_negative = self.negative_indices[i * self.batch_size // 2:(i + 1) * self.batch_size // 2]
            
            # 合并并打乱顺序
            batch = np.concatenate([batch_positive, batch_negative])
            if self.shuffle:
                np.random.shuffle(batch)
            
            yield batch
    
    def __len__(self):
        """返回批次数量"""
        min_samples = min(len(self.positive_indices), len(self.negative_indices))
        return (min_samples * 2) // self.batch_size


def create_data_loaders(
    dataset: LWEDataset,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    balanced_sampling: bool = True,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证、测试数据加载器
    
    Args:
        dataset: LWE数据集
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        balanced_sampling: 是否使用平衡采样
        shuffle: 是否打乱数据
        random_seed: 随机种子
        
    Returns:
        (train_loader, val_loader, test_loader) 元组
    """
    # 验证比例总和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"
    
    # 设置随机种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # 分离正负样本
    positive_indices = np.where(dataset.labels == 1)[0]
    negative_indices = np.where(dataset.labels == 0)[0]
    
    # 分别对正负样本进行分割
    pos_train_idx = int(len(positive_indices) * train_ratio)
    pos_val_idx = pos_train_idx + int(len(positive_indices) * val_ratio)
    
    neg_train_idx = int(len(negative_indices) * train_ratio)
    neg_val_idx = neg_train_idx + int(len(negative_indices) * val_ratio)
    
    # 创建索引分割
    if shuffle:
        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)
    
    train_indices = np.concatenate([
        positive_indices[:pos_train_idx],
        negative_indices[:neg_train_idx]
    ])
    
    val_indices = np.concatenate([
        positive_indices[pos_train_idx:pos_val_idx],
        negative_indices[neg_train_idx:neg_val_idx]
    ])
    
    test_indices = np.concatenate([
        positive_indices[pos_val_idx:],
        negative_indices[neg_val_idx:]
    ])
    
    # 打乱各集合内的顺序
    if shuffle:
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
    
    # 创建子数据集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # 为子集添加标签属性，以便 LWEBatchSampler 使用
    train_dataset.labels = dataset.labels[train_indices]
    val_dataset.labels = dataset.labels[val_indices]
    test_dataset.labels = dataset.labels[test_indices]
    
    # 创建数据加载器
    if balanced_sampling:
        train_sampler = LWEBatchSampler(train_dataset, batch_size, shuffle)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


class LWEIterativeDataset(Dataset):
    """
    迭代式LWE数据集，用于大规模数据或在线学习场景
    只在需要时生成数据，节省内存
    """
    
    def __init__(
        self,
        n: int,
        q: int,
        secret: np.ndarray,
        num_samples: int,
        error_std: float = 0.5,
        transform: Optional[callable] = None
    ):
        self.n = n
        self.q = q
        self.secret = secret
        self.num_samples = num_samples
        self.error_std = error_std
        self.transform = transform
        
        # 预计算一半LWE样本，一半均匀样本
        self.lwe_count = num_samples // 2
        self.uniform_count = num_samples - self.lwe_count
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 根据索引决定生成LWE样本还是均匀样本
        if idx < self.lwe_count:
            # 生成LWE样本
            A = np.random.randint(0, self.q, self.n)
            e = np.random.normal(0, self.error_std)
            b = (np.dot(A, self.secret) + e) % self.q
            label = 1
        else:
            # 生成均匀样本
            A = np.random.randint(0, self.q, self.n)
            b = np.random.randint(0, self.q)
            label = 0
        
        # 转换为tensor
        A_tensor = torch.tensor(A, dtype=torch.long)
        b_tensor = torch.tensor(b, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # 应用转换（如果有）
        if self.transform:
            A_tensor, b_tensor = self.transform(A_tensor, b_tensor)
        
        return A_tensor, b_tensor, label_tensor


# 数据转换函数
def random_shift(A: torch.Tensor, b: torch.Tensor, max_shift: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """随机平移增强"""
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    A_shifted = (A + shift) % A.max().item() if A.max().item() > 0 else A
    b_shifted = (b + shift) % b.max().item() if b.max().item() > 0 else b
    return A_shifted, b_shifted


def add_noise(A: torch.Tensor, b: torch.Tensor, noise_level: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """添加噪声增强"""
    A_noise = A + torch.randn_like(A.float()) * noise_level
    b_noise = b + torch.randn_like(b.float()) * noise_level
    return A_noise.long(), b_noise.long()


def create_composite_transform(*transforms):
    """组合多个转换函数"""
    def composite_transform(A, b):
        for transform in transforms:
            A, b = transform(A, b)
        return A, b
    return composite_transform
