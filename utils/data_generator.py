import numpy as np
import torch
from typing import Tuple, Optional, List, Dict, Any
import math


def generate_lwe_samples(
    n: int,
    q: int,
    secret: np.ndarray,
    num_samples: int,
    error_std: float = 0.5,
    error_distribution: str = 'gaussian',
    secret_distribution: str = 'uniform'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成LWE样本: (A, b = A·s + e mod q)
    
    Args:
        n: LWE维度
        q: 模数
        secret: 秘密向量
        num_samples: 样本数量
        error_std: 误差标准差
        error_distribution: 误差分布 ('gaussian', 'uniform', 'binary')
        secret_distribution: 秘密分布 ('uniform', 'binary', 'ternary')
        
    Returns:
        (A, b) 元组
    """
    # 生成随机矩阵A
    A = np.random.randint(0, q, (num_samples, n))
    
    # 生成误差向量e
    if error_distribution == 'gaussian':
        e = np.random.normal(0, error_std, num_samples)
    elif error_distribution == 'uniform':
        e = np.random.uniform(-error_std, error_std, num_samples)
    elif error_distribution == 'binary':
        e = np.random.choice([-1, 1], num_samples) * error_std
    else:
        raise ValueError(f"Unsupported error distribution: {error_distribution}")
    
    # 计算b = A·s + e mod q
    b = (np.dot(A, secret) + e) % q
    b = b.astype(np.int64)  # 转换为整数
    
    return A, b


def generate_uniform_samples(
    n: int,
    q: int,
    num_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成均匀随机样本：(A, b) ~ U(Z_q^n x Z_q)
    
    Args:
        n: 维度
        q: 模数
        num_samples: 样本数量
        
    Returns:
        (A, b) 元组
    """
    A = np.random.randint(0, q, (num_samples, n))
    b = np.random.randint(0, q, num_samples)
    
    return A, b


def generate_secret(
    n: int,
    q: int,
    distribution: str = 'uniform',
    sparse_ratio: float = 0.5
) -> np.ndarray:
    """
    生成LWE秘密向量
    
    Args:
        n: 向量维度
        q: 模数
        distribution: 分布类型 ('uniform', 'binary', 'ternary', 'sparse')
        sparse_ratio: 稀疏比例(仅用于sparse分布)
        
    Returns:
        秘密向量
    """
    if distribution == 'uniform':
        return np.random.randint(0, q, n)
    elif distribution == 'binary':
        return np.random.randint(0, 2, n)
    elif distribution == 'ternary':
        return np.random.randint(-1, 2, n)
    elif distribution == 'sparse':
        secret = np.zeros(n, dtype=int)
        #　num_nonzero = max(1, int(n * sparse_ratio))
        num_nonzero = 10
        indices = np.random.choice(n, num_nonzero, replace=False)
        secret[indices] = np.random.randint(1, q, num_nonzero)
        return secret
    else:
        raise ValueError(f"Unsupported secret distribution: {distribution}")


def generate_lwe_problem(
    n: int,
    q: int,
    num_samples: int,
    error_std: float = 0.5,
    secret_distribution: str = 'uniform',
    error_distribution: str = 'gaussian',
    return_secret: bool = False
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Optional[np.ndarray]]:
    """
    生成完整的decisional-LWE问题数据集
    
    Args:
        n: LWE维度
        q: 模数
        num_samples: 总样本数量(一半LWE, 一半均匀)
        error_std: 误差标准差
        secret_distribution: 秘密分布
        error_distribution: 误差分布
        return_secret: 是否返回秘密向量
        
    Returns:
        (lwe_samples, uniform_samples, secret) 元组
    """
    # 生成秘密向量
    secret = generate_secret(n, q, secret_distribution)
    
    # 生成LWE样本
    lwe_samples = generate_lwe_samples(
        n, q, secret, num_samples // 2,
        error_std, error_distribution, secret_distribution
    )
    
    # 生成均匀样本
    uniform_samples = generate_uniform_samples(n, q, num_samples // 2)
    
    if return_secret:
        return lwe_samples, uniform_samples, secret
    else:
        return lwe_samples, uniform_samples, None


def generate_structured_lwe_samples(
    n: int,
    q: int,
    secret: np.ndarray,
    num_samples: int,
    structure_type: str = 'toeplitz',
    error_std: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成结构化LWE样本
    
    Args:
        n: LWE维度
        q: 模数
        secret: 秘密向量
        num_samples: 样本数量
        structure_type: 结构类型 ('toeplitz', 'cyclic', 'block')
        error_std: 误差标准差
        
    Returns:
        (A, b) 元组
    """
    if structure_type == 'toeplitz':
        # Toeplitz矩阵：每条对角线上的元素相同
        A = np.zeros((num_samples, n), dtype=int)
        for i in range(num_samples):
            first_row = np.random.randint(0, q, n)
            first_col = np.random.randint(0, q, num_samples)
            first_col[0] = first_row[0]
            for j in range(n):
                if i - j >= 0:
                    A[i, j] = first_col[i - j]
                else:
                    A[i, j] = first_row[j - i]
    
    elif structure_type == 'cyclic':
        # 循环矩阵：每行是前一行的循环移位
        A = np.zeros((num_samples, n), dtype=int)
        first_row = np.random.randint(0, q, n)
        for i in range(num_samples):
            A[i] = np.roll(first_row, i)
    
    elif structure_type == 'block':
        # 分块对角矩阵
        block_size = max(2, n // 4)
        A = np.zeros((num_samples, n), dtype=int)
        for i in range(0, num_samples, block_size):
            for j in range(0, n, block_size):
                block = np.random.randint(0, q, (min(block_size, num_samples - i),
                                               min(block_size, n - j)))
                A[i:i+block_size, j:j+block_size] = block
    
    else:
        raise ValueError(f"Unsupported structure type: {structure_type}")
    
    # 生成误差并计算b
    e = np.random.normal(0, error_std, num_samples)
    b = (np.dot(A, secret) + e) % q
    b = b.astype(np.int64)
    
    return A, b


def calculate_lwe_parameters(
    security_level: int,
    q: int = None
) -> Dict[str, Any]:
    """
    根据安全等级计算LWE参数
    
    Args:
        security_level: 安全等级（比特）
        q: 模数(如果为None则自动计算)
        
    Returns:
        LWE参数字典
    """
    # 基于经验公式的简单估计
    # 实际应用中应使用更精确的估计
    
    if q is None:
        # 根据安全等级选择模数
        if security_level <= 80:
            q = 2**13  # 8192
        elif security_level <= 128:
            q = 2**15  # 32768
        else:
            q = 2**16  # 65536
    
    # 估计维度n
    n = int(security_level * 2.5)
    
    # 估计误差标准差
    error_std = math.sqrt(n) / 2
    
    return {
        'n': n,
        'q': q,
        'error_std': error_std,
        'security_level': security_level,
        'recommended_samples': n * 10  # 经验推荐样本数量
    }


def validate_lwe_parameters(
    n: int,
    q: int,
    error_std: float
) -> Dict[str, bool]:
    """
    验证LWE参数的安全性
    
    Args:
        n: 维度
        q: 模数
        error_std: 误差标准差
        
    Returns:
        验证结果字典
    """
    results = {}
    
    # 检查维度是否足够大
    results['dimension_adequate'] = n >= 64
    
    # 检查模数大小
    results['modulus_adequate'] = q >= 1024
    
    # 检查误差率（误差标准差相对于模数的比例）
    error_ratio = error_std / q
    results['error_ratio_adequate'] = 0.001 <= error_ratio <= 0.1
    
    # 检查安全等级估计
    estimated_security = min(n * 0.4, math.log2(q) * 0.8)
    results['security_adequate'] = estimated_security >= 80
    
    return results


def add_label_noise(
    labels: np.ndarray,
    noise_ratio: float
) -> np.ndarray:
    """
    为标签添加噪声（用于研究模型的鲁棒性）
    
    Args:
        labels: 原始标签
        noise_ratio: 噪声比例
        
    Returns:
        添加噪声后的标签
    """
    noisy_labels = labels.copy()
    n_noisy = int(len(labels) * noise_ratio)
    noisy_indices = np.random.choice(len(labels), n_noisy, replace=False)
    noisy_labels[noisy_indices] = 1 - noisy_labels[noisy_indices]  # 翻转标签
    
    return noisy_labels


def generate_parameter_sweep_datasets(
    n_range: List[int],
    q_range: List[int],
    error_std_range: List[float],
    samples_per_config: int = 1000
) -> Dict[str, Any]:
    """
    生成参数扫描数据集, 用于研究不同参数对LWE问题难度的影响
    
    Args:
        n_range: 维度范围
        q_range: 模数范围
        error_std_range: 误差标准差范围
        samples_per_config: 每个配置的样本数量
        
    Returns:
        参数字典，包含所有配置的数据集
    """
    datasets = {}
    
    for n in n_range:
        for q in q_range:
            for error_std in error_std_range:
                key = f"n_{n}_q_{q}_error_{error_std}"
                
                # 生成秘密向量
                secret = generate_secret(n, q, 'spare')
                
                # 生成LWE和均匀样本
                lwe_samples = generate_lwe_samples(n, q, secret, samples_per_config // 2, error_std)
                uniform_samples = generate_uniform_samples(n, q, samples_per_config // 2)
                
                datasets[key] = {
                    'lwe_samples': lwe_samples,
                    'uniform_samples': uniform_samples,
                    'secret': secret,
                    'parameters': {'n': n, 'q': q, 'error_std': error_std}
                }
    
    return datasets


def save_lwe_dataset(
    dataset: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    filepath: str,
    metadata: Dict[str, Any] = None
) -> None:
    """
    保存LWE数据集到文件
    
    Args:
        dataset: 数据集元组 (lwe_samples, uniform_samples)
        filepath: 文件路径
        metadata: 元数据
    """
    lwe_samples, uniform_samples = dataset
    
    data_to_save = {
        'lwe_samples': lwe_samples,
        'uniform_samples': uniform_samples,
        'metadata': metadata or {}
    }
    
    np.savez(filepath, **data_to_save)
    print(f"Dataset saved to {filepath}")


def load_lwe_dataset(filepath: str) -> Tuple[Tuple, Tuple, Dict]:
    """
    从文件加载LWE数据集
    
    Args:
        filepath: 文件路径
        
    Returns:
        (lwe_samples, uniform_samples, metadata) 元组
    """
    data = np.load(filepath, allow_pickle=True)
    
    lwe_samples = (data['lwe_samples'][0], data['lwe_samples'][1])
    uniform_samples = (data['uniform_samples'][0], data['uniform_samples'][1])
    metadata = data['metadata'].item() if 'metadata' in data else {}
    
    print(f"Dataset loaded from {filepath}")
    return lwe_samples, uniform_samples, metadata