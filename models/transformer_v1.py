import torch
import torch.nn as nn
import math
import numpy as np
from typing import Tuple, Optional

class PositionalEncoding(nn.Module):
    """
    位置编码层, 为Transformer添加序列位置信息
    
    Args:
        d_model: 嵌入维度
        max_len: 最大序列长度
        dropout: Dropout率
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # 注册为buffer（不参与训练的参数）
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [seq_len, batch_size, d_model] 或 [batch_size, seq_len, d_model]
        
        Returns:
            添加位置编码后的张量
        """
        # 根据输入形状调整位置编码
        if x.dim() == 3 and x.size(1) == self.pe.size(0):  # [batch_size, seq_len, d_model]
            x = x + self.pe.permute(1, 0, 2)[:, :x.size(1), :]
        else:  # [seq_len, batch_size, d_model]
            x = x + self.pe[:x.size(0), :]
        
        return self.dropout(x)


class LWETransformer(nn.Module):
    """
    基于Transformer的decisional-LWE分类器
    
    Args:
        vocab_size: 词汇表大小(模数q)
        d_model: 模型维度
        nhead: 注意力头数
        num_layers: Transformer层数
        dim_feedforward: 前馈网络维度
        max_seq_len: 最大序列长度 (n+1)
        n_classes: 分类类别数
        dropout: Dropout率
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        max_seq_len: int,
        n_classes: int = 2,
        dropout: float = 0.1
    ):
        super(LWETransformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # 使用batch_first格式
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self) -> None:
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        A: torch.Tensor, 
        b: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            A: 矩阵A [batch_size, n]
            b: 向量b [batch_size]
            attention_mask: 注意力掩码
            
        Returns:
            分类logits [batch_size, n_classes]
        """
        batch_size = A.size(0)
        
        # 将A和b组合成序列 [A, b]
        # A: [batch_size, n], b: [batch_size] -> b: [batch_size, 1]
        b_expanded = b.unsqueeze(1)
        sequence = torch.cat([A, b_expanded], dim=1)  # [batch_size, n+1]
        
        # 嵌入层
        embedded = self.embedding(sequence)  # [batch_size, n+1, d_model]
        
        # 缩放嵌入（Transformer的标准做法）
        embedded = embedded * math.sqrt(self.d_model)
        
        # 位置编码
        encoded = self.pos_encoder(embedded)  # [batch_size, n+1, d_model]
        
        # 创建序列掩码（防止关注到填充位置）
        if attention_mask is None:
            # 假设所有位置都是有效的
            src_key_padding_mask = None
        else:
            src_key_padding_mask = ~attention_mask.bool()
        
        # Transformer编码
        transformer_output = self.transformer_encoder(
            encoded, 
            src_key_padding_mask=src_key_padding_mask
        )  # [batch_size, n+1, d_model]
        
        # 使用第一个token（CLS token等价物）进行分类
        cls_output = transformer_output[:, 0, :]  # [batch_size, d_model]
        
        # 分类
        logits = self.classifier(cls_output)  # [batch_size, n_classes]
        
        return logits
    
    def predict_proba(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        预测概率
        
        Args:
            A: 矩阵A
            b: 向量b
            
        Returns:
            类别概率 [batch_size, n_classes]
        """
        with torch.no_grad():
            logits = self.forward(A, b)
            return torch.softmax(logits, dim=-1)
    
    def predict(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        预测类别
        
        Args:
            A: 矩阵A
            b: 向量b
            
        Returns:
            预测类别 [batch_size]
        """
        with torch.no_grad():
            logits = self.forward(A, b)
            return torch.argmax(logits, dim=-1)


class LWETransformerWithFeatures(LWETransformer):
    """
    带额外特征的LWE Transformer
    
    可以添加LWE问题的统计特征来增强模型
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        max_seq_len: int,
        feature_dim: int = 10,
        n_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__(
            vocab_size, d_model, nhead, num_layers, 
            dim_feedforward, max_seq_len, n_classes, dropout
        )
        
        # 特征处理层
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 修改分类器以包含特征信息
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, dim_feedforward),  # 2倍维度
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_classes)
        )
    
    def extract_features(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        提取LWE样本的统计特征
        
        Args:
            A: 矩阵A
            b: 向量b
            
        Returns:
            特征向量 [batch_size, feature_dim]
        """
        features = []
        
        # A的统计特征
        features.append(A.float().mean(dim=1))  # 均值
        features.append(A.float().std(dim=1))   # 标准差
        features.append(A.float().max(dim=1)[0])  # 最大值
        features.append(A.float().min(dim=1)[0])  # 最小值
        
        # b的统计特征
        features.append(b.float().unsqueeze(1))  # b值本身
        
        # A和b的组合特征
        A_float = A.float()
        b_float = b.float().unsqueeze(1)
        dot_products = torch.bmm(A_float.unsqueeze(1), A_float.unsqueeze(2)).squeeze()
        features.append(dot_products)
        
        return torch.stack(features, dim=1).squeeze()
    
    def forward(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        batch_size = A.size(0)
        
        # Transformer路径
        sequence = torch.cat([A, b.unsqueeze(1)], dim=1)
        embedded = self.embedding(sequence) * math.sqrt(self.d_model)
        encoded = self.pos_encoder(embedded)
        transformer_output = self.transformer_encoder(encoded)
        cls_output = transformer_output[:, 0, :]
        
        # 特征路径
        features = self.extract_features(A, b)
        processed_features = self.feature_processor(features)
        
        # 合并两个路径
        combined = torch.cat([cls_output, processed_features], dim=1)
        logits = self.classifier(combined)
        
        return logits


# 便捷函数，用于创建标准模型
def create_standard_lwe_transformer(
    vocab_size: int,
    n: int,
    n_classes: int = 2
) -> LWETransformer:
    """
    创建标准配置的LWE Transformer
    
    Args:
        vocab_size: 词汇表大小(模数q)
        n: LWE维度
        n_classes: 分类数
        
    Returns:
        配置好的Transformer模型
    """
    return LWETransformer(
        vocab_size=vocab_size,
        d_model=128,
        nhead=8,
        num_layers=6,
        dim_feedforward=256,
        max_seq_len=n + 1,
        n_classes=n_classes,
        dropout=0.1
    )


def create_large_lwe_transformer(
    vocab_size: int,
    n: int,
    n_classes: int = 2
) -> LWETransformer:
    """
    创建大型LWE Transformer
    
    Args:
        vocab_size: 词汇表大小(模数q)
        n: LWE维度
        n_classes: 分类数
        
    Returns:
        配置好的大型Transformer模型
    """
    return LWETransformer(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_layers=8,
        dim_feedforward=512,
        max_seq_len=n + 1,
        n_classes=n_classes,
        dropout=0.1
    )