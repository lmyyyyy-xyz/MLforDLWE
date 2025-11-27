import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# 最大序列长度
N_MAX_POSITIONS = 4096

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    """自定义嵌入层初始化"""
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m

def create_sinusoidal_embeddings(n_pos, dim, out):
    """创建正弦位置编码"""
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False

def gelu(x):
    """GELU激活函数"""
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

def get_masks(slen, lengths, causal):
    """生成注意力掩码"""
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    if causal:
        attn_mask = (alen[None, :].repeat(slen, 1) <= alen[:, None]).repeat(bs, 1, 1)
    else:
        attn_mask = mask

    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, src_dim, dropout, normalized_attention=False, xav_init=False):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.src_dim = src_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.normalized_attention = normalized_attention
        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(src_dim, dim)
        self.v_lin = nn.Linear(src_dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        
        if self.normalized_attention:
            self.attention_scale = nn.Parameter(
                torch.tensor(1.0 / math.sqrt(dim // n_heads))
            )
            
        if xav_init:
            gain = (1 / math.sqrt(2)) if self.src_dim == self.dim else 1.0
            nn.init.xavier_uniform_(self.q_lin.weight, gain=gain)
            nn.init.xavier_uniform_(self.k_lin.weight, gain=gain)
            nn.init.xavier_uniform_(self.v_lin.weight, gain=gain)
            nn.init.xavier_uniform_(self.out_lin.weight)
            nn.init.constant_(self.out_lin.bias, 0.0)

    def forward(self, input, mask, kv=None, use_cache=False, first_loop=True):
        """
        前向传播
        
        Args:
            input: 输入张量 [batch_size, seq_len, dim]
            mask: 注意力掩码
            kv: 键值对 (可选)
            use_cache: 是否使用缓存
            first_loop: 是否第一次循环
        """
        bs, qlen, dim = input.size()
        
        if kv is None:
            klen = qlen
        else:
            klen = kv.size(1)
            
        assert dim == self.dim, f"维度不匹配: {dim} 输入 vs {self.dim} 配置"

        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """投影到多头形状"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """从多头形状恢复"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        # 查询投影
        q = shape(self.q_lin(input))  # [bs, n_heads, qlen, dim_per_head]
        
        # 键值投影
        if kv is None:
            k = shape(self.k_lin(input))
            v = shape(self.v_lin(input))
        else:
            k = shape(self.k_lin(kv))
            v = shape(self.v_lin(kv))

        # 归一化注意力
        if self.normalized_attention:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            q = q * self.attention_scale
        else:
            q = q / math.sqrt(dim_per_head)

        # 注意力分数
        scores = torch.matmul(q, k.transpose(2, 3))  # [bs, n_heads, qlen, klen]
        
        # 应用掩码
        mask = ((mask == 0).view(mask_reshape).expand_as(scores))
        scores.masked_fill_(mask, -float("inf"))

        # 注意力权重
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        # 上下文计算
        context = torch.matmul(weights, v)  # [bs, n_heads, qlen, dim_per_head]
        context = unshape(context)  # [bs, qlen, dim]

        return self.out_lin(context)

class TransformerFFN(nn.Module):
    """Transformer前馈网络"""
    def __init__(self, in_dim, dim_hidden, out_dim, hidden_layers=1, dropout=0.1, 
                 gelu_activation=False, xav_init=False):
        super().__init__()
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.act = gelu if gelu_activation else F.relu
        
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        self.midlin = nn.ModuleList()
        
        for i in range(1, self.hidden_layers):
            self.midlin.append(nn.Linear(dim_hidden, dim_hidden))
            
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        
        if xav_init:
            nn.init.xavier_uniform_(self.lin1.weight)
            nn.init.constant_(self.lin1.bias, 0.0)
            for mlin in self.midlin:
                nn.init.xavier_uniform_(mlin.weight)
                nn.init.constant_(mlin.bias, 0.0)
            nn.init.xavier_uniform_(self.lin2.weight)
            nn.init.constant_(self.lin2.bias, 0.0)

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        
        for mlin in self.midlin:
            x = mlin(x)
            x = self.act(x)
            
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class Gate(nn.Module):
    """门控机制"""
    def __init__(self, dimension, scalar=True, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.gate1 = nn.Linear(dimension, 4 * dimension)
        self.gate2 = nn.Linear(4 * dimension, 1 if scalar else dimension)

    def forward(self, x):
        outp = self.gate1(x)
        outp = F.relu(outp)
        outp = F.dropout(outp, p=self.dropout, training=self.training)
        outp = self.gate2(outp)
        return torch.sigmoid(outp)

class TransformerLayer(nn.Module):
    """Transformer层"""
    def __init__(self, dim, src_dim, n_heads, hidden_dim, n_hidden_layers=1, 
                 dropout=0.1, attention_dropout=0.1, normalized_attention=False,
                 gated=False, scalar_gate=True, is_encoder=True):
        super().__init__()
        
        self.dim = dim
        self.src_dim = src_dim
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.normalized_attention = normalized_attention
        self.gated = gated
        self.scalar_gate = scalar_gate
        self.is_encoder = is_encoder

        assert self.dim % self.n_heads == 0

        # 自注意力
        self.self_attention = MultiHeadAttention(
            self.n_heads,
            self.dim,
            self.dim,
            dropout=self.attention_dropout,
            normalized_attention=normalized_attention,
        )
        self.layer_norm1 = nn.LayerNorm(self.dim, eps=1e-12)

        # 交叉注意力 (仅用于解码器)
        if not self.is_encoder:
            self.layer_norm15 = nn.LayerNorm(self.dim, eps=1e-12)
            self.cross_attention = MultiHeadAttention(
                self.n_heads,
                self.dim,
                self.src_dim,
                dropout=self.attention_dropout,
                normalized_attention=normalized_attention,
            )

        # 前馈网络
        self.ffn = TransformerFFN(
            self.dim,
            self.hidden_dim,
            self.dim,
            hidden_layers=self.n_hidden_layers,
            dropout=self.dropout,
            gelu_activation=True,
        )
        self.layer_norm2 = nn.LayerNorm(self.dim, eps=1e-12)
        
        # 门控机制
        if self.gated:
            self.gate = Gate(self.dim, self.scalar_gate, self.dropout)

    def forward(self, x, attn_mask, src_mask=None, src_enc=None, loop_count=1):
        """
        前向传播
        
        Args:
            x: 输入张量
            attn_mask: 注意力掩码
            src_mask: 源序列掩码 (用于解码器)
            src_enc: 编码器输出 (用于解码器)
            loop_count: 循环次数 (用于自适应停止)
        """
        tensor = x
        
        for i in range(loop_count):
            # 自注意力
            attn = self.self_attention(tensor, attn_mask)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            output = tensor + attn
            output = self.layer_norm1(output)

            if self.gated:
                gate = self.gate(output)
            
            # 交叉注意力 (仅解码器)
            if not self.is_encoder and src_enc is not None:
                attn = self.cross_attention(output, src_mask, kv=src_enc)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                output = output + attn
                output = self.layer_norm15(output)

            # 前馈网络
            ffn_output = self.ffn(output)
            output = output + ffn_output
            output = self.layer_norm2(output)
            
            # 门控融合
            if self.gated:
                tensor = gate * output + (1 - gate) * tensor
            else:
                tensor = output
                
        return tensor

class AdaptiveHalt(nn.Module):
    """自适应停止机制"""
    def __init__(self, dim, max_loops, threshold=0.99, ponder_coupling=1e-2, 
                 gated=False, scalar_gate=True, dropout=0.1, **layer_kwargs):
        super().__init__()
        self.dim = dim
        self.max_loops = max_loops
        self.threshold = 1.0 - threshold
        self.ponder_coupling = ponder_coupling
        
        self.halt_prob = nn.Linear(self.dim, 1)
        self.ponder_penalty = 0
        self.layer = TransformerLayer(dim=dim, **layer_kwargs)

    def forward(self, input, attn_mask, src_mask=None, src_enc=None, loop_count=1):
        bs, slen, dim = input.size()
        shape = (bs, slen)
        
        halting_probability = torch.zeros(shape, device=input.device)
        remainders = torch.zeros_like(halting_probability)
        acc_state = torch.zeros_like(input)
        
        for i in range(self.max_loops):
            # 停止概率
            p = torch.squeeze(torch.sigmoid(self.halt_prob(input)), -1)
            
            # 仍在运行的token
            still_running = (halting_probability < 1.0).float()
            
            # 本轮停止的token
            new_halted = ((halting_probability + p * still_running) > self.threshold).float() * still_running
            
            # 更新运行状态
            still_running = ((halting_probability + p * still_running) <= self.threshold).float() * still_running

            # 更新停止概率
            halting_probability = halting_probability + p * still_running
            
            # 计算余量
            remainders = remainders + new_halted * (1 - halting_probability)
            halting_probability = halting_probability + new_halted * remainders

            # 更新状态
            input = self.layer.forward(input, attn_mask, src_mask, src_enc, loop_count)
            
            # 加权最终状态
            update_weights = torch.unsqueeze(p * still_running + new_halted * remainders, -1)
            acc_state = (input * update_weights) + (acc_state * (1 - update_weights))
            
            # 如果所有token都停止，提前退出
            if still_running.sum() == 0:
                break
        
        # 计算惩罚项
        remainders = remainders + (halting_probability < 1.0).float() * (1 - halting_probability)
        self.ponder_penalty = self.ponder_coupling * torch.mean(remainders)
        
        return acc_state

class PositionalEncoding(nn.Module):
    """位置编码层 (兼容原有接口)"""
    def __init__(self, d_model, max_len=5000, dropout=0.1, sinusoidal=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.sinusoidal = sinusoidal
        
        if sinusoidal:
            # 正弦位置编码
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))
        else:
            # 可学习的位置编码
            self.pe = Embedding(max_len, d_model)

    def forward(self, x):
        if self.sinusoidal:
            x = x + self.pe[:, :x.size(1)]
        else:
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            position_embeddings = self.pe(positions)
            x = x + position_embeddings
            
        return self.dropout(x)

class LWETransformer(nn.Module):
    """
    LWE Transformer 模型
    基于提供的代码重构，包含自适应停止、门控机制等高级特性
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
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        normalized_attention: bool = False,
        gated: bool = False,
        adaptive_halt: bool = False,
        max_halt_loops: int = 10,
        halt_threshold: float = 0.99,
        sinusoidal_embeddings: bool = True,
        scalar_gate: bool = True,
        n_hidden_layers: int = 1,
        xav_init: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_len = max_seq_len
        self.n_classes = n_classes
        self.adaptive_halt = adaptive_halt
        self.max_halt_loops = max_halt_loops
        
        # 嵌入层
        self.embedding = Embedding(vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(
            d_model, 
            max_len=max_seq_len, 
            dropout=dropout,
            sinusoidal=sinusoidal_embeddings
        )
        
        # Transformer 层
        self.layers = nn.ModuleList()
        
        for layer_id in range(num_layers):
            layer_kwargs = {
                'dim': d_model,
                'src_dim': d_model,
                'n_heads': nhead,
                'hidden_dim': dim_feedforward,
                'n_hidden_layers': n_hidden_layers,
                'dropout': dropout,
                'attention_dropout': attention_dropout,
                'normalized_attention': normalized_attention,
                'gated': gated and (layer_id == num_layers - 1),  # 仅在最后一层使用门控
                'scalar_gate': scalar_gate,
                'is_encoder': True
            }
            
            if adaptive_halt and layer_id == num_layers - 1:  # 在最后一层使用自适应停止
                self.layers.append(AdaptiveHalt(
                    dim=d_model,
                    max_loops=max_halt_loops,
                    threshold=halt_threshold,
                    gated=gated,
                    scalar_gate=scalar_gate,
                    dropout=dropout,
                    **layer_kwargs
                ))
            else:
                self.layers.append(TransformerLayer(**layer_kwargs))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_classes)
        )
        
        # 层归一化
        self.layer_norm_emb = nn.LayerNorm(d_model, eps=1e-12)
        
        # 初始化权重
        if xav_init:
            self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
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
        batch_size, n = A.size()
        
        # 组合序列 [A, b]
        b_expanded = b.unsqueeze(1)  # [batch_size, 1]
        sequence = torch.cat([A, b_expanded], dim=1)  # [batch_size, n+1]
        
        # 嵌入层
        embedded = self.embedding(sequence)  # [batch_size, n+1, d_model]
        embedded = embedded * math.sqrt(self.d_model)  # 缩放
        
        # 位置编码
        encoded = self.positional_encoding(embedded)
        encoded = self.layer_norm_emb(encoded)
        
        # 生成掩码
        seq_len = n + 1
        lengths = torch.full((batch_size,), seq_len, device=A.device)
        mask, attn_mask = get_masks(seq_len, lengths, causal=False)
        
        # 应用自定义注意力掩码
        if attention_mask is not None:
            attn_mask = attention_mask
        
        # Transformer 层
        tensor = encoded
        for layer in self.layers:
            if isinstance(layer, AdaptiveHalt):
                tensor = layer(tensor, attn_mask, loop_count=1)
            else:
                tensor = layer(tensor, attn_mask, loop_count=1)
        
        # 使用第一个token (CLS) 进行分类
        cls_output = tensor[:, 0, :]  # [batch_size, d_model]
        
        # 分类
        logits = self.classifier(cls_output)  # [batch_size, n_classes]
        
        return logits
    
    def predict_proba(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """预测概率"""
        with torch.no_grad():
            logits = self.forward(A, b)
            return torch.softmax(logits, dim=-1)
    
    def predict(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """预测类别"""
        with torch.no_grad():
            logits = self.forward(A, b)
            return torch.argmax(logits, dim=-1)

class AdvancedLWETransformer(LWETransformer):
    """
    高级LWE Transformer，包含更多特性
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
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        normalized_attention: bool = True,
        gated: bool = True,
        adaptive_halt: bool = True,
        max_halt_loops: int = 8,
        halt_threshold: float = 0.95,
        n_hidden_layers: int = 2,
        use_gelu: bool = True,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_seq_len=max_seq_len,
            n_classes=n_classes,
            dropout=dropout,
            attention_dropout=attention_dropout,
            normalized_attention=normalized_attention,
            gated=gated,
            adaptive_halt=adaptive_halt,
            max_halt_loops=max_halt_loops,
            halt_threshold=halt_threshold,
            n_hidden_layers=n_hidden_layers,
            **kwargs
        )
        
        # 修改分类器使用GELU
        if use_gelu:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, n_classes)
            )

# 便捷创建函数
def create_standard_lwe_transformer(
    vocab_size: int,
    n: int,
    n_classes: int = 2,
    **kwargs
) -> LWETransformer:
    """创建标准LWE Transformer"""
    return LWETransformer(
        vocab_size=vocab_size,
        d_model=128,
        nhead=8,
        num_layers=6,
        dim_feedforward=256,
        max_seq_len=n + 1,
        n_classes=n_classes,
        **kwargs
    )

def create_advanced_lwe_transformer(
    vocab_size: int,
    n: int,
    n_classes: int = 2,
    **kwargs
) -> AdvancedLWETransformer:
    """创建高级LWE Transformer"""
    return AdvancedLWETransformer(
        vocab_size=vocab_size,
        d_model=192,
        nhead=8,
        num_layers=8,
        dim_feedforward=384,
        max_seq_len=n + 1,
        n_classes=n_classes,
        normalized_attention=True,
        gated=True,
        adaptive_halt=True,
        max_halt_loops=6,
        **kwargs
    )

def create_sparse_lwe_transformer(
    vocab_size: int,
    n: int,
    n_classes: int = 2,
    **kwargs
) -> AdvancedLWETransformer:
    """创建针对稀疏LWE的Transformer"""
    return AdvancedLWETransformer(
        vocab_size=vocab_size,
        d_model=256,
        nhead=12,
        num_layers=10,
        dim_feedforward=512,
        max_seq_len=n + 1,
        n_classes=n_classes,
        normalized_attention=True,
        gated=True,
        adaptive_halt=True,
        max_halt_loops=8,
        n_hidden_layers=2,
        **kwargs
    )