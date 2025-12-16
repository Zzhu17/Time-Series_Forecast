import torch
import torch.nn as nn
import math
import configs
from typing import Optional, Tuple, List
# ======== DataEmbedding: 多变量/多步输入兼容 ========
class DataEmbedding(nn.Module):
    def __init__(self, input_dim:int , d_model: int, dropout:float=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(input_dim, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, f"DataEmbedding 期望 3D (batch, seq, input_dim), got {x.shape}"
        # x: (batch, seq, input_dim) —— input_dim可为1（单变量）或n（多变量）
        x = self.value_embedding(x)  # (batch, seq, d_model)
        x = x + self.position_embedding(x)
        return self.dropout(x)

# ======== PositionalEmbedding: 不变 ========
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model:int, max_len:int=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model) — we want positional encoding shaped (1, seq, d_model)
        seq_len = x.size(1)
        # stored pe shape: (max_len, 1, d_model); slice first by seq_len then transpose to (1, seq, d_model)
        return self.pe[:seq_len].transpose(0, 1)

# ======== FeedForward: 标准前馈网络 ========
class FeedForward(nn.Module):
    def __init__(self, d_model:int , d_ff:int , dropout:float=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
    
# ======== Attention Layer (Full & ProbSparse) ========"    
class FullAttention(nn.Module):
    """
    标准的多头自注意力机制 (作为对比)。
    """
    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1. / math.sqrt(self.d_k)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = q.shape
        _, S, _ = k.shape
        H = self.n_heads
        q = self.q_proj(q).view(B, L, H, -1).transpose(1, 2)
        k = self.k_proj(k).view(B, S, H, -1).transpose(1, 2)
        v = self.v_proj(v).view(B, S, H, -1).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(context)

# ======== ProbSparse Attention Layer ========
class ProbAttention(nn.Module):
    """
    ProbSparse Self-Attention Mechanism.
    """
    def __init__(self, n_heads: int, d_model: int, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1. / math.sqrt(self.d_k)

    def _prob_QK(self, Q: torch.Tensor, K: torch.Tensor, sample_k: int, n_top: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        核心采样逻辑
        Args:
            Q (torch.Tensor): (B, H, L_q, D)
            K (torch.Tensor): (B, H, L_k, D)
            sample_k (int): 每轮随机采样的key数量，用于计算稀疏度
            n_top (int): 最终要选择的top-k个key
        Returns:
            Tuple: (top_k_scores, top_k_indices)
        """
        B, H, L_K, _ = K.shape
        _, _, L_Q, _ = Q.shape

        # 1. 随机采样 K，用于计算稀疏度得分
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, self.d_k)
        
        # 随机选择 sample_k 个 key 的索引
        index_sample = torch.randint(L_K, (L_Q, sample_k)) 
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        
        # 2. 计算采样后的 QK 得分
        Q_reshaped = Q.unsqueeze(-2)
        scores_sample = torch.matmul(Q_reshaped, K_sample.transpose(-2, -1)).squeeze(-2)

        # 3. 计算稀疏度度量 M
        # M = max(scores) - mean(scores)
        M = scores_sample.max(-1)[0] - torch.div(scores_sample.sum(-1), L_K)
        
        # 4. 选择稀疏度最高的 n_top 个 Query
        _, top_indices = M.topk(n_top, sorted=False)

        # 5. 使用选出的Query的索引，来获取它们对应的原始K
        # 创建一个 (B, H, n_top, D) 的 Q_reduce 张量
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     top_indices, :]
        
        # 6. 计算 Q_reduce 和所有 K 的注意力分数
        scores_reduced = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return scores_reduced, top_indices

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L_Q, _ = q.shape
        _, L_K, _ = k.shape
        H = self.n_heads

        # 线性投影并分头
        queries = self.q_proj(q).view(B, L_Q, H, -1).transpose(1, 2)
        keys = self.k_proj(k).view(B, L_K, H, -1).transpose(1, 2)
        values = self.v_proj(v).view(B, L_K, H, -1).transpose(1, 2)

        # --- ProbSparse 核心逻辑 ---
        # U_part 是我们要为每个query选择的key的数量
        U_part = self.factor * int(math.ceil(math.log(L_K))) 
        # m_top 是我们要选择的query的数量
        m_top = self.factor * int(math.ceil(math.log(L_Q))) 
        
        # 如果序列长度较短，则退化为全量注意力
        if L_Q * L_K <= U_part * m_top:
            scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        else:
            scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=m_top)
            # 使用稀疏分数更新原始分数矩阵
            scores = torch.zeros(B, H, L_Q, L_K).to(queries.device)
            scores.scatter_(-1, index.unsqueeze(-1).expand(-1, -1, -1, L_K), scores_top)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, values)
        context = context.transpose(1, 2).contiguous().view(B, L_Q, -1)

        return self.out_proj(context)
    
# ======== EncoderLayer: 无 squeeze，参数灵活 ========
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, attn_type: str = 'prob', factor: int = 5):
        super().__init__()
        if attn_type == 'prob':
            self.self_attn = ProbAttention(n_heads=n_heads, d_model=d_model, factor=factor, dropout=dropout)
        else:
            self.self_attn = FullAttention(n_heads=n_heads, d_model=d_model, dropout=dropout)

        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x2 = self.self_attn(x, x, x, mask)
        x = self.norm1(x + x2)
        x2 = self.ff(x)
        x = self.norm2(x + x2)
        return x

# ======== Decoder Layer: 组合两层 Attention 和 FeedForward ========"
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, attn_type: str = 'prob', factor: int = 5):
        super().__init__()
        if attn_type == 'prob':
            self.self_attn = ProbAttention(n_heads, d_model, factor=factor, dropout=dropout)
            self.cross_attn = ProbAttention(n_heads, d_model, factor=factor, dropout=dropout)
        else:
            self.self_attn = FullAttention(n_heads, d_model, dropout=dropout)
            self.cross_attn = FullAttention(n_heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x2 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + x2)
        x2 = self.cross_attn(q=x, k=memory, v=memory, mask=memory_mask)
        x = self.norm2(x + x2)
        x2 = self.ff(x)
        x = self.norm3(x + x2)
        return x

# ======== Encoder: 可堆叠多层 ========
class Encoder(nn.Module):
    def __init__(self, layers: List[nn.Module], norm_layer: Optional[nn.Module] = None, d_model: Optional[int] = None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        # 显式传入 d_model，避免从 EncoderLayer 猜测属性
        if d_model is None:
            raise ValueError("Encoder requires d_model; please pass d_model to Encoder(...).")
        self.norm = nn.LayerNorm(d_model) if norm_layer is None else norm_layer

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# ======== Decoder: 可堆叠多层 ========
class Decoder(nn.Module):
    def __init__(self, layers: List[nn.Module], norm_layer: Optional[nn.Module] = None, d_model: Optional[int] = None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        if d_model is None:
            raise ValueError("Decoder requires d_model; please pass d_model to Decoder(...).")
        self.norm = nn.LayerNorm(d_model) if norm_layer is None else norm_layer

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        return self.norm(x)

# ======== Output Head: 支持多变量输出 ========
class InformerOutputHead(nn.Module):
    def __init__(self, d_model: int, out_features: int):
        super().__init__()
        self.proj = nn.Linear(d_model, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)