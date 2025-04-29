import jax
import jax.numpy as jnp
import flax.linen as nn
import math
from einops import rearrange
from netket.nn import log_cosh
from jax._src import dtypes

# 位置编码
class PositionalEncoding(nn.Module):
    """位置编码模块"""
    d_model: int
    max_len: int = 5000
    param_dtype: jnp.dtype = jnp.float64  # 改为实值

    def setup(self):
        # 创建位置编码矩阵
        position = jnp.arange(self.max_len)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = jnp.zeros((self.max_len, self.d_model), dtype=self.param_dtype)
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe

    def __call__(self, x):
        """
        参数:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:x.shape[1]]
        return x

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    d_model: int
    heads: int
    param_dtype: jnp.dtype = jnp.float64  # 改为实值

    def setup(self):
        self.head_dim = self.d_model // self.heads
        self.wq = nn.Dense(self.d_model, param_dtype=self.param_dtype, dtype=self.param_dtype)
        self.wk = nn.Dense(self.d_model, param_dtype=self.param_dtype, dtype=self.param_dtype)
        self.wv = nn.Dense(self.d_model, param_dtype=self.param_dtype, dtype=self.param_dtype)
        self.wo = nn.Dense(self.d_model, param_dtype=self.param_dtype, dtype=self.param_dtype)

    def __call__(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # 重塑为多头格式
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.heads)
        
        # 计算注意力分数
        attn_scores = jnp.einsum('bhid,bhjd->bhij', q, k) / jnp.sqrt(self.head_dim)
        attention = nn.softmax(attn_scores, axis=-1)
        
        # 应用注意力权重
        out = jnp.einsum('bhij,bhjd->bhid', attention, v)
        out = rearrange(out, 'b h s d -> b s (h d)')
        
        return self.wo(out)

# 前馈神经网络
class MLP(nn.Module):
    """前馈神经网络"""
    d_model: int
    mlp_dim: int
    param_dtype: jnp.dtype = jnp.float64  # 改为实值

    def setup(self):
        self.fc1 = nn.Dense(self.mlp_dim, param_dtype=self.param_dtype, dtype=self.param_dtype)
        self.fc2 = nn.Dense(self.d_model, param_dtype=self.param_dtype, dtype=self.param_dtype)
        self.gelu = lambda x: x * 0.5 * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))

    def __call__(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

# Transformer编码器块
class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    d_model: int
    heads: int
    mlp_dim: int
    param_dtype: jnp.dtype = jnp.float64  # 改为实值

    def setup(self):
        self.attn = MultiHeadAttention(d_model=self.d_model, heads=self.heads, param_dtype=self.param_dtype)
        self.norm1 = nn.LayerNorm(dtype=self.param_dtype, param_dtype=self.param_dtype)
        self.norm2 = nn.LayerNorm(dtype=self.param_dtype, param_dtype=self.param_dtype)
        self.mlp = MLP(d_model=self.d_model, mlp_dim=self.mlp_dim, param_dtype=self.param_dtype)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# Transformer编码器
class Encoder(nn.Module):
    """Transformer编码器"""
    num_layers: int
    d_model: int
    heads: int
    mlp_dim: int
    param_dtype: jnp.dtype = jnp.float64  # 改为实值

    def setup(self):
        self.layers = [
            EncoderBlock(
                d_model=self.d_model,
                heads=self.heads,
                mlp_dim=self.mlp_dim,
                param_dtype=self.param_dtype
            ) 
            for _ in range(self.num_layers)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 输出头
class OutputHead(nn.Module):
    d_model: int
    param_dtype: jnp.dtype = jnp.complex128  # 保持复值
    complex: bool = True

    def setup(self):
        # 实值输入的规范化层
        self.norm = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)
        
        # 实值到复值的转换层
        self.real_to_complex = nn.Dense(
            self.d_model, 
            param_dtype=jnp.complex128,  # 复值参数
            dtype=jnp.complex128,
            kernel_init=nn.initializers.xavier_uniform()
        )
        
        # 复值输出层
        self.norm0 = nn.LayerNorm(use_scale=True, use_bias=True, 
                                 dtype=jnp.complex128, param_dtype=jnp.complex128)
        self.norm1 = nn.LayerNorm(use_scale=True, use_bias=True, 
                                 dtype=jnp.complex128, param_dtype=jnp.complex128)
        
        self.output_layer0 = nn.Dense(
            self.d_model, 
            param_dtype=jnp.complex128, 
            dtype=jnp.complex128,
            kernel_init=nn.initializers.xavier_uniform()
        )
        
        self.output_layer1 = nn.Dense(
            self.d_model, 
            param_dtype=jnp.complex128, 
            dtype=jnp.complex128,
            kernel_init=nn.initializers.xavier_uniform()
        )

    def __call__(self, x, return_z=False):
        # 取[CLS]位置的实值特征向量
        z = self.norm(x[:, 0])
        
        if return_z:
            return z
        
        # 从实值转换为复值
        z_complex = self.real_to_complex(jnp.asarray(z, dtype=jnp.complex128))
        
        # 复值输出处理
        amp = self.norm0(self.output_layer0(z_complex))
        
        if self.complex:
            sign = self.norm1(self.output_layer1(z_complex))
            out = amp + 1j * sign
        else:
            out = amp
        
        return jnp.sum(log_cosh(out), axis=-1)

# ViT量子态模型
class ViTFNQS(nn.Module):
    num_layers: int
    d_model: int
    heads: int
    mlp_dim: int
    patch_size: int
    n_sites: int
    param_dtype: jnp.dtype = jnp.float64  # 改为实值默认
    complex: bool = True

    def setup(self):
        # CLS标记 (实值)
        self.cls_token = self.param(
            'cls_token', 
            nn.initializers.zeros, 
            (1, 1, self.d_model),
            self.param_dtype
        )
        
        # Patch嵌入层 (实值)
        self.patch_embed = nn.Dense(
            self.d_model, 
            param_dtype=self.param_dtype, 
            dtype=self.param_dtype,
            kernel_init=nn.initializers.xavier_uniform()
        )
        
        # 位置编码 (实值)
        self.pos_embedding = PositionalEncoding(d_model=self.d_model, param_dtype=self.param_dtype)
        
        # Transformer编码器 (实值)
        self.encoder = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
            param_dtype=self.param_dtype
        )
        
        # 输出头 (复值) - 这是唯一使用复值参数的部分
        self.output = OutputHead(
            self.d_model,
            param_dtype=jnp.complex128,  # 明确使用复值
            complex=self.complex
        )
    
    def extract_patches(self, x, batch_size=None):
        """从输入中提取patch"""
        if batch_size is None:
            batch_size = x.shape[0]
            
        # 重塑为patch
        if self.patch_size > 1:
            # 将邻居聚集为patch
            x = rearrange(
                x, 
                'b (n p) -> b n p', 
                p=self.patch_size
            )
        else:
            # 如果patch_size=1，每个site作为一个独立patch
            x = x.reshape(batch_size, self.n_sites, 1)
            
        return x

    def __call__(self, x, return_z=False):
        batch_size = x.shape[0]
        
        # 提取patch
        x = self.extract_patches(x, batch_size)
        
        # 嵌入patch
        x = self.patch_embed(x)
        
        # 添加CLS标记
        cls_tokens = jnp.repeat(self.cls_token, batch_size, axis=0)
        x = jnp.concatenate([cls_tokens, x], axis=1)
        
        # 添加位置编码
        x = self.pos_embedding(x)
        
        # 通过编码器
        x = self.encoder(x)
        
        # 通过输出头
        out = self.output(x, return_z=return_z)
        
        return out