import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange
from netket.nn import log_cosh

# ---------------------------
# Convolutional Unit
# ---------------------------
class ConvUnit(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, x):
        # x shape: [batch, n_tokens, d_model]
        batch, n_tokens, d_model = x.shape
        grid = int(np.sqrt(n_tokens))
        x_grid = x.reshape(batch, grid, grid, d_model)
        conv_out = nn.Conv(features=self.d_model,
                           kernel_size=(3, 3),
                           padding="SAME",
                           kernel_init=nn.initializers.xavier_uniform(),
                           param_dtype=jnp.float64,
                           dtype=jnp.float64)(x_grid)
        conv_out = nn.gelu(conv_out)
        out = conv_out.reshape(batch, n_tokens, d_model)
        return out

# ---------------------------
# Multi-Head Self-Attention with Relative Position Encoding
# ---------------------------
class CT_MHSA(nn.Module):
    d_model: int
    h: int
    n_tokens: int  # Number of tokens (should be a square number for 2D grid)
    
    def setup(self):
        self.d_head = self.d_model // self.h
        # Linear projections for Q, K, V.
        self.WQ = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                           param_dtype=jnp.float64, dtype=jnp.float64)
        self.WK = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                           param_dtype=jnp.float64, dtype=jnp.float64)
        self.WV = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                           param_dtype=jnp.float64, dtype=jnp.float64)
        # Relative positional encoding parameter: shape [h, n_tokens, n_tokens]
        self.P = self.param("RPE", nn.initializers.xavier_uniform(), 
                            (self.h, self.n_tokens, self.n_tokens), jnp.float64)
        self.WO = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                           param_dtype=jnp.float64, dtype=jnp.float64)

    def __call__(self, x):
        batch, n, _ = x.shape

        Q = self.WQ(x)  # [batch, n, d_model]
        K = self.WK(x)
        V = self.WV(x)
        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.h)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.h)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.h)
        scale = np.sqrt(self.d_head)
        attn_scores = jnp.einsum('bhid,bhjd->bhij', Q, K)  # [batch, h, n, n]
        attn_scores = attn_scores / scale + self.P  # 添加RPE，保持平移不变性
        attn = nn.softmax(attn_scores, axis=-1)
        attn_out = jnp.einsum('bhij,bhjd->bhid', attn, V)
        attn_out = rearrange(attn_out, 'b h n d -> b n (h d)')
        out = self.WO(attn_out)
        return out

# ---------------------------
# Inverted Residual Feed-Forward Network (IRFFN)
# ---------------------------
class IRFFN(nn.Module):
    d_model: int
    expansion_factor: int = 2  # 可调扩展因子

    @nn.compact
    def __call__(self, x):
        # x: [batch, n_tokens, d_model]
        batch, n_tokens, d_model = x.shape
        grid = int(np.sqrt(n_tokens))
        expanded_dim = self.expansion_factor * self.d_model
        
        # 线性投影，将维度扩展为 expansion_factor * d_model
        hidden = nn.Dense(expanded_dim, 
                          kernel_init=nn.initializers.xavier_uniform(),
                          param_dtype=jnp.float64, 
                          dtype=jnp.float64)(x)
        hidden = nn.gelu(hidden)
        
        # 调整为2维空间的形状：grid x grid
        hidden = hidden.reshape(batch, grid, grid, expanded_dim)
        
        # 使用group convolution实现depthwise convolution, groups=channels
        hidden = nn.Conv(features=expanded_dim,
                         kernel_size=(3, 3),
                         padding="SAME",
                         feature_group_count=expanded_dim,
                         kernel_init=nn.initializers.xavier_uniform(),
                         param_dtype=jnp.float64,
                         dtype=jnp.float64)(hidden)
        hidden = nn.gelu(hidden)
        
        # 恢复为原token维度：[batch, n_tokens, expanded_dim]
        hidden = hidden.reshape(batch, n_tokens, expanded_dim)
        
        # 线性映射回 d_model 维度
        out = nn.Dense(self.d_model, 
                       kernel_init=nn.initializers.xavier_uniform(),
                       param_dtype=jnp.float64, 
                       dtype=jnp.float64)(hidden)
        return out



# ---------------------------
# CTWF Encoder Block
# ---------------------------
class EncoderCTWF(nn.Module):
    d_model: int
    h: int
    n_tokens: int

    @nn.compact
    def __call__(self, x):
        # 1. Convolutional Unit with LayerNorm and residual connection
        norm1 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)(x)
        conv_out = ConvUnit(self.d_model)(norm1)
        x = x + conv_out

        # 2. Multi-Head Self-Attention with LayerNorm and residual connection
        norm2 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)(x)
        attn_out = CT_MHSA(self.d_model, self.h, self.n_tokens)(norm2)
        x = x + attn_out

        # 3. Inverted Residual Feed-Forward Network with LayerNorm and residual connection
        norm3 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)(x)
        ffn_out = IRFFN(self.d_model)(norm3)
        x = x + ffn_out

        return x


# ---------------------------
# Output Head
# ---------------------------
class OutputHead(nn.Module):
    d_model: int

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)
        self.norm0 = nn.LayerNorm(use_scale=True, use_bias=True,
                                  dtype=jnp.float64, param_dtype=jnp.float64)
        self.norm1 = nn.LayerNorm(use_scale=True, use_bias=True,
                                  dtype=jnp.float64, param_dtype=jnp.float64)
        self.output_layer0 = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                                      bias_init=jax.nn.initializers.zeros,
                                      param_dtype=jnp.float64, dtype=jnp.float64)
        self.output_layer1 = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                                      bias_init=jax.nn.initializers.zeros,
                                      param_dtype=jnp.float64, dtype=jnp.float64)

    def __call__(self, x):
        # 汇聚token的信息，进行归一化处理后得到振幅和符号部分，并使用 log_cosh 激活
        z = self.out_layer_norm(x.sum(axis=1))
        amp = self.norm0(self.output_layer0(z))
        sign = self.norm1(self.output_layer1(z))
        out = amp + 1j * sign
        return jnp.sum(log_cosh(out), axis=-1)

# ---------------------------
# Overall CTWF Network with Convolutional Embed
# ---------------------------
class CTWFNQS(nn.Module):
    num_layers: int
    d_model: int
    heads: int
    n_sites: int         # 总格点数，例如 L x L
    patch_size: int      # Patch 尺寸

    def setup(self):
        # 输入为一维展开的格点数据，重构为二维时的边长
        self.L = int(np.sqrt(self.n_sites))
        # 经过卷积嵌入后，输出尺度为 (L/patch_size) x (L/patch_size)
        self.n_tokens_side = self.L // self.patch_size
        self.n_tokens = self.n_tokens_side * self.n_tokens_side
        # Embedding using convolution: kernel_size 和 strides 均为 patch_size
        self.embed = nn.Conv(features=self.d_model,
                             kernel_size=(self.patch_size, self.patch_size),
                             strides=(self.patch_size, self.patch_size),
                             padding="VALID",
                             kernel_init=nn.initializers.xavier_uniform(),
                             param_dtype=jnp.float64,
                             dtype=jnp.float64)
        # 堆叠多层 CTWF encoder blocks
        self.blocks = [EncoderCTWF(d_model=self.d_model,
                                     h=self.heads,
                                     n_tokens=self.n_tokens)
                       for _ in range(self.num_layers)]
        self.output = OutputHead(self.d_model)

    def __call__(self, spins):
        # spins shape: [batch, n_sites]，其中 n_sites = L^2
        x = jnp.atleast_2d(spins)
        batch = x.shape[0]
        # 重构为二维格点，并增加 channel 维度（例如1）
        x = x.reshape(batch, self.L, self.L, 1)
        # 进行卷积嵌入，输出 shape: [batch, n_tokens_side, n_tokens_side, d_model]
        x = self.embed(x)
        # Flatten成 token 序列: [batch, n_tokens, d_model]
        x = x.reshape(batch, self.n_tokens, self.d_model)
        # 依次通过 encoder block
        for block in self.blocks:
            x = block(x)
        # 利用 output head 汇聚信息输出标量（波函数对数幅值）
        out = self.output(x)
        return out
