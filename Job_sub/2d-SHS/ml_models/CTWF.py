import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange
from netket.nn import log_cosh

# ----------------------------------------------
# 1. Convolutional Embedding：将2D自旋构型嵌入为 Token 序列
# ----------------------------------------------
class ConvEmbedding(nn.Module):
    d_model: int      # 输出通道数（嵌入维度）
    patch_size: int   # patch 尺寸

    def setup(self):
        self.conv = nn.Conv(
            features=self.d_model,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64
        )

    def __call__(self, x):
        # x shape: [batch, L, L, channels]，这里 channels=1
        x = self.conv(x)  # 输出 shape: [batch, L//patch, L//patch, d_model]
        batch, H, W, _ = x.shape
        tokens = x.reshape(batch, H * W, self.d_model)
        return tokens

# ----------------------------------------------
# 2. ConvUnit：利用卷积捕获局部特征
# ----------------------------------------------
class ConvUnit(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, x):
        # 输入 x shape: [batch, n_tokens, d_model]
        batch, n_tokens, d_model = x.shape
        grid = int(np.sqrt(n_tokens))
        x_grid = x.reshape(batch, grid, grid, d_model)
        conv_out = nn.Conv(
            features=self.d_model,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64
        )(x_grid)
        conv_out = nn.gelu(conv_out)
        out = conv_out.reshape(batch, n_tokens, self.d_model)
        return out

# ----------------------------------------------
# 3. CT_MHSA：带相对位置编码的多头自注意力模块
# ----------------------------------------------
class CT_MHSA(nn.Module):
    d_model: int
    h: int         # 注意力头数
    n_tokens: int  # Token 数（应为完全平方数）
    
    def setup(self):
        self.d_head = self.d_model // self.h
        self.WQ = nn.Dense(self.d_model,
                           kernel_init=nn.initializers.xavier_uniform(),
                           param_dtype=jnp.float64,
                           dtype=jnp.float64)
        self.WK = nn.Dense(self.d_model,
                           kernel_init=nn.initializers.xavier_uniform(),
                           param_dtype=jnp.float64,
                           dtype=jnp.float64)
        self.WV = nn.Dense(self.d_model,
                           kernel_init=nn.initializers.xavier_uniform(),
                           param_dtype=jnp.float64,
                           dtype=jnp.float64)
        # Relative positional encoding, shape: [h, n_tokens, n_tokens]
        self.P = self.param("RPE", nn.initializers.xavier_uniform(),
                            (self.h, self.n_tokens, self.n_tokens), jnp.float64)
        self.WO = nn.Dense(self.d_model,
                           kernel_init=nn.initializers.xavier_uniform(),
                           param_dtype=jnp.float64,
                           dtype=jnp.float64)

    def __call__(self, x):
        batch, n, _ = x.shape
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)
        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.h)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.h)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.h)
        scale = np.sqrt(self.d_head)
        attn_scores = jnp.einsum('bhid,bhjd->bhij', Q, K) / scale
        attn_scores = attn_scores + self.P
        attn = nn.softmax(attn_scores, axis=-1)
        attn_out = jnp.einsum('bhij,bhjd->bhid', attn, V)
        attn_out = rearrange(attn_out, 'b h n d -> b n (h d)')
        out = self.WO(attn_out)
        return out

# ----------------------------------------------
# 4. IRFFN：Inverted Residual Feed-Forward Network
# ----------------------------------------------
class IRFFN(nn.Module):
    d_model: int
    expansion_factor: int = 2

    @nn.compact
    def __call__(self, x):
        batch, n_tokens, _ = x.shape
        expanded_dim = self.expansion_factor * self.d_model
        hidden = nn.Dense(expanded_dim,
                          kernel_init=nn.initializers.xavier_uniform(),
                          param_dtype=jnp.float64,
                          dtype=jnp.float64)(x)
        hidden = nn.gelu(hidden)
        grid = int(np.sqrt(n_tokens))
        hidden = hidden.reshape(batch, grid, grid, expanded_dim)
        hidden = nn.Conv(
            features=expanded_dim,
            kernel_size=(3, 3),
            padding="SAME",
            feature_group_count=expanded_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64
        )(hidden)
        hidden = nn.gelu(hidden)
        hidden = hidden.reshape(batch, n_tokens, expanded_dim)
        out = nn.Dense(self.d_model,
                       kernel_init=nn.initializers.xavier_uniform(),
                       param_dtype=jnp.float64,
                       dtype=jnp.float64)(hidden)
        return out

# ----------------------------------------------
# 5. FBlock：RevBlock 内部分支1（卷积单元 + CT_MHSA）
# ----------------------------------------------
class FBlock(nn.Module):
    d_model: int    # 该分支通道数，应为整体 d_model/2
    h: int
    n_tokens: int

    @nn.compact
    def __call__(self, x):
        h1 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)(x)
        h1 = ConvUnit(self.d_model)(h1)
        h1 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)(h1)
        h1 = CT_MHSA(self.d_model, self.h, self.n_tokens)(h1)
        return h1

# ----------------------------------------------
# 6. GBlock：RevBlock 内部分支2（IRFFN 模块）
# ----------------------------------------------
class GBlock(nn.Module):
    d_model: int  # 该分支通道数，应为整体 d_model/2

    @nn.compact
    def __call__(self, x):
        h2 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)(x)
        h2 = IRFFN(self.d_model)(h2)
        return h2

# ----------------------------------------------
# 7. RevBlock：可逆残差块，将输入分通道更新
# ----------------------------------------------
class RevBlock(nn.Module):
    d_model: int  # overall d_model（必须为偶数）
    h: int
    n_tokens: int

    def setup(self):
        # 内部分支各处理 d_model/2 通道
        self.f = FBlock(d_model=self.d_model // 2, h=self.h, n_tokens=self.n_tokens)
        self.g = GBlock(d_model=self.d_model // 2)

    def __call__(self, x):
        # x: [batch, n_tokens, d_model]
        x1, x2 = jnp.split(x, 2, axis=-1)
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return jnp.concatenate([y1, y2], axis=-1)

# ----------------------------------------------
# 8. OutputHead：汇聚 Token 后输出复数波函数振幅
# ----------------------------------------------
class OutputHead(nn.Module):
    d_model: int

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)
        self.norm0 = nn.LayerNorm(use_scale=True, use_bias=True,
                                  dtype=jnp.float64, param_dtype=jnp.float64)
        self.norm1 = nn.LayerNorm(use_scale=True, use_bias=True,
                                  dtype=jnp.float64, param_dtype=jnp.float64)
        self.output_layer0 = nn.Dense(self.d_model,
                                      kernel_init=nn.initializers.xavier_uniform(),
                                      bias_init=nn.initializers.zeros,
                                      param_dtype=jnp.float64,
                                      dtype=jnp.float64)
        self.output_layer1 = nn.Dense(self.d_model,
                                      kernel_init=nn.initializers.xavier_uniform(),
                                      bias_init=nn.initializers.zeros,
                                      param_dtype=jnp.float64,
                                      dtype=jnp.float64)

    def __call__(self, x):
        z = self.out_layer_norm(x.sum(axis=1))
        amp = self.norm0(self.output_layer0(z))
        sign = self.norm1(self.output_layer1(z))
        out = amp + 1j * sign
        return jnp.sum(log_cosh(out), axis=-1)

# ----------------------------------------------
# 9. OptimalNQS：最终综合设计的架构
# ----------------------------------------------
class CTWFNQS(nn.Module):
    num_layers: int     # RevBlock 层数（例如4～8层）
    d_model: int        # 总通道数（须为偶数，建议32～48）
    heads: int          # 注意力头数（例如4～8）
    n_sites: int        # 总格点数，例如 L x L，n_sites=L^2
    patch_size: int     # patch 尺寸

    def setup(self):
        # 系统边长 L
        self.L = int(np.sqrt(self.n_sites))
        # 每边 token 数 = L // patch_size，token 总数
        self.n_tokens_side = self.L // self.patch_size
        self.n_tokens = self.n_tokens_side * self.n_tokens_side
        self.embedding = ConvEmbedding(d_model=self.d_model, patch_size=self.patch_size)
        # 堆叠多个可逆残差块
        self.rev_blocks = [RevBlock(d_model=self.d_model, h=self.heads, n_tokens=self.n_tokens)
                           for _ in range(self.num_layers)]
        self.output = OutputHead(d_model=self.d_model)

    def __call__(self, spins):
        # spins: [batch, n_sites]，一维数组后 reshape 成 [batch, L, L, 1]
        x = jnp.atleast_2d(spins)
        batch = x.shape[0]
        x = x.reshape(batch, self.L, self.L, 1)
        x = self.embedding(x)  # [batch, n_tokens, d_model]
        for block in self.rev_blocks:
            x = block(x)
        out = self.output(x)
        return out

