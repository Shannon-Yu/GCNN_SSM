import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange
import math
from functools import partial

# 实现roll函数，用于实现平移不变性
def roll(J, shift, axis=-1):
    return jnp.roll(J, shift, axis=axis)

# 实现二维平移
@partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
def roll2d(spins, i, j):
    side = int(spins.shape[-1]**0.5)
    spins = spins.reshape(spins.shape[0], side, side)
    spins = jnp.roll(jnp.roll(spins, i, axis=-2), j, axis=-1)
    return spins.reshape(spins.shape[0], -1)

# 实现提取块函数
def extract_patches1d(x, b):
    return rearrange(x, 'batch (L_eff b) -> batch L_eff b', b=b)

def extract_patches2d(x, b):
    batch = x.shape[0]
    L_eff = int((x.shape[1] // b**2)**0.5)
    x = x.reshape(batch, L_eff, b, L_eff, b)   # [L_eff, b, L_eff, b]
    x = x.transpose(0, 1, 3, 2, 4)         # [L_eff, L_eff, b, b]
    # 压平patches
    x = x.reshape(batch, L_eff, L_eff, -1)     # [L_eff, L_eff, b*b]
    x = x.reshape(batch, L_eff*L_eff, -1)      # [L_eff*L_eff, b*b]
    return x

# 平移不变的多头注意力机制
class FMHA(nn.Module):
    d_model: int
    h: int
    L_eff: int
    transl_invariant: bool = True
    two_dimensional: bool = False

    def setup(self):
        self.v = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=jnp.float64, dtype=jnp.float64)
        if self.transl_invariant:
            self.J = self.param("J", nn.initializers.xavier_uniform(), (self.h, self.L_eff), jnp.float64)
            if self.two_dimensional:
                sq_L_eff = int(self.L_eff**0.5)
                assert sq_L_eff * sq_L_eff == self.L_eff
                self.J = roll2d(self.J, jnp.arange(sq_L_eff), jnp.arange(sq_L_eff))
                self.J = self.J.reshape(self.h, -1, self.L_eff)
            else:
                self.J = jax.vmap(roll, (None, 0), out_axes=1)(self.J, jnp.arange(self.L_eff))
        else:
            self.J = self.param("J", nn.initializers.xavier_uniform(), (self.h, self.L_eff, self.L_eff), jnp.float64)

        self.W = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=jnp.float64, dtype=jnp.float64)

    def __call__(self, x):
        v = self.v(x)
        v = rearrange(v, 'batch L_eff (h d_eff) -> batch L_eff h d_eff', h=self.h)
        v = rearrange(v, 'batch L_eff h d_eff -> batch h L_eff d_eff')
        x = jnp.matmul(self.J, v)
        x = rearrange(x, 'batch h L_eff d_eff  -> batch L_eff h d_eff')
        x = rearrange(x, 'batch L_eff h d_eff ->  batch L_eff (h d_eff)')

        x = self.W(x)
        return x

# 实现log_cosh激活函数
def log_cosh(x):
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)

# Patch嵌入层
class Embed(nn.Module):
    d_model: int
    b: int
    two_dimensional: bool = False

    def setup(self):
        if self.two_dimensional:
            self.extract_patches = extract_patches2d
        else:
            self.extract_patches = extract_patches1d

        self.embed = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), 
                             param_dtype=jnp.float64, dtype=jnp.float64)

    def __call__(self, x):
        x = self.extract_patches(x, self.b)
        x = self.embed(x)
        return x

# 平移不变的编码器块
class EncoderBlock(nn.Module):
    d_model: int
    h: int
    L_eff: int
    transl_invariant: bool = True
    two_dimensional: bool = False

    def setup(self):
        self.attn = FMHA(d_model=self.d_model, h=self.h, L_eff=self.L_eff, 
                         transl_invariant=self.transl_invariant, two_dimensional=self.two_dimensional)
        
        self.layer_norm_1 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)
        self.layer_norm_2 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)

        self.ff = nn.Sequential([
            nn.Dense(4*self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=jnp.float64, dtype=jnp.float64),
            nn.gelu,
            nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=jnp.float64, dtype=jnp.float64),
        ])

    def __call__(self, x):
        x = x + self.attn(self.layer_norm_1(x))
        x = x + self.ff(self.layer_norm_2(x))
        return x

# 平移不变的编码器
class Encoder(nn.Module):
    num_layers: int
    d_model: int
    h: int
    L_eff: int
    transl_invariant: bool = True
    two_dimensional: bool = False

    def setup(self):
        self.layers = [EncoderBlock(d_model=self.d_model, h=self.h, L_eff=self.L_eff, 
                                   transl_invariant=self.transl_invariant, 
                                   two_dimensional=self.two_dimensional) 
                      for _ in range(self.num_layers)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 输出头
class OuputHead(nn.Module):
    d_model: int
    complex: bool = False

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)

        self.norm0 = nn.LayerNorm(use_scale=True, use_bias=True, dtype=jnp.float64, param_dtype=jnp.float64)
        self.norm1 = nn.LayerNorm(use_scale=True, use_bias=True, dtype=jnp.float64, param_dtype=jnp.float64)

        self.output_layer0 = nn.Dense(self.d_model, param_dtype=jnp.float64, dtype=jnp.float64, 
                                     kernel_init=nn.initializers.xavier_uniform(), 
                                     bias_init=jax.nn.initializers.zeros)
        self.output_layer1 = nn.Dense(self.d_model, param_dtype=jnp.float64, dtype=jnp.float64, 
                                     kernel_init=nn.initializers.xavier_uniform(), 
                                     bias_init=jax.nn.initializers.zeros)

    def __call__(self, x, return_z=False):
        z = self.out_layer_norm(x.sum(axis=1))

        if return_z:
            return z
        
        amp = self.norm0(self.output_layer0(z))
        
        if self.complex:
            sign = self.norm1(self.output_layer1(z))
            out = amp + 1j*sign
        else:
            out = amp

        return jnp.sum(log_cosh(out), axis=-1)

# 完整的ViT FNQS模型
class ViTFNQS(nn.Module):
    num_layers: int
    d_model: int
    heads: int
    L_eff: int
    b: int
    complex: bool = False
    disorder: bool = False
    transl_invariant: bool = True
    two_dimensional: bool = False

    def setup(self):
        if self.disorder:
            self.patches_and_embed = Embed(self.d_model//2, self.b, two_dimensional=self.two_dimensional)
            self.patches_and_embed_coup = Embed(self.d_model//2, self.b, two_dimensional=self.two_dimensional)
        else:
            self.embed = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), 
                                 param_dtype=jnp.float64, dtype=jnp.float64)

        self.encoder = Encoder(num_layers=self.num_layers, d_model=self.d_model, h=self.heads, 
                              L_eff=self.L_eff, transl_invariant=self.transl_invariant, 
                              two_dimensional=self.two_dimensional)

        self.output = OuputHead(self.d_model, complex=self.complex)

    def __call__(self, spins, coups, return_z=False):
        x = jnp.atleast_2d(spins)

        if self.disorder:
            x_spins = self.patches_and_embed(x)
            x_coups = self.patches_and_embed(coups)
            x = jnp.concatenate((x_spins, x_coups), axis=-1)
        else:
            if self.two_dimensional:
                x = extract_patches2d(x, self.b)
            else:
                x = extract_patches1d(x, self.b)
            coups = jnp.broadcast_to(coups, (x.shape[0], x.shape[1], 2))

            x = jnp.concatenate((x, coups), axis=-1)
            x = self.embed(x)

        x = self.encoder(x)
        out = self.output(x, return_z=return_z)

        return out
