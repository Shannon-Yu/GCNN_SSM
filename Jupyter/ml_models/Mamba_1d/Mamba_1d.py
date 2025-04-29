from flax import linen as nn

from typing import Sequence

import jax
import jax.numpy as jnp
from jax._src import dtypes

# 自定义均匀初始化器
def custom_uniform(scale=1e-2, dtype=jnp.float_):
    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        return (2.0 * jax.random.uniform(key, shape, dtype) - 1.0) * scale
    return init

@jax.jit
def log_cosh(x):
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)

@jax.jit
def attention(J, values, shift):
    values = jnp.roll(values, shift, axis=0)
    return jnp.sum(J * values, axis=0)

class EncoderBlock(nn.Module):
    d_model: int
    h: int
    L: int
    b: int

    def setup(self):
        scale = (3.0 * 0.7 / self.L) ** 0.5
        self.v_projR = nn.Dense(self.d_model, param_dtype=jnp.float64,
                                kernel_init=jax.nn.initializers.variance_scaling(0.3, "fan_in", "uniform"),
                                bias_init=nn.initializers.zeros)
        self.v_projI = nn.Dense(self.d_model, param_dtype=jnp.float64,
                                kernel_init=jax.nn.initializers.variance_scaling(0.3, "fan_in", "uniform"),
                                bias_init=nn.initializers.zeros)
        self.JR = self.param("JR", custom_uniform(scale=scale), (self.L, self.h, 1), jnp.float64)
        self.JI = self.param("JI", custom_uniform(scale=scale), (self.L, self.h, 1), jnp.float64)
        self.W0R = nn.Dense(self.d_model, param_dtype=jnp.float64,
                            kernel_init=jax.nn.initializers.variance_scaling(0.065, "fan_in", "uniform"),
                            bias_init=nn.initializers.zeros)
        self.W0I = nn.Dense(self.d_model, param_dtype=jnp.float64,
                            kernel_init=jax.nn.initializers.variance_scaling(0.065, "fan_in", "uniform"),
                            bias_init=nn.initializers.zeros)

    def __call__(self, x):
        J = self.JR + 1j * self.JI
        x = self.v_projR(x).reshape(self.L, self.h, -1) + 1j * self.v_projI(x).reshape(self.L, self.h, -1)
        x = jax.vmap(attention, (None, None, 0))(J, x, jnp.arange(self.L))
        x = x.reshape(self.L, -1)
        x = self.W0R(x) + 1j * self.W0I(x)
        return log_cosh(x)

class Transformer_Enc(nn.Module):
    d_model: int
    h: int
    L: int
    b: int

    def setup(self):
        self.encoder = EncoderBlock(self.d_model, self.h, self.L, self.b)

    def __call__(self, x):
        x = x.reshape(x.shape[0], -1, self.b)
        x = jax.vmap(self.encoder)(x)
        return jnp.sum(x, axis=(1, 2))