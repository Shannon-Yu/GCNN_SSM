"""
量子态生成和管理模块，提供创建和重建量子态的功能。
"""

import pickle
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
from netket.utils.group import PointGroup, Identity
from netket.utils.group.planar import rotation, glide_group

from configs.config import ModelConfig, TrainingConfig, SystemConfig
from src.physics.shastry_sutherland import (
    shastry_sutherland_lattice, 
    shastry_sutherland_hamiltonian, 
    shastry_sutherland_all_symmetries
)

def create_quantum_state(L, J2, J1, n_samples=None, n_discard=None, chunk_size=None):
    """
    创建Shastry-Sutherland模型的量子态
    
    参数:
    L: 晶格大小
    J2: J2耦合强度
    J1: J1耦合强度
    n_samples: 采样数量，默认使用TrainingConfig.N_samples
    n_discard: 丢弃的样本数，默认使用TrainingConfig.N_discard
    chunk_size: 批处理大小，默认使用TrainingConfig.chunk_size
    
    返回:
    vqs: 变分量子态
    lattice: 晶格
    hilbert: 希尔伯特空间
    hamiltonian: 哈密顿量
    """
    # 使用默认值
    if n_samples is None:
        n_samples = TrainingConfig.N_samples
    if n_discard is None:
        n_discard = TrainingConfig.N_discard
    if chunk_size is None:
        chunk_size = TrainingConfig.chunk_size
    
    # 计算Q参数
    Q = 1.00 - J2
    
    # 创建晶格和哈密顿量
    lattice = shastry_sutherland_lattice(L, L)
    hamiltonian, hilbert = shastry_sutherland_hamiltonian(lattice, J1, J2, Q)
    
    # 创建采样器
    sampler = nk.sampler.MetropolisExchange(
        hilbert=hilbert, 
        graph=lattice, 
        n_chains=n_samples, 
        d_max=2
    )
    
    # 定义局部掩码
    mask = jnp.zeros(lattice.n_nodes, dtype=bool)
    for i in range(lattice.n_nodes):
        mask = mask.at[i].set(True)
    
    # 获取对称性
    symmetries = shastry_sutherland_all_symmetries(lattice)
    
    # 设置动量和表示
    nc = 4
    cyclic_4 = PointGroup(
        [Identity()] + [rotation((360 / nc) * i) for i in range(1, nc)],
        ndim=2,
    )
    C4v = glide_group(trans=(1, 1), origin=(0, 0)) @ cyclic_4
    sgb = lattice.space_group_builder(point_group=C4v)
    momentum = [0.0, 0.0]
    chi = sgb.space_group_irreps(momentum)[0]
    
    # 创建GCNN模型
    model = nk.models.GCNN(
        symmetries=symmetries,
        layers=ModelConfig.num_layers,
        param_dtype=jnp.complex128,
        features=ModelConfig.num_features,
        equal_amplitudes=False,
        parity=1,
        input_mask=mask,
        characters=chi
    )
    
    # 创建变分量子态
    vqs = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard,
        chunk_size=chunk_size,
    )
    
    return vqs, lattice, hilbert, hamiltonian

def save_quantum_state(vqs, file_path):
    """
    保存量子态参数到文件
    
    参数:
    vqs: 变分量子态
    file_path: 保存路径
    """
    with open(file_path, "wb") as f:
        pickle.dump(vqs.parameters, f)

def load_quantum_state(file_path, L, J2, J1, n_samples=2**14, n_discard=50, chunk_size=2**10):
    """
    从文件加载量子态参数
    
    参数:
    file_path: 参数文件路径
    L: 晶格大小
    J2: J2耦合强度
    J1: J1耦合强度
    n_samples: 采样数量
    n_discard: 丢弃的样本数
    chunk_size: 批处理大小
    
    返回:
    vqs: 变分量子态
    lattice: 晶格
    hilbert: 希尔伯特空间
    hamiltonian: 哈密顿量
    """
    # 创建量子态
    vqs, lattice, hilbert, hamiltonian = create_quantum_state(
        L, J2, J1, 
        n_samples=n_samples, 
        n_discard=n_discard, 
        chunk_size=chunk_size
    )
    
    # 加载参数
    with open(file_path, "rb") as f:
        parameters = pickle.load(f)
    
    # 设置参数
    vqs.parameters = parameters
    
    return vqs, lattice, hilbert, hamiltonian
