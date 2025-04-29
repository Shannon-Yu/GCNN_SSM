import os
import datetime
import pytz
import jax
import jax.numpy as jnp
from functools import partial
from jax import tree
import netket as nk
from netket.operator import AbstractOperator
from netket.vqs import VariationalState

def T_logp2(params, model, inputs, temperature):
    """计算熵梯度T_logp2，直接接收model作为参数"""
    variables = {"params": params}
    preds = model.apply(variables, inputs)
    return 2.0 * temperature * jnp.mean(jnp.real(preds) * jnp.real(preds))

def T_logp_2(params, model, inputs, temperature):
    """计算熵梯度T_logp_2，直接接收model作为参数"""
    variables = {"params": params}
    preds = model.apply(variables, inputs)
    return 2.0 * temperature * jnp.mean(jnp.real(preds)) * jnp.mean(jnp.real(preds))

def custom_sr_free_energy(
    hamiltonian: AbstractOperator,    
    vstate: VariationalState,                                            
    lr: float,                                         
    temperature: float,
    n_ann: int,
    n_train: int,
    energy_log: str
):
    """自定义SR自由能优化函数"""
    
    # 获取模型引用
    model = vstate.model
    
    
    for i in range(n_ann):
        # 计算当前温度
        temperature_i = temperature * (jnp.exp(-i / 50.0) / 2.0)
        
        # 内层训练循环
        for j in range(n_train):
            # 计算能量和梯度
            energy, f = vstate.expect_and_grad(hamiltonian)
            variables = vstate.variables 
            inputs0 = vstate.samples
            
            # 处理样本形状
            sample_shape = inputs0.shape
            if len(sample_shape) == 3:  # 如果形状是(n_chains, batch_size, n_nodes)
                inputs = inputs0
            else:
                log_message(energy_log, f"警告: 意外的样本形状 {sample_shape}，可能需要调整处理逻辑")
                inputs = inputs0
            
            # 每隔一定步数记录当前能量
            log_message(energy_log, 
                        f"Iter {i}/{n_ann}, T={temperature_i:.4f}, "
                        f"Energy = {energy.mean.real:.6f} ± {energy.error_of_mean:.6f}, "
                        f"Variance = {energy.variance:.6f}")
            
            # 计算量子几何张量和梯度
            G = vstate.quantum_geometric_tensor(nk.optimizer.qgt.QGTJacobianDense(diag_shift=0.001, diag_scale=0.001, chunk_size=vstate.chunk_size//4))
            
            # 直接将model作为参数传递给梯度函数
            mT_grad_S_1 = jax.grad(T_logp2, argnums=0)(variables["params"], model, inputs, temperature_i)
            mT_grad_S_2 = jax.grad(T_logp_2, argnums=0)(variables["params"], model, inputs, temperature_i)
            
            mT_grad_S = tree.map(lambda x, y: x - y, mT_grad_S_1, mT_grad_S_2)
            gamma_S = tree.map(lambda x: -1.0 * jnp.conj(x), mT_grad_S)
            gamma_f = tree.map(lambda x: -1.0 * x, f)
            gamma_tot = tree.map(lambda x, y: x + y, gamma_f, gamma_S)
            
            # 求解方程并更新参数
            dtheta, _ = G.solve(jax.scipy.sparse.linalg.cg, gamma_tot)
            vstate.parameters = tree.map(lambda x, y: x + lr * y, vstate.parameters, dtheta)