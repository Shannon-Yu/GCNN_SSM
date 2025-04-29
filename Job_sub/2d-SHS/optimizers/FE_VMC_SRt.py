import time
import jax
import jax.numpy as jnp
from jax import tree_util
from netket.experimental.driver.vmc_srt import VMC_SRt
import datetime
import pytz


def log_message(log_file, message):
    """记录带时间戳的消息"""
    tz = pytz.timezone("Asia/Shanghai")
    timestamp = datetime.datetime.now(tz).strftime("[%Y-%m-%d %H:%M:%S]")
    log_line = f"{timestamp} {message}"
    with open(log_file, "a") as f:
        f.write(log_line + "\n")
    print(log_line)
    
# 定义熵梯度计算函数
def T_logp2(params, inputs, temperature, model_instance):
    variables = {"params": params}
    preds = model_instance.apply(variables, inputs)
    return 2.0 * temperature * jnp.mean(jnp.real(preds)**2)

def T_logp_2(params, inputs, temperature, model_instance):
    variables = {"params": params}
    preds = model_instance.apply(variables, inputs)
    return 2.0 * temperature * (jnp.mean(jnp.real(preds)))**2

# 基于 VMC_SRt 实现自由能 F = E - T*S 的优化
def clip_gradients(grad, max_norm):
    # 计算所有梯度的L2范数
    total_norm = jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(grad)]))
    # 如果总范数超过max_norm，则计算缩放因子，否则为1.0
    clip_coef = jnp.where(total_norm > max_norm, max_norm / (total_norm + 1e-6), 1.0)
    # 对所有梯度乘以缩放因子
    return jax.tree_map(lambda x: x * clip_coef, grad)

# 在你的训练步长中，例如在 _step_with_state 中修改梯度更新部分：
class FreeEnergyVMC_SRt(VMC_SRt):
    def __init__(self, temperature, clip_norm=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_temperature = temperature
        self.temperature = temperature
        self.clip_norm = clip_norm  # 设置梯度裁剪的阈值

    def _step_with_state(self, state):
        new_state = super()._step_with_state(state)
        params = new_state.parameters
        inputs = new_state.samples

        # 计算熵梯度部分（同之前代码）
        mT_grad_S_1 = jax.grad(T_logp2, argnums=0)(params, inputs, self.temperature, self.variational_state.model)
        mT_grad_S_2 = jax.grad(T_logp_2, argnums=0)(params, inputs, self.temperature, self.variational_state.model)
        mT_grad_S = tree_util.tree_map(lambda x, y: x - y, mT_grad_S_1, mT_grad_S_2)
        
        total_grad = tree_util.tree_map(lambda g_e, g_s: g_e - g_s,
                                          new_state.gradient, mT_grad_S)
        
        # 对梯度进行裁剪
        # total_grad = clip_gradients(total_grad, self.clip_norm)

        new_params = self.optimizer.update(total_grad, params)
        new_state = new_state.replace(parameters=new_params)
        return new_state

# 添加进度条以及温度递减方案
class CustomFreeEnergyVMC_SRt(FreeEnergyVMC_SRt):
    def __init__(self, reference_energy=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_energy = reference_energy

    def run(self, n_iter, energy_log):
        """
        运行优化，通过 log_message 打印 Temperature, Learning Rate, Energy, 
        E_var, E_err，以及当提供 reference_energy 时显示的 Rel_err(%)
        """
        for i in range(n_iter):
            # 更新温度（以及其他内部状态）
            self.temperature = self.init_temperature * (jnp.exp(-i / 50.0) / 2.0)
            self.advance(1)


            energy_mean = self.energy.mean
            energy_var = self.energy.variance
            energy_error = self.energy.error_of_mean

            if self.reference_energy is not None:
                relative_error = abs((energy_mean - self.reference_energy) / self.reference_energy) * 100
                log_message(
                    energy_log,
                    f"Iteration: {i}/{n_iter}, Temp: {self.temperature:.4f}, Energy: {energy_mean:.6f}, Rel_err(%): {relative_error:.4f}"
                )
            else:
                log_message(
                    energy_log,
                    f"Iteration: {i}/{n_iter}, Temp: {self.temperature:.4f}, Energy: {energy_mean:.6f}"
                )
        return self

