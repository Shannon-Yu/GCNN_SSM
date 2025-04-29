import jax
import jax.numpy as jnp  
import netket as nk
import time
import numpy as np
from netket.experimental.driver.vmc_srt import VMC_SRt
import json
import optax
import os
import datetime
import pytz
import sys
from netket.nn.blocks import SymmExpSum

from config import ModelConfig, TrainingConfig, SystemConfig
from models.transformer import Transformer_Enc

import os
# Set CUDA environment variables
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

def log_message(log_file, message):
    """记录带时间戳的消息"""
    tz = pytz.timezone("Asia/Singapore")
    timestamp = datetime.datetime.now(tz).strftime("[%Y-%m-%d %H:%M:%S]")
    log_line = f"{timestamp} {message}"
    with open(log_file, "a") as f:
        f.write(log_line + "\n")
    print(log_line)

def run_simulation(L, J2):
    """执行单个模拟"""
    # 系统参数
    N = SystemConfig.get_size(L)
    
    # 创建目录结构
    os.makedirs("results", exist_ok=True)
    result_dir = f"results/L={L}/J2={J2:.3f}"
    os.makedirs(result_dir, exist_ok=True)

    # 将 energy.log 文件保存在各自的文件夹中
    energy_log = os.path.join(result_dir, f"energy_L={L}_J2={J2:.3f}.log")
    output_file = os.path.join(result_dir, f"L={L}_J2={J2:.3f}")

    # 记录新的模拟开始
    log_message(energy_log, "\n" + "="*50)
    log_message(energy_log, f"Starting calculation for L={L}, J2={J2}")
    log_message(energy_log, "="*50)
    log_message(energy_log, f"System size: L={L}, N={N}")
    log_message(energy_log, f"Parameters: J1={SystemConfig.J1}, J2={J2}")
    log_message(energy_log, "Optimization parameters:")
    log_message(energy_log, f"  - Learning rate: {TrainingConfig.eta}")
    log_message(energy_log, f"  - Number of iterations: {TrainingConfig.N_opt}")
    log_message(energy_log, f"  - Number of samples: {TrainingConfig.N_samples}")
    log_message(energy_log, f"  - Number of samples to discard: {TrainingConfig.N_discard}")
    log_message(energy_log, f"  - Chunk size: {TrainingConfig.chunk_size}")
    log_message(energy_log, "-"*50)

    # 设置物理系统
    lattice = nk.graph.Grid(extent=[L, L], pbc=[True, True], max_neighbor_order=2)
    hilbert = nk.hilbert.Spin(s=1/2, N=lattice.n_nodes, total_sz=0)
    hamiltonian = nk.operator.Heisenberg(hilbert=hilbert, graph=lattice, J=[SystemConfig.J1, J2])

    symm_group = lattice.point_group()
    # 初始化变分波函数
    wf_no_symm = Transformer_Enc(
        d_model=ModelConfig.d_model,
        num_heads=ModelConfig.heads,
        num_patches=N // ModelConfig.patch_size,
        patch_size=ModelConfig.patch_size,
        n_layers=ModelConfig.n_layers
    )

    wf_sym = SymmExpSum(module=wf_no_symm, symm_group=symm_group, character_id=None)

    # 初始化采样器和状态
    key = jax.random.PRNGKey(TrainingConfig.seed)
    key, subkey = jax.random.split(key)
    params = wf_sym.init(subkey, jnp.zeros((1, lattice.n_nodes)))
    sampler = nk.sampler.MetropolisExchange(
        hilbert=hilbert,
        graph=lattice,
        d_max=2,
        n_chains=TrainingConfig.N_samples,
        sweep_size=lattice.n_nodes
    )

    key, subkey = jax.random.split(key)
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=wf_sym,
        sampler_seed=subkey,
        n_samples=TrainingConfig.N_samples,
        n_discard_per_chain=TrainingConfig.N_discard,
        variables=params,
        chunk_size=TrainingConfig.chunk_size
    )

    # 记录参数数量
    n_params = nk.jax.tree_size(vstate.parameters)
    log_message(energy_log, f"Number of parameters = {n_params}")

    # 记录模拟使用的各项参数到 txt 文件（JSON 格式保存）
    params_dict = {
        "seed": TrainingConfig.seed,
        "diag_shift": TrainingConfig.diag_shift,
        "eta": TrainingConfig.eta,
        "N_opt": TrainingConfig.N_opt,
        "N_samples": TrainingConfig.N_samples,
        "N_discard": TrainingConfig.N_discard,
        "f": getattr(ModelConfig, "f", ModelConfig.d_model),
        "heads": ModelConfig.heads,
        "d_model": ModelConfig.d_model,
        "patch_size": ModelConfig.patch_size,
        "n_layers": ModelConfig.n_layers,
        "L": L,
        "N": N,
        "J2": J2,
        "lattice_extent": [L, L],
        "lattice_pbc": [True, True],
        "max_neighbor_order": 2,
        "hilbert_spin": 0.5,
        "total_sz": 0
    }

    params_file = os.path.join(result_dir, f"parameters_L={L}_J2={J2:.3f}.txt")
    with open(params_file, "w") as f_out:
        json.dump(params_dict, f_out, indent=4)
    log_message(energy_log, "Starting optimization...")

    # 优化器设置
    clipper = optax.clip_by_global_norm(1.0)
    optimizer = optax.chain(clipper, optax.sgd(learning_rate=TrainingConfig.eta))
    vmc = VMC_SRt(
        hamiltonian=hamiltonian,
        optimizer=optimizer,
        diag_shift=TrainingConfig.diag_shift,
        variational_state=vstate
    )

    def check_energy(step, log_data, driver):
        """回调函数，用于检查和记录能量"""
        energy = driver.state.expect(hamiltonian).mean
        total_energy = energy / 4
        if step % 100 == 0:
            log_message(energy_log, 
                        f"Step {step}/{TrainingConfig.N_opt}: "
                        f"Current Total Energy = {total_energy:.6f}")
        
        if np.isnan(energy):
            log_message(energy_log, f"NaN energy detected at step {step}")
            driver.stop_training = True
            return False
        return True

    try:
        start = time.time()
        vmc.run(out=output_file, n_iter=TrainingConfig.N_opt, callback=check_energy)
        end = time.time()
        
        runtime = end - start
        log_message(energy_log, "="*50)
        log_message(energy_log, f"Optimization completed for L={L}, J2={J2}")
        log_message(energy_log, f"Total runtime = {runtime:.2f} seconds")

        # 保存优化后的量子态（参数）
        from flax.serialization import to_bytes  # 导入序列化工具
        state_file = os.path.join(result_dir, f"state_L={L}_J2={J2:.3f}.bin")
        with open(state_file, "wb") as f_state:
            f_state.write(to_bytes(vstate.parameters))
        log_message(energy_log, "="*50)
        
    except Exception as e:
        log_message(energy_log, f"Error during optimization: {str(e)}")
        raise

def main():
    """主函数"""
    if len(sys.argv) != 3:
        print("Usage: python main.py <L> <J2>")
        sys.exit(1)

    L = int(sys.argv[1])
    J2 = float(sys.argv[2])

    print("\nStarting ViT-NQS simulation")
    print(f"GPU Information:")
    os.system("nvidia-smi")
    print("\n")

    try:
        run_simulation(L, J2)
    except Exception as e:
        print(f"Failed for L={L}, J2={J2}: {str(e)}")
        raise

if __name__ == "__main__":
    main()
