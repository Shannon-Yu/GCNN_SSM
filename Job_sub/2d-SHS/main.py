# 在main.py最开始添加
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"  # 启用NetKet的分片功能
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # 使用平台特定的内存分配器
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # 禁用预分配
os.environ["JAX_PLATFORM_NAME"] = "gpu"


import sys
import time
import json
import datetime
import pytz
import jax
import numpy as np
import jax.numpy as jnp
import netket as nk
from flax.serialization import to_bytes
from netket.utils.group import PermutationGroup
from netket.nn.blocks import SymmExpSum
from netket.utils.group.planar import rotation, glide_group
from netket.utils.group import PointGroup, Identity
import netket.optimizer as nk_opt


#自定库
from config import ModelConfig, TrainingConfig, SystemConfig

# from ml_models.standard_ViT import ViTFNQS
# from ml_models.ViT import ViTFNQS
from ml_models.CTWF import CTWFNQS
from optimizers.FE_VMC_SRt import CustomFreeEnergyVMC_SRt, log_message
from phy_models.shastry_sutherland import shastry_sutherland_lattice, shastry_sutherland_hamiltonian, shastry_sutherland_all_symmetries, shastry_sutherland_point_symmetries


def run_simulation(L, J2, J1):
    """执行单个模拟, 固定J2=1.0, 变化J1"""
    # 固定J2值
    Q = 1.00 - J2
    
    # 创建目录结构
    os.makedirs("results", exist_ok=True)
    result_dir = f"results/L={L}/J2={J2:.2f}/J1={J1:.2f}"
    os.makedirs(result_dir, exist_ok=True)

    # 将 energy.log 文件保存在各自的文件夹中
    energy_log = os.path.join(result_dir, f"energy_L={L}_J2={J2:.2f}_J1={J1:.2f}.log")

    # 创建Shastry-Sutherland晶格
    lattice = shastry_sutherland_lattice(L,L)
    N = lattice.n_nodes

    # 记录新的模拟开始
    log_message(energy_log, "="*50)
    log_message(energy_log, f"Study of the Shastry-Sutherland Model")

    log_message(energy_log, "="*50)
    log_message(energy_log, f"System parameters: L={L}, N={N}")
    log_message(energy_log, f"  - System size: L={L}, N={N}")
    log_message(energy_log, f"  - System parameters: J1={J1}, J2={J2}, Q={Q}")
    

    # 创建Shastry-Sutherland模型的Hamiltonian
    ha, hi = shastry_sutherland_hamiltonian(
        lattice=lattice,
        J1=J1,
        J2=J2,
        spin=0.5,
        Q=Q,
        total_sz=0
    )

 # 为Hilbert空间配置Metropolis采样器，利用多GPU
    sampler = nk.sampler.MetropolisExchange(
        hilbert=hi, 
        graph=lattice, 
        n_chains=TrainingConfig.N_samples, 
        d_max=2,
        n_chains_per_rank=None  # 允许基于可用设备自动分配链数
    )

    
    model_no_symm = CTWFNQS(
        num_layers=ModelConfig.num_layers,
        d_model=ModelConfig.d_model,
        heads=ModelConfig.heads,
        n_sites=lattice.n_nodes,
        patch_size=ModelConfig.patch_size,
    )

    # 获取晶格的对称性群
    symmetries = shastry_sutherland_all_symmetries(lattice)

    # 使用SymmExpSum对Transformer模型进行对称化
    model = SymmExpSum(
        module=model_no_symm, 
        symm_group=symmetries, 
        character_id=None
    )

    # 初始化参数
    key = jax.random.PRNGKey(TrainingConfig.seed)
    key, subkey = jax.random.split(key)
    
    # 创建变分量子态
    vqs = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=TrainingConfig.N_samples,
        n_samples_per_rank=None,
        n_discard_per_chain=TrainingConfig.N_discard,
        chunk_size=TrainingConfig.chunk_size,
        sampler_seed=subkey,
    )
    # 模型参数
    n_params = nk.jax.tree_size(vqs.parameters)
    log_message(energy_log, "-"*50)
    log_message(energy_log, "Model parameters:")
    log_message(energy_log, f"  - Number of layers = {ModelConfig.num_layers}")
    log_message(energy_log, f"  - d_model = {ModelConfig.d_model}")
    log_message(energy_log, f"  - Number of heads = {ModelConfig.heads}")
    log_message(energy_log, f"  - Patch size = {ModelConfig.patch_size}")
    # log_message(energy_log, f"  - Complex = {complex}")
    # log_message(energy_log, f"  - Translation invariant = {transl_invariant}")
    log_message(energy_log, f"  - Symmetries used: {len(symmetries)}")
    log_message(energy_log, f"  - Total parameters = {n_params}")

    #训练参数
    log_message(energy_log, "-"*50)
    log_message(energy_log, "Training parameters:")
    log_message(energy_log, f"  - Learning rate: {TrainingConfig.learning_rate}")
    log_message(energy_log, f"  - Total annealing steps: {TrainingConfig.N_iters}")
    log_message(energy_log, f"  - Samples: {TrainingConfig.N_samples}")
    log_message(energy_log, f"  - Discarded samples: {TrainingConfig.N_discard}")
    log_message(energy_log, f"  - Chunk size: {TrainingConfig.chunk_size}")
    
    # 添加检查Sharding状态的日志
    log_message(energy_log, "-"*50)
    log_message(energy_log, "Device status:")
    log_message(energy_log, f"  - Number of devices: {len(jax.devices())}")
    log_message(energy_log, f"  - Sharding: {nk.config.netket_experimental_sharding}")

    
    # 记录模拟使用的各项参数到 txt 文件（JSON 格式保存）
    params_dict = {
        "seed": TrainingConfig.seed,
        "diag_shift": TrainingConfig.diag_shift,
        "learning_rate": TrainingConfig.learning_rate,
        "N_opt": TrainingConfig.N_iters,
        "N_samples": TrainingConfig.N_samples,
        "N_discard": TrainingConfig.N_discard,
        "d_model": ModelConfig.d_model,
        "heads": ModelConfig.heads,
        "num_layers": ModelConfig.num_layers,
        "patch_size": ModelConfig.patch_size,
        "L": L,
        "N": N,
        "J1": J1,
        "J2": J2,
        "Q": Q,
        "lattice_extent": [L, L],
        "lattice_pbc": [True, True],
        "total_sz": 0
    }

    params_file = os.path.join(result_dir, f"parameters_L={L}_J2={J2:.2f}_J1={J1:.2f}.txt")
    with open(params_file, "w") as f_out:
        json.dump(params_dict, f_out, indent=4)

    log_message(energy_log, "-"*50)
    log_message(energy_log, "Start training...")


    # 使用学习率调度器创建优化器
    # 使用学习率调度器创建优化器
    # import optax
    optimizer = nk_opt.Sgd(learning_rate=TrainingConfig.learning_rate)
    # optimizer = optax.noisy_sgd(
    #     learning_rate=TrainingConfig.learning_rate,  # 可以与调度器结合使用
    #     eta=0.01,                   # 噪声初始方差
    #     gamma=0.55                 # 噪声衰减率
    # )

    # optimizer = nk_opt.Sgd(learning_rate=TrainingConfig.eta)
    # 使用自定义的退火优化器
    vmc = CustomFreeEnergyVMC_SRt(
    reference_energy=SystemConfig.reference_energy,
    temperature=TrainingConfig.temperature,
    hamiltonian=ha,
    optimizer=optimizer,
    diag_shift=TrainingConfig.diag_shift,
    variational_state=vqs
    )

    start = time.time()
    vmc.run(n_iter=TrainingConfig.N_iters, energy_log=energy_log)
    end = time.time()
    
    runtime = end - start
    log_message(energy_log, "="*50)
    log_message(energy_log, f"Training finished, total running time = {runtime:.2f} seconds")

    # 保存优化后的量子态（参数）
    import pickle
    state_file = os.path.join(result_dir, f"ViTNQS_L={L}_J2={J2:.2f}_J1={J1:.2f}.pkl")
    with open(state_file, "wb") as f_state:
        pickle.dump(vqs.parameters, f_state)
    log_message(energy_log, f"The trained quantum state parameters have been saved to: {state_file}")
    log_message(energy_log, "="*50)
        


def main():
    """主函数"""
    if len(sys.argv) != 4:
        print("使用方法: python main.py <L> <J2> <J1>")
        sys.exit(1)

    L = int(sys.argv[1])
    J2 = float(sys.argv[2])
    J1 = float(sys.argv[3])

    try:
        run_simulation(L, J2, J1)
    except Exception as e:
        print(f"L={L}, J2={J2}, J1={J1} 的模拟失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()