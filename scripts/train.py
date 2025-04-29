#!/usr/bin/env python3
"""
训练脚本，用于训练Shastry-Sutherland模型的GCNN量子态。
"""

# 设置环境变量
import os
import sys

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"  # 启用NetKet的分片功能
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # 使用平台特定的内存分配器
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # 禁用预分配
os.environ["JAX_PLATFORM_NAME"] = "gpu"

import time
import json
import jax
import netket as nk
import netket.optimizer as nk_opt

# 导入自定义模块
from configs.training_config import ModelConfig, TrainingConfig, SystemConfig
from src.models.gcnn import CustomFreeEnergyVMC_SRt
from src.utils.logging import log_message
from src.models.quantum_state import create_quantum_state, save_quantum_state


def run_gcnn_simulation(L, J2, J1):
    """
    利用Group-equivariant Convolutional Neural Network (GCNN)进行变分量子态优化模拟，
    并将主要信息和参数保存到日志文件和参数文件中。
    """
    Q = 1.00 - J2
    # ----------------- 创建结果目录 -----------------
    result_dir = f"results/L={L}/J2={J2:.2f}/J1={J1:.2f}/training"
    os.makedirs(result_dir, exist_ok=True)
    energy_log = os.path.join(result_dir, f"energy_L={L}_J2={J2:.2f}_J1={J1:.2f}.log")

    # ----------------- 创建量子态 -----------------
    vqs, lattice, hilbert, hamiltonian = create_quantum_state(L, J2, J1)
    N = lattice.n_nodes

    # ----------------- 记录初始信息 -----------------
    log_message(energy_log, "="*50)
    log_message(energy_log, "GCNN for Shastry-Sutherland lattice")

    log_message(energy_log, "="*50)
    log_message(energy_log, f"System parameters:")
    log_message(energy_log, f"  - System size: L={L}, N={N}")
    log_message(energy_log, f"  - System parameters: J1={J1}, J2={J2}, Q={Q}")

    # 记录模型参数
    n_params = nk.jax.tree_size(vqs.parameters)
    log_message(energy_log, "-"*50)
    log_message(energy_log, "Model parameters:")
    log_message(energy_log, f"  - Number of layers = {ModelConfig.num_layers}")
    log_message(energy_log, f"  - Number of features = {ModelConfig.num_features}")
    log_message(energy_log, f"  - Total parameters = {n_params}")

    # 训练参数
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

    # 使用学习率调度器创建优化器
    optimizer = nk_opt.Sgd(learning_rate=TrainingConfig.learning_rate)

    # ----------------- 构建退火优化器 -----------------
    n_train = 1   # 每个温度下训练步数

    vmc = CustomFreeEnergyVMC_SRt(
        reference_energy=SystemConfig.reference_energy,
        temperature=TrainingConfig.temperature,
        hamiltonian=hamiltonian,
        optimizer=optimizer,
        diag_shift=TrainingConfig.diag_shift,
        variational_state=vqs
    )

    # ----------------- 记录训练参数 -----------------
    sim_params = {
        "L": L,
        "J1": J1,
        "J2": J2,
        "Q": Q,
        "N_features": ModelConfig.num_features,
        "N_layers": ModelConfig.num_layers,
        "n_samples": TrainingConfig.N_samples,
        "chunk_size": TrainingConfig.chunk_size,
        "learning_rate": TrainingConfig.learning_rate,
        "n_iters": TrainingConfig.N_iters,
        "n_train": n_train,
        "temperature": TrainingConfig.temperature,
        "reference_energy": SystemConfig.reference_energy,
        "n_params": n_params,
    }
    params_file = os.path.join(result_dir, f"parameters_L={L}_Q={Q:.2f}_J1={J1:.2f}.txt")
    with open(params_file, "w") as f:
        json.dump(sim_params, f, indent=4)

    log_message(energy_log, "-"*50)
    log_message(energy_log, "Start training...")

    # ----------------- 启动训练 -----------------
    start_time = time.time()
    vmc.run(n_iter=TrainingConfig.N_iters, energy_log=energy_log)
    end_time = time.time()
    runtime = end_time - start_time

    log_message(energy_log, "="*50)
    log_message(energy_log, f"Training finished, total runtime = {runtime:.2f} seconds")

    # ----------------- 保存训练后量子态参数 -----------------
    state_file = os.path.join(result_dir, f"GCNN_L={L}_J2={J2:.2f}_J1={J1:.2f}.pkl")
    save_quantum_state(vqs, state_file)
    log_message(energy_log, f"Trained quantum state parameters saved to: {state_file}")
    log_message(energy_log, "="*50)


def main():
    """主函数"""
    if len(sys.argv) != 4:
        print("使用方法: python train.py <L> <J2> <J1>")
        sys.exit(1)

    L = int(sys.argv[1])
    J2 = float(sys.argv[2])
    J1 = float(sys.argv[3])

    print("\n开始 ViT-NQS 模拟")
    print(f"Sharding 启用状态: {nk.config.netket_experimental_sharding}")
    print(f"可用设备: {jax.devices()}")
    print(f"GPU 信息:")
    os.system("nvidia-smi")
    print("\n")

    try:
        run_gcnn_simulation(L, J2, J1)
    except Exception as e:
        print(f"L={L}, J2={J2}, J1={J1} 的模拟失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()
