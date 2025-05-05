#!/usr/bin/env python3
"""
分析脚本，用于分析Shastry-Sutherland模型的GCNN量子态。
"""

# 设置环境变量
import os
import sys

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 设置环境变量
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"  # 保留NetKet的分片功能
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "gpu"

import argparse
import traceback
import numpy as np

# 导入自定义模块
from src.utils.logging import log_message
from src.utils.plotting import plot_structure_factor
from src.models.quantum_state import load_quantum_state
from src.analysis.structure_factors import (
    calculate_spin_structure_factor,
    calculate_plaquette_structure_factor,
    calculate_dimer_structure_factor,
    calculate_correlation_ratios,
    calculate_af_order_parameter,
    calculate_plaquette_order_parameter,
    calculate_dimer_order_parameter
)

def main(args=None):
    """主函数，接受命令行参数L, J2, J1"""
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='分析量子态结构因子')
    parser.add_argument('--L', type=int, required=True, help='晶格大小')
    parser.add_argument('--J2', type=float, required=True, help='J2耦合强度')
    parser.add_argument('--J1', type=float, required=True, help='J1耦合强度')
    args = parser.parse_args(args)

    # 获取参数
    L = args.L
    J2 = args.J2
    J1 = args.J1

    # 创建结果目录
    result_dir = f"results/L={L}/J2={J2:.2f}/J1={J1:.2f}"
    analysis_dir = os.path.join(result_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # 创建日志文件
    analyze_log = os.path.join(analysis_dir, f"analyze_L={L}_J2={J2:.2f}_J1={J1:.2f}.log")
    log_message(analyze_log, "="*80)
    log_message(analyze_log, f"开始分析量子态: L={L}, J2={J2:.2f}, J1={J1:.2f}")

    # 不再需要单独的plots目录，图像将保存在各自的目录中

    try:
        # 构建模型文件路径
        model_file = os.path.join(result_dir, "training", f"GCNN_L={L}_J2={J2:.2f}_J1={J1:.2f}.pkl")
        if not os.path.exists(model_file):
            log_message(analyze_log, f"错误: 未找到模型文件 {model_file}")
            return

        # 加载量子态 - 使用较小的采样数初始化
        log_message(analyze_log, "-"*80)
        log_message(analyze_log, f"加载量子态: L={L}, J2={J2:.2f}, J1={J1:.2f}")
        vqs, lattice, _, _ = load_quantum_state(
            model_file, L, J2, J1,
            n_samples=2**12,  # 初始使用较小的采样数
            n_discard=0,
            chunk_size=2**10  # 使用较小的chunk_size
        )

        # 在计算前增加采样数
        log_message(analyze_log, f"增加采样数量到 2^20...")
        vqs.n_samples = 2**20  # 在实际计算前增加采样数

        # 创建子目录
        spin_dir = os.path.join(analysis_dir, "spin")
        plaquette_dir = os.path.join(analysis_dir, "plaquette")
        dimer_dir = os.path.join(analysis_dir, "dimer")

        os.makedirs(spin_dir, exist_ok=True)
        os.makedirs(plaquette_dir, exist_ok=True)
        os.makedirs(dimer_dir, exist_ok=True)

        # 计算自旋因子
        log_message(analyze_log, "="*80)
        log_message(analyze_log, "1. 计算自旋结构因子")
        k_points_tuple, spin_sf = calculate_spin_structure_factor(vqs, lattice, L, spin_dir, analyze_log)
        plot_structure_factor(k_points_tuple, spin_sf, L, J2, J1, "Spin", spin_dir)
        # 加载自旋相关函数数据，与plaquette和dimer处理方式保持一致
        spin_data = np.load(os.path.join(spin_dir, "spin_correlation_data.npy"), allow_pickle=True)
        af_order = calculate_af_order_parameter(k_points_tuple, spin_sf, L, spin_dir, analyze_log, spin_data)
        neel_ratio, _ = calculate_correlation_ratios(k_points_tuple, spin_sf, spin_dir, "neel", analyze_log)

        # 计算二聚体结构因子
        log_message(analyze_log, "="*80)
        log_message(analyze_log, "2. 计算二聚体结构因子")
        k_points_tuple, dimer_sf = calculate_dimer_structure_factor(vqs, lattice, L, dimer_dir, analyze_log)
        plot_structure_factor(k_points_tuple, dimer_sf, L, J2, J1, "Dimer", dimer_dir)
        dimer_data = np.load(os.path.join(dimer_dir, "dimer_correlation_data.npy"), allow_pickle=True)
        dimer_order = calculate_dimer_order_parameter(dimer_data, L, dimer_dir, analyze_log)
        dimer_ratio, _ = calculate_correlation_ratios(k_points_tuple, dimer_sf, dimer_dir, "dimer", analyze_log)

        # 计算简盘因子
        log_message(analyze_log, "="*80)
        log_message(analyze_log, "3. 计算简盘结构因子")
        k_points_tuple, plaq_sf = calculate_plaquette_structure_factor(vqs, lattice, L, plaquette_dir, analyze_log)
        plot_structure_factor(k_points_tuple, plaq_sf, L, J2, J1, "Plaquette", plaquette_dir)
        plaquette_data = np.load(os.path.join(plaquette_dir, "plaquette_correlation_data.npy"), allow_pickle=True)
        plaq_order = calculate_plaquette_order_parameter(plaquette_data, L, plaquette_dir, analyze_log)
        plaq_ratio, _ = calculate_correlation_ratios(k_points_tuple, plaq_sf, plaquette_dir, "plaquette", analyze_log)



        # 输出结果摘要
        log_message(analyze_log, "="*80)
        log_message(analyze_log, f"分析完成! 相关比率: Neel={neel_ratio:.4f}, Plaquette={plaq_ratio:.4f}, Dimer={dimer_ratio:.4f}")
        log_message(analyze_log, f"序参量: AF Order={af_order:.6f}, Plaquette Order={plaq_order:.6f}, Dimer Order={dimer_order:.6f}")
        log_message(analyze_log, "="*80)

    except Exception as e:
        log_message(analyze_log, "!"*80)
        log_message(analyze_log, f"处理 L={L}, J2={J2:.2f}, J1={J1:.2f} 时出错: {str(e)}")
        log_message(analyze_log, traceback.format_exc())
        log_message(analyze_log, "!"*80)

if __name__ == "__main__":
    main()
