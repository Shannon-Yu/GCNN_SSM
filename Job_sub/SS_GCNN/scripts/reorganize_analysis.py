#!/usr/bin/env python3
"""
重组 analysis 目录结构，对 npy 文件进行分类，删除单独的 log 文件。
"""

import os
import glob
import shutil
import re

# 项目根目录
ROOT_DIR = "/home/users/ntu/s240076/Repositories/Projects/ViT_NQS/Job_sub/SS_GCNN"
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

def reorganize_analysis_folders():
    """重组 analysis 目录结构"""
    # 遍历所有结果目录
    for l_dir in glob.glob(os.path.join(RESULTS_DIR, "L=*")):
        for j2_dir in glob.glob(os.path.join(l_dir, "J2=*")):
            for j1_dir in glob.glob(os.path.join(j2_dir, "J1=*")):
                analysis_dir = os.path.join(j1_dir, "analysis")
                
                if not os.path.exists(analysis_dir):
                    continue
                
                # 创建子目录
                spin_dir = os.path.join(analysis_dir, "spin")
                plaquette_dir = os.path.join(analysis_dir, "plaquette")
                dimer_dir = os.path.join(analysis_dir, "dimer")
                correlation_dir = os.path.join(analysis_dir, "correlation")
                
                os.makedirs(spin_dir, exist_ok=True)
                os.makedirs(plaquette_dir, exist_ok=True)
                os.makedirs(dimer_dir, exist_ok=True)
                os.makedirs(correlation_dir, exist_ok=True)
                
                # 移动文件到相应目录
                for file in glob.glob(os.path.join(analysis_dir, "*.npy")):
                    filename = os.path.basename(file)
                    
                    if filename.startswith("spin_"):
                        shutil.move(file, os.path.join(spin_dir, filename))
                    elif filename.startswith("plaquette_"):
                        shutil.move(file, os.path.join(plaquette_dir, filename))
                    elif filename.startswith("dimer_"):
                        shutil.move(file, os.path.join(dimer_dir, filename))
                    elif "_correlation_" in filename:
                        shutil.move(file, os.path.join(correlation_dir, filename))
                    elif "k_points.npy" == filename:
                        # k_points.npy 文件复制到所有子目录
                        shutil.copy2(file, os.path.join(spin_dir, filename))
                        shutil.copy2(file, os.path.join(plaquette_dir, filename))
                        shutil.copy2(file, os.path.join(dimer_dir, filename))
                        os.remove(file)  # 删除原始文件
                
                # 删除单独的 log 文件
                for log_file in glob.glob(os.path.join(analysis_dir, "*_calculation.log")):
                    os.remove(log_file)

if __name__ == "__main__":
    print("开始重组 analysis 目录结构...")
    reorganize_analysis_folders()
    print("重组完成！")
