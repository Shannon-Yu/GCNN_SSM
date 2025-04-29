# 结果目录

此目录用于存储模拟结果。结果将按以下结构组织：

```
results/
└── L=<size>/                # 按系统大小分类
    └── J2=<value>/          # 按J2参数分类
        └── J1=<value>/      # 按J1参数分类
            ├── training/    # 训练结果
            │   ├── energy_L=<size>_J2=<value>_J1=<value>.log  # 能量日志
            │   ├── parameters_L=<size>_Q=<value>_J1=<value>.txt  # 参数记录
            │   └── GCNN_L=<size>_J2=<value>_J1=<value>.pkl  # 训练后的量子态
            └── analysis/    # 分析结果
                ├── spin_structure_factor.npy      # 自旋结构因子数据
                ├── plaquette_structure_factor.npy # 简盘结构因子数据
                ├── dimer_structure_factor.npy     # 二聚体结构因子数据
                └── plots/                         # 结构因子图像
```

## 文件说明

### 训练结果文件
- `energy_*.log`: 记录训练过程中的能量变化和其他信息
- `parameters_*.txt`: 记录模拟使用的参数配置
- `GCNN_*.pkl`: 保存训练后的量子态参数，可用于后续分析

### 分析结果文件
- `*_structure_factor.npy`: 结构因子数据（自旋、简盘、二聚体）
- `*_correlation_data.npy`: 相关函数数据
- `*_correlation_ratio.npy`: 相关比率数据
- `plots/*.png`: 结构因子热图
