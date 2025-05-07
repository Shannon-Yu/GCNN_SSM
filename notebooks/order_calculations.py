import numpy as np
import os
import matplotlib.pyplot as plt

# 定义日志记录函数
def log_message(log_file, message):
    print(message) # 在脚本执行时直接打印
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

def calculate_af_order_parameter(k_points_tuple, spin_sf, L, save_dir, log_file=None, spin_data=None):
    """
    计算反铁磁序参数 (AF Order Parameter)：m^2(L) = S(π, π)/L^2
    
    参数:
    - k_points_tuple: 包含k_points_x和k_points_y的元组
    - spin_sf: 自旋结构因子
    - L: 系统大小
    - save_dir: 保存目录
    - log_file: 日志文件
    - spin_data: 自旋相关函数数据列表（可选，用于计算实空间的反铁磁序参量）
    """
    k_points_x, k_points_y = k_points_tuple
    log_message(log_file, "-"*80)

    pi_idx_x = np.argmin(np.abs(k_points_x - np.pi))
    pi_idx_y = np.argmin(np.abs(k_points_y - np.pi))
    S_pi_pi = spin_sf[pi_idx_y, pi_idx_x]
    
    # 记录找到的k点位置
    k_pi_x = k_points_x[pi_idx_x]
    k_pi_y = k_points_y[pi_idx_y]
    
    # 找到结构因子的最大值位置，用于比较
    max_idx = np.unravel_index(np.argmax(spin_sf), spin_sf.shape)
    k_max_x = k_points_x[max_idx[1]]
    k_max_y = k_points_y[max_idx[0]]
    S_max = spin_sf[max_idx].item()
    
    log_message(log_file, f"反铁磁序参量S(π, π) = {S_pi_pi:.6f}")
    log_message(log_file, f"峰值位置: ({k_max_x:.4f}, {k_max_y:.4f}), 峰值 S = {S_max:.6f}")
    
    return S_pi_pi

def calculate_dimer_order_parameter(dimer_data, L, save_dir, log_file=None):
    """
    计算二聚体序参量 (Dimer Order Parameter)：D^2 = \frac{1}{N} \sum_r C_d(r)(-1)^{r_x}

    参数:
    - dimer_data: 二聚体相关函数数据列表
    - L: 系统大小
    - save_dir: 保存目录
    - log_file: 日志文件
    """
    log_message(log_file, "-"*80)
    log_message(log_file, "计算二聚体序参量...")

    # 计算二聚体序参量，分别处理x方向和y方向
    D_squared_sum_x = 0.0  # x方向
    D_squared_sum_y = 0.0  # y方向
    count_x = 0
    count_y = 0

    for data in dimer_data:
        r_x = data['r_x']
        r_y = data['r_y']
        direction = data.get('direction', 'x')  # 默认为x方向

        # 根据二聚体方向选择合适的坐标计算符号
        if direction == 'x':
            # x方向二聚体使用x坐标计算符号
            sign = (-1) ** int(round(r_x))
            D_squared_sum_x += data['corr'] * sign
            count_x += 1
        elif direction == 'y':
            # y方向二聚体使用y坐标计算符号
            sign = (-1) ** int(round(r_y))
            D_squared_sum_y += data['corr'] * sign
            count_y += 1

    # 总位点数
    N = L * L * 4

    # 分别归一化x和y方向
    if count_x > 0:
        D_squared_x = D_squared_sum_x / N
    else:
        D_squared_x = 0.0
        log_message(log_file, "警告: 没有x方向二聚体相关函数数据")

    if count_y > 0:
        D_squared_y = D_squared_sum_y / N
    else:
        D_squared_y = 0.0
        log_message(log_file, "警告: 没有y方向二聚体相关函数数据")

    # 计算总的二聚体序参量（取平均）
    D_squared = 0.5 * (D_squared_x + D_squared_y)

    log_message(log_file, f"x方向二聚体序参量: D^2_x = {D_squared_x:.6f}")
    log_message(log_file, f"y方向二聚体序参量: D^2_y = {D_squared_y:.6f}")
    log_message(log_file, f"平均二聚体序参量 D^2 = {D_squared:.6f}")

    return D_squared, (D_squared_x, D_squared_y)

def calculate_plaquette_order_parameter(plaquette_data, L, save_dir, log_file=None):
    """
    计算简盘序参量 (Plaquette Order Parameter)：m_p(L) = |C(L/2, L/2) - C(L/2 - 1, L/2 - 1)|

    参数:
    - plaquette_data: 简盘相关函数数据列表
    - L: 系统大小
    - save_dir: 保存目录
    - log_file: 日志文件
    """
    log_message(log_file, "-"*80)
    log_message(log_file, "计算简盘序参量...")

    # 从plaquette_data中提取相关函数
    # 获取晶格的物理尺寸（每个简盘在物理空间中占据2x2的区域）
    # 由于我们没有直接访问lattice对象，使用L作为简盘数量来计算物理尺寸
    physical_size_x = L * 2.0  # 每个简盘在x方向占2个单位
    physical_size_y = L * 2.0  # 每个简盘在y方向占2个单位

    # 寻找位置接近系统中心和接近中心的数据点
    C_L2_L2 = None
    C_L2m1_L2m1 = None

    # 系统中心位置
    target_r1 = np.array([physical_size_x/2, physical_size_y/2])
    # 接近中心的位置（移动一个自旋的距离，而不是一个简盘）
    target_r2 = np.array([physical_size_x/2 - 2.0, physical_size_y/2 - 2.0])

    min_dist1 = float('inf')
    min_dist2 = float('inf')
    r1_actual = None
    r2_actual = None

    for data in plaquette_data:
        r = np.array([data['r_x'], data['r_y']])

        # 计算与目标位置的距离
        dist1 = np.linalg.norm(r - target_r1)
        dist2 = np.linalg.norm(r - target_r2)

        # 更新最接近系统中心的点
        if dist1 < min_dist1:
            min_dist1 = dist1
            C_L2_L2 = data['corr']
            r1_actual = r

        # 更新最接近次中心位置的点
        if dist2 < min_dist2:
            min_dist2 = dist2
            C_L2m1_L2m1 = data['corr']
            r2_actual = r

    # 如果找不到合适的点，给出警告
    if C_L2_L2 is None or C_L2m1_L2m1 is None:
        log_message(log_file, "警告: 无法找到合适的相关函数数据点来计算简盘序参量")
        return 0.0, (None, None)

    # 计算简盘序参量
    m_p = abs(C_L2_L2 - C_L2m1_L2m1)

    log_message(log_file, f"C(L/2,L/2) 在 {r1_actual}: {C_L2_L2:.6f} (距离: {min_dist1:.4f})")
    log_message(log_file, f"C(L/2-1,L/2-1) 在 {r2_actual}: {C_L2m1_L2m1:.6f} (距离: {min_dist2:.4f})")
    log_message(log_file, f"简盘序参量 m_p(L) = {m_p:.6f}")

    return m_p, (C_L2_L2, C_L2m1_L2m1)

def calculate_correlation_ratios(k_points_tuple, structure_factor, save_dir, type_name, log_file=None):
    """
    计算相关比率 R = 1 - S(k+δk)/S(k)，其中δk = 2π/L
    使用最近的网格点计算S(k+δk)，而不是插值

    参数:
    - k_points_tuple: 包含k_points_x和k_points_y的元组
    - structure_factor: 结构因子数据
    - save_dir: 保存目录
    - type_name: 结构因子类型名称
    - log_file: 日志文件
    """
    # 解包k点
    k_points_x, k_points_y = k_points_tuple

    log_message(log_file, "-"*80)

    # 找到结构因子的最大值位置
    max_idx = np.unravel_index(np.argmax(structure_factor), structure_factor.shape)
    max_idx_y, max_idx_x = max_idx  # 注意：第一个索引是y，第二个是x
    k_max_x = k_points_x[max_idx_x]
    k_max_y = k_points_y[max_idx_y]
    S_max = structure_factor[max_idx].item()

    # 找到最接近k_max + δk的网格点
    n_kx = len(k_points_x)
    n_ky = len(k_points_y)

    # 直接使用相邻索引确定点
    idx_x_plus = max_idx_x + 1 if max_idx_x < n_kx - 1 else max_idx_x
    idx_y_plus = max_idx_y + 1 if max_idx_y < n_ky - 1 else max_idx_y
    
    k_x_plus_actual = k_points_x[idx_x_plus]

    # y方向
    k_y_plus_actual = k_points_y[idx_y_plus]

    # 获取对应的结构因子值
    S_kxplus = structure_factor[max_idx_y, idx_x_plus].item()
    S_kyplus = structure_factor[idx_y_plus, max_idx_x].item()

    # 取平均
    S_kplus = 0.5 * (S_kxplus + S_kyplus)

    log_message(log_file, f"峰值位置: ({k_max_x:.4f}, {k_max_y:.4f}), 峰值 S = {S_max:.6f}")
    log_message(log_file, f"最近网格点: x=({k_x_plus_actual:.4f}, {k_max_y:.4f}), y=({k_max_x:.4f}, {k_y_plus_actual:.4f})")
    log_message(log_file, f"S_(kx+δ) = {S_kxplus:.6f}, S_(ky+δ) = {S_kyplus:.6f}, S_(k+δ) = {S_kplus:.6f}")

    # 计算相关比率
    ratio = 1.0 - S_kplus / S_max

    log_message(log_file, f"{type_name.capitalize()} 相关比率: {ratio:.4f}")

    return ratio, (k_max_x, k_max_y, S_max, S_kplus)

# 绘图函数
def plot_structure_factor_with_max(k_points_tuple, structure_factor, L, J2, J1, type_name, save_dir=None):
    """绘制结构因子并标记最大值位置"""
    k_points_x, k_points_y = k_points_tuple
    
    # 创建网格
    kx, ky = np.meshgrid(k_points_x, k_points_y)
    
    # 找到最大值位置
    max_idx = np.unravel_index(np.argmax(structure_factor), structure_factor.shape)
    k_max_x = k_points_x[max_idx[1]]
    k_max_y = k_points_y[max_idx[0]]
    S_max = structure_factor[max_idx]
    
    # 绘制
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(kx, ky, structure_factor, 50, cmap='viridis')
    plt.colorbar(cp, label=f'{type_name} Structure Factor')
    plt.xlabel('$k_x$')
    plt.ylabel('$k_y$')
    plt.title(f'{type_name} Structure Factor (L={L}, J2={J2:.2f}, J1={J1:.2f})')
    
    # 标记最大值
    plt.plot(k_max_x, k_max_y, 'r*', markersize=10, label=f'Max: S({k_max_x:.2f}, {k_max_y:.2f}) = {S_max:.4f}')
    
    # 标记 (π,π) 点 - 对于Néel反铁磁顺序很重要
    pi_idx_x = np.argmin(np.abs(k_points_x - np.pi))
    pi_idx_y = np.argmin(np.abs(k_points_y - np.pi))
    k_pi_x = k_points_x[pi_idx_x]
    k_pi_y = k_points_y[pi_idx_y]
    S_pi_pi = structure_factor[pi_idx_y, pi_idx_x]
    plt.plot(k_pi_x, k_pi_y, 'go', markersize=8, label=f'S(π,π) = {S_pi_pi:.4f}')
    
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{type_name.lower()}_sf_L{L}_J2{J2:.2f}_J1{J1:.2f}.png"), dpi=300, bbox_inches='tight')
    
    plt.show()
    
def plot_order_parameters_vs_J1(J1_values, order_values, system_size, J2, type_name):
    """绘制序参量随J1变化的曲线"""
    plt.figure(figsize=(8, 6))
    plt.plot(J1_values, order_values, 'o-', markersize=8, linewidth=2)
    
    plt.xlabel('$J_1$')
    plt.ylabel(f'{type_name} Order Parameter')
    plt.title(f'{type_name} Order vs $J_1$ (L={system_size}, $J_2$={J2:.2f})')
    
    plt.grid(alpha=0.3)
    plt.show()
    
def plot_correlation_ratios_vs_J1(J1_values, ratio_values, system_size, J2, type_name):
    """绘制相关比率随J1变化的曲线"""
    plt.figure(figsize=(8, 6))
    plt.plot(J1_values, ratio_values, 'o-', markersize=8, linewidth=2)
    
    plt.xlabel('$J_1$')
    plt.ylabel(f'{type_name} Correlation Ratio')
    plt.title(f'{type_name} Correlation Ratio vs $J_1$ (L={system_size}, $J_2$={J2:.2f})')
    
    plt.grid(alpha=0.3)
    plt.show()
    
def plot_all_order_parameters(J1_values, data_dict, system_size, J2):
    """在一张图上绘制所有序参量随J1变化的曲线"""
    plt.figure(figsize=(10, 7))
    
    if 'af_order' in data_dict:
        af_data = data_dict['af_order']
        plt.plot(J1_values, af_data, 'o-', label='AF Order', linewidth=2)
    
    if 'dimer_order' in data_dict:
        dimer_data = data_dict['dimer_order']
        plt.plot(J1_values, dimer_data, 's-', label='Dimer Order', linewidth=2)
    
    if 'plaq_order' in data_dict:
        plaq_data = data_dict['plaq_order']
        plt.plot(J1_values, plaq_data, '^-', label='Plaquette Order', linewidth=2)
    
    plt.xlabel('$J_1/J_2$', fontsize=14)
    plt.ylabel('Order Parameter', fontsize=14)
    plt.title(f'Order Parameters vs $J_1/J_2$ (L={system_size}, $J_2$={J2:.2f})', fontsize=16)
    
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    
def plot_all_correlation_ratios(J1_values, data_dict, system_size, J2):
    """在一张图上绘制所有相关比率随J1变化的曲线"""
    plt.figure(figsize=(10, 7))
    
    if 'neel_ratio' in data_dict:
        neel_data = data_dict['neel_ratio']
        plt.plot(J1_values, neel_data, 'o-', label='Neel Ratio', linewidth=2)
    
    if 'dimer_ratio' in data_dict:
        dimer_data = data_dict['dimer_ratio']
        plt.plot(J1_values, dimer_data, 's-', label='Dimer Ratio', linewidth=2)
    
    if 'plaq_ratio' in data_dict:
        plaq_data = data_dict['plaq_ratio']
        plt.plot(J1_values, plaq_data, '^-', label='Plaquette Ratio', linewidth=2)
    
    plt.xlabel('$J_1/J_2$', fontsize=14)
    plt.ylabel('Correlation Ratio', fontsize=14)
    plt.title(f'Correlation Ratios vs $J_1/J_2$ (L={system_size}, $J_2$={J2:.2f})', fontsize=16)
    
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show() 