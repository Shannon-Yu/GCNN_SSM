import numpy as np
import netket as nk
import os
from src.utils.logging import log_message

def create_k_mesh(n_points=30):
    """创建k点网格"""
    k_points = np.linspace(-np.pi, np.pi, n_points)
    kx, ky = np.meshgrid(k_points, k_points)
    return k_points, kx, ky

def calculate_spin_structure_factor(vqs, lattice, L, save_dir, log_file=None):
    """计算自旋结构因子 S(k) = ∑_r e^(ik·r) <S_0·S_r>"""
    if log_file is None:
        # 默认日志文件将由调用者提供
        log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L}_J2=0.05_J1=0.05.log")
    log_message(log_file, "-"*80)
    log_message(log_file, "开始计算自旋结构因子...")

    N = lattice.n_nodes

    # 创建k点网格
    k_points, _, _ = create_k_mesh(30)

    # 初始化结构因子
    spin_sf = np.zeros((len(k_points), len(k_points)), dtype=complex)

    # 计算自旋-自旋相关函数并构建结构因子
    correlation_data = []

    for i in range(N):
        pos_i = lattice.positions[i]
        sx_i = nk.operator.spin.sigmax(vqs.hilbert, i) * 0.5
        sy_i = nk.operator.spin.sigmay(vqs.hilbert, i) * 0.5
        sz_i = nk.operator.spin.sigmaz(vqs.hilbert, i) * 0.5

        # 每处理10个位点记录一次进度
        if i % 10 == 0:
            log_message(log_file, f"处理位点 {i}/{N}...")

        for j in range(N):
            pos_j = lattice.positions[j]
            r_ij = np.array(pos_j) - np.array(pos_i)

            # 计算三个方向的自旋相关函数
            sx_j = nk.operator.spin.sigmax(vqs.hilbert, j) * 0.5
            sy_j = nk.operator.spin.sigmay(vqs.hilbert, j) * 0.5
            sz_j = nk.operator.spin.sigmaz(vqs.hilbert, j) * 0.5

            # 分别计算xx, yy, zz相关函数并获取实际值
            corr_xx_op = sx_i @ sx_j
            corr_yy_op = sy_i @ sy_j
            corr_zz_op = sz_i @ sz_j

            # 使用.mean属性获取期望值的均值
            corr_xx_val = vqs.expect(corr_xx_op).mean.real
            corr_yy_val = vqs.expect(corr_yy_op).mean.real
            corr_zz_val = vqs.expect(corr_zz_op).mean.real

            # 总自旋相关函数
            corr_ij = corr_xx_val + corr_yy_val + corr_zz_val

            # 保存相关函数数据
            correlation_data.append({
                'i': i, 'j': j,
                'r_x': r_ij[0], 'r_y': r_ij[1],
                'corr': corr_ij
            })

            # 计算结构因子的傅里叶变换
            for kx_idx, kx_val in enumerate(k_points):
                for ky_idx, ky_val in enumerate(k_points):
                    k_vec = np.array([kx_val, ky_val])
                    phase = np.exp(1j * np.dot(k_vec, r_ij))
                    spin_sf[ky_idx, kx_idx] += corr_ij * phase

    # 归一化
    spin_sf /= N

    # 保存相关函数数据
    np.save(os.path.join(save_dir, "spin_correlation_data.npy"), correlation_data)
    np.save(os.path.join(save_dir, "spin_structure_factor.npy"), spin_sf.real)
    np.save(os.path.join(save_dir, "k_points.npy"), k_points)

    log_message(log_file, "自旋结构因子计算完成")

    return k_points, spin_sf.real

def calculate_plaquette_structure_factor(vqs, lattice, L, save_dir, log_file=None):
    """计算简盘结构因子，使用自旋交换算符实现循环置换"""
    if log_file is None:
        # 默认日志文件将由调用者提供
        log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L}_J2=0.05_J1=0.05.log")
    log_message(log_file, "-"*80)
    log_message(log_file, "开始计算简盘结构因子...")

    # 创建k点网格
    k_points, _, _ = create_k_mesh(30)

    # 初始化简盘结构因子
    plaq_sf = np.zeros((len(k_points), len(k_points)), dtype=complex)

    # 识别所有简盘(每个单元格4个位点形成一个简盘)
    plaquettes = []
    for x in range(L):
        for y in range(L):
            base = 4 * (y + x * L)
            # 单元格内的四个点，按照左下、右下、右上、左上排列
            plaquettes.append([base, base+1, base+2, base+3])

    plaquette_data = []

    # 循环计算所有简盘对的相关函数
    for i, plaq_i in enumerate(plaquettes):
        # 计算简盘中心位置
        pos_i_x = np.mean([lattice.positions[p][0] for p in plaq_i])
        pos_i_y = np.mean([lattice.positions[p][1] for p in plaq_i])
        pos_i = np.array([pos_i_x, pos_i_y])

        # 使用自旋交换算符构建循环置换操作符
        P_i, P_i_inv = construct_plaquette_permutation(vqs.hilbert, plaq_i)

        # 每处理几个简盘记录一次进度
        if i % 5 == 0:
            log_message(log_file, f"处理简盘 {i}/{len(plaquettes)}...")

        for j, plaq_j in enumerate(plaquettes):
            # 计算简盘中心位置
            pos_j_x = np.mean([lattice.positions[p][0] for p in plaq_j])
            pos_j_y = np.mean([lattice.positions[p][1] for p in plaq_j])
            pos_j = np.array([pos_j_x, pos_j_y])

            # 计算位移向量
            r_ij = pos_j - pos_i

            # 构建简盘j的循环置换操作符
            P_j, P_j_inv = construct_plaquette_permutation(vqs.hilbert, plaq_j)

            # 构建操作符 (P_i + P_i^-1) * (P_j + P_j^-1)
            op_i = P_i + P_i_inv
            op_j = P_j + P_j_inv
            combined_op = op_i @ op_j

            # 计算期望值
            corr_ij = vqs.expect(combined_op).mean.real / 4.0  # 除以4得到正确的归一化

            plaquette_data.append({
                'plaq_i': i, 'plaq_j': j,
                'r_x': r_ij[0], 'r_y': r_ij[1],
                'corr': corr_ij
            })

            # 计算结构因子
            for kx_idx, kx_val in enumerate(k_points):
                for ky_idx, ky_val in enumerate(k_points):
                    k_vec = np.array([kx_val, ky_val])
                    phase = np.exp(1j * np.dot(k_vec, r_ij))
                    plaq_sf[ky_idx, kx_idx] += corr_ij * phase

    # 归一化
    plaq_sf /= len(plaquettes)

    # 保存数据
    np.save(os.path.join(save_dir, "plaquette_correlation_data.npy"), plaquette_data)
    np.save(os.path.join(save_dir, "plaquette_structure_factor.npy"), plaq_sf.real)

    log_message(log_file, "简盘结构因子计算完成")

    return k_points, plaq_sf.real

def construct_plaquette_permutation(hilbert, plaq_sites):
    """
    构建简盘循环置换操作符，使用自旋交换算符
    P = S_{1,2} S_{2,3} S_{3,4} 实现循环置换 (1,2,3,4) -> (4,1,2,3)
    P^-1 = S_{1,4} S_{4,3} S_{3,2} 实现逆循环置换 (1,2,3,4) -> (2,3,4,1)

    其中 S_{i,j} = (1/2 + 2S_i·S_j) 是交换算符
    """
    a, b, c, d = plaq_sites

    # 构建各个位点的自旋算符
    S_a = [nk.operator.spin.sigmax(hilbert, a) * 0.5,
           nk.operator.spin.sigmay(hilbert, a) * 0.5,
           nk.operator.spin.sigmaz(hilbert, a) * 0.5]

    S_b = [nk.operator.spin.sigmax(hilbert, b) * 0.5,
           nk.operator.spin.sigmay(hilbert, b) * 0.5,
           nk.operator.spin.sigmaz(hilbert, b) * 0.5]

    S_c = [nk.operator.spin.sigmax(hilbert, c) * 0.5,
           nk.operator.spin.sigmay(hilbert, c) * 0.5,
           nk.operator.spin.sigmaz(hilbert, c) * 0.5]

    S_d = [nk.operator.spin.sigmax(hilbert, d) * 0.5,
           nk.operator.spin.sigmay(hilbert, d) * 0.5,
           nk.operator.spin.sigmaz(hilbert, d) * 0.5]

    # 构建交换算符 S_{i,j} = (1/2 + 2S_i·S_j)
    def exchange_op(S_i, S_j):
        # 计算 S_i·S_j = S^x_i·S^x_j + S^y_i·S^y_j + S^z_i·S^z_j
        SiSj = S_i[0] @ S_j[0] + S_i[1] @ S_j[1] + S_i[2] @ S_j[2]
        # 返回 1/2 + 2(S_i·S_j)
        return nk.operator.LocalOperator(hilbert, constant=0.5) + 2.0 * SiSj

    # 构建正向循环置换 P = S_{1,2} S_{2,3} S_{3,4}
    S_ab = exchange_op(S_a, S_b)
    S_bc = exchange_op(S_b, S_c)
    S_cd = exchange_op(S_c, S_d)

    # 构建逆向循环置换 P^-1 = S_{1,4} S_{4,3} S_{3,2}
    S_ad = exchange_op(S_a, S_d)
    S_dc = exchange_op(S_d, S_c)
    S_cb = exchange_op(S_c, S_b)

    # 组合操作符：P = S_{a,b} S_{b,c} S_{c,d}
    P = S_ab @ S_bc @ S_cd

    # 组合操作符：P^-1 = S_{a,d} S_{d,c} S_{c,b}
    P_inv = S_ad @ S_dc @ S_cb

    return P, P_inv

def compute_spin_dot_product(hilbert, site_i, site_j):
    """计算两个位点间的自旋点积算符 S_i·S_j"""
    # 构建自旋算符
    sx_i = nk.operator.spin.sigmax(hilbert, site_i) * 0.5
    sy_i = nk.operator.spin.sigmay(hilbert, site_i) * 0.5
    sz_i = nk.operator.spin.sigmaz(hilbert, site_i) * 0.5

    sx_j = nk.operator.spin.sigmax(hilbert, site_j) * 0.5
    sy_j = nk.operator.spin.sigmay(hilbert, site_j) * 0.5
    sz_j = nk.operator.spin.sigmaz(hilbert, site_j) * 0.5

    # 计算 S_i·S_j = S^x_i·S^x_j + S^y_i·S^y_j + S^z_i·S^z_j
    return sx_i @ sx_j + sy_i @ sy_j + sz_i @ sz_j

def calculate_dimer_structure_factor(vqs, lattice, L, save_dir, log_file=None):
    """
    准确计算二聚体-二聚体结构因子
    C_d(r) = <[S(0)·S(0+x)][S(r)·S(r+x)]>
    """
    if log_file is None:
        # 默认日志文件将由调用者提供
        log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L}_J2=0.05_J1=0.05.log")
    log_message(log_file, "-"*80)
    log_message(log_file, "开始计算二聚体结构因子...")

    # 创建k点网格
    k_points, _, _ = create_k_mesh(30)

    # 初始化dimer结构因子
    dimer_sf = np.zeros((len(k_points), len(k_points)), dtype=complex)

    # 收集所有水平方向的二聚体键
    dimers = []
    dimer_positions = []  # 存储每个二聚体的中心位置

    for x in range(L):
        for y in range(L):
            for unit_idx in range(4):
                site_i = 4 * (y + x * L) + unit_idx

                # 找到水平相邻位点
                for edge in lattice.edges():
                    if edge[0] == site_i or edge[1] == site_i:
                        site_j = edge[1] if edge[0] == site_i else edge[0]

                        # 检查是否为水平边 (x方向)
                        pos_i = lattice.positions[site_i]
                        pos_j = lattice.positions[site_j]

                        # 如果是水平方向的边
                        if abs(pos_j[0] - pos_i[0]) > 0 and abs(pos_j[1] - pos_i[1]) < 0.1:
                            # 避免重复计算
                            if site_i < site_j:
                                dimers.append((site_i, site_j))
                                # 计算二聚体的中心位置
                                dimer_center = 0.5 * (np.array(pos_i) + np.array(pos_j))
                                dimer_positions.append(dimer_center)

    # 如果没有找到水平方向的二聚体，给出警告
    if len(dimers) == 0:
        log_message(log_file, "警告: 没有找到水平方向的二聚体！")
        return k_points, np.zeros((len(k_points), len(k_points)))

    log_message(log_file, f"找到 {len(dimers)} 个水平方向的二聚体")

    # 准备保存二聚体相关数据
    dimer_data = []

    # 计算所有二聚体对之间的相关函数
    for i, ((i1, i2), pos_i) in enumerate(zip(dimers, dimer_positions)):
        # 创建二聚体算符 S_i1·S_i2
        S_i1_dot_S_i2 = compute_spin_dot_product(vqs.hilbert, i1, i2)

        # 每处理10个二聚体记录一次进度
        if i % 10 == 0:
            log_message(log_file, f"处理二聚体 {i+1}/{len(dimers)}...")

        for j, ((j1, j2), pos_j) in enumerate(zip(dimers, dimer_positions)):
            # 计算位移向量
            r_ij = pos_j - pos_i

            # 创建二聚体算符 S_j1·S_j2
            S_j1_dot_S_j2 = compute_spin_dot_product(vqs.hilbert, j1, j2)

            # 计算二聚体-二聚体相关函数: C_d(r) = <[S(0)·S(0+x)][S(r)·S(r+x)]>
            dimer_corr = vqs.expect(S_i1_dot_S_i2 @ S_j1_dot_S_j2).mean.real

            # 保存数据
            dimer_data.append({
                'dimer_i': (i1, i2), 'dimer_j': (j1, j2),
                'r_x': r_ij[0], 'r_y': r_ij[1],
                'corr': dimer_corr
            })

            # 计算结构因子
            for kx_idx, kx_val in enumerate(k_points):
                for ky_idx, ky_val in enumerate(k_points):
                    k_vec = np.array([kx_val, ky_val])
                    phase = np.exp(1j * np.dot(k_vec, r_ij))
                    dimer_sf[ky_idx, kx_idx] += dimer_corr * phase

    # 归一化
    dimer_sf /= len(dimers)

    # 保存数据
    np.save(os.path.join(save_dir, "dimer_correlation_data.npy"), dimer_data)
    np.save(os.path.join(save_dir, "dimer_structure_factor.npy"), dimer_sf.real)

    log_message(log_file, "二聚体结构因子计算完成")

    return k_points, dimer_sf.real

def calculate_correlation_ratios(k_points, structure_factor, save_dir, type_name, log_file=None):
    """计算相关比率 R = 1 - S(k+δk)/S(k)"""
    if log_file is None:
        # 默认日志文件将由调用者提供
        log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L=4_J2=0.05_J1=0.05.log")
    # 找到结构因子的最大值位置
    max_idx = np.unravel_index(np.argmax(structure_factor), structure_factor.shape)
    k_max_x = k_points[max_idx[1]]
    k_max_y = k_points[max_idx[0]]
    S_max = structure_factor[max_idx]

    # 寻找 k+δk 的位置
    # 近似δk (不直接使用，仅作为注释说明)

    # 找到最接近 k_max + dk 的k点
    kx_idx = max_idx[1]
    ky_idx = max_idx[0]

    if kx_idx + 1 < len(k_points):
        kx_plus_dk = kx_idx + 1
    else:
        kx_plus_dk = kx_idx - 1

    if ky_idx + 1 < len(k_points):
        ky_plus_dk = ky_idx + 1
    else:
        ky_plus_dk = ky_idx - 1

    # 计算 S(k+δk_x)
    S_kxplus = structure_factor[ky_idx, kx_plus_dk]

    # 计算 S(k+δk_y)
    S_kyplus = structure_factor[ky_plus_dk, kx_idx]

    # 取平均
    S_kplus = 0.5 * (S_kxplus + S_kyplus)

    # 计算相关比率
    ratio = 1.0 - S_kplus / S_max

    # 保存结果
    ratio_data = {
        'k_max': (k_max_x, k_max_y),
        'S_max': S_max,
        'S_kplus': S_kplus,
        'ratio': ratio
    }

    np.save(os.path.join(save_dir, f"{type_name}_correlation_ratio.npy"), ratio_data)

    # 记录相关比率信息
    log_message(log_file, "-"*80)
    log_message(log_file, f"{type_name.capitalize()} 相关比率: {ratio:.4f}, 峰值位置: ({k_max_x:.2f}, {k_max_y:.2f})")

    return ratio, (k_max_x, k_max_y)
