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
    """
    计算自旋结构因子 S(k) = ∑_r e^(ik·r) <S_0·S_r>
    优化版本：预计算操作符，使用向量化计算加速傅里叶变换
    """
    if log_file is None:
        # 默认日志文件将由调用者提供
        log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L}_J2=0.05_J1=0.05.log")
        # 尝试从目录名称中提取J2和J1值
        try:
            dir_parts = save_dir.split('/')
            for part in dir_parts:
                if part.startswith('J2='):
                    J2_str = part.split('=')[1]
                    if 'J1=' in dir_parts[dir_parts.index(part)+1]:
                        J1_str = dir_parts[dir_parts.index(part)+1].split('=')[1]
                        log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L}_J2={J2_str}_J1={J1_str}.log")
                        break
        except:
            # 如果提取失败，使用默认值
            pass
    log_message(log_file, "="*80)
    log_message(log_file, "开始计算自旋结构因子...")

    N = lattice.n_nodes

    # 创建k点网格
    k_points, _, _ = create_k_mesh(30)
    n_k = len(k_points)

    # 初始化结构因子
    spin_sf = np.zeros((n_k, n_k), dtype=complex)

    # 预计算所有位点的自旋操作符
    log_message(log_file, "预计算自旋操作符...")
    spin_ops = []
    for i in range(N):
        sx_i = nk.operator.spin.sigmax(vqs.hilbert, i) * 0.5
        sy_i = nk.operator.spin.sigmay(vqs.hilbert, i) * 0.5
        sz_i = nk.operator.spin.sigmaz(vqs.hilbert, i) * 0.5

        # 将操作符转换为JAX操作符
        sx_i = sx_i.to_jax_operator()
        sy_i = sy_i.to_jax_operator()
        sz_i = sz_i.to_jax_operator()

        spin_ops.append((sx_i, sy_i, sz_i))

    # 计算自旋-自旋相关函数并构建结构因子
    correlation_data = []

    # 创建位置矩阵，用于向量化计算
    positions = np.array([lattice.positions[i] for i in range(N)])

    # 计算所有位点对之间的位移向量
    log_message(log_file, "计算位移向量...")
    r_vectors = np.zeros((N, N, 2))
    for i in range(N):
        r_vectors[i] = positions - positions[i]

    # 使用NetKet的sharding功能并行计算相关函数
    log_message(log_file, "计算自旋相关函数...")

    # 分批处理以减少内存使用
    batch_size = 10  # 每批处理的位点数
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        log_message(log_file, f"处理位点 {batch_start}-{batch_end-1}/{N}...")

        for i in range(batch_start, batch_end):
            sx_i, sy_i, sz_i = spin_ops[i]

            for j in range(N):
                # 计算位移向量
                r_ij = r_vectors[i, j]

                # 获取j位点的自旋操作符
                sx_j, sy_j, sz_j = spin_ops[j]

                # 构建自旋点积操作符
                spin_dot_op = sx_i @ sx_j + sy_i @ sy_j + sz_i @ sz_j

                # 计算期望值
                corr_ij = vqs.expect(spin_dot_op).mean.real

                # 保存相关函数数据
                correlation_data.append({
                    'i': i, 'j': j,
                    'r_x': r_ij[0], 'r_y': r_ij[1],
                    'corr': corr_ij
                })

    # 向量化计算傅里叶变换
    log_message(log_file, "计算傅里叶变换...")

    # 提取相关数据用于向量化计算
    r_values = np.array([[data['r_x'], data['r_y']] for data in correlation_data])
    corr_values = np.array([data['corr'] for data in correlation_data])

    # 创建k网格
    kx_grid, ky_grid = np.meshgrid(k_points, k_points)
    k_grid = np.stack([kx_grid.flatten(), ky_grid.flatten()], axis=1)

    # 向量化计算傅里叶变换
    for idx, k_vec in enumerate(k_grid):
        # 计算所有r的相位因子
        phases = np.exp(1j * np.dot(r_values, k_vec))
        # 计算结构因子
        sf_value = np.sum(corr_values * phases)
        # 存储结果
        kx_idx = idx % n_k
        ky_idx = idx // n_k
        spin_sf[ky_idx, kx_idx] = sf_value

    # 归一化
    spin_sf /= N

    # 保存相关函数数据
    np.save(os.path.join(save_dir, "spin_correlation_data.npy"), correlation_data)
    np.save(os.path.join(save_dir, "spin_structure_factor.npy"), spin_sf.real)
    np.save(os.path.join(save_dir, "k_points.npy"), k_points)

    log_message(log_file, "自旋结构因子计算完成")

    return k_points, spin_sf.real

def calculate_plaquette_structure_factor(vqs, lattice, L, save_dir, log_file=None):
    """
    计算简盘结构因子，使用自旋交换算符实现循环置换
    优化版本：预计算操作符，使用向量化计算加速傅里叶变换
    """
    if log_file is None:
        # 默认日志文件将由调用者提供
        log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L}_J2=0.05_J1=0.05.log")
        # 尝试从目录名称中提取J2和J1值
        try:
            dir_parts = save_dir.split('/')
            for part in dir_parts:
                if part.startswith('J2='):
                    J2_str = part.split('=')[1]
                    if 'J1=' in dir_parts[dir_parts.index(part)+1]:
                        J1_str = dir_parts[dir_parts.index(part)+1].split('=')[1]
                        log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L}_J2={J2_str}_J1={J1_str}.log")
                        break
        except:
            # 如果提取失败，使用默认值
            pass
    log_message(log_file, "="*80)
    log_message(log_file, "开始计算简盘结构因子...")

    # 创建k点网格
    k_points, _, _ = create_k_mesh(30)
    n_k = len(k_points)

    # 初始化简盘结构因子
    plaq_sf = np.zeros((n_k, n_k), dtype=complex)

    # 识别所有简盘(每个单元格4个位点形成一个简盘)
    log_message(log_file, "识别所有简盘...")
    plaquettes = []
    plaquette_positions = []  # 存储每个简盘的中心位置

    for x in range(L):
        for y in range(L):
            base = 4 * (y + x * L)
            # 单元格内的四个点，按照左下、右下、右上、左上排列
            plaq = [base, base+1, base+2, base+3]
            plaquettes.append(plaq)

            # 计算简盘中心位置
            pos_x = np.mean([lattice.positions[p][0] for p in plaq])
            pos_y = np.mean([lattice.positions[p][1] for p in plaq])
            plaquette_positions.append(np.array([pos_x, pos_y]))

    # 预计算所有简盘的循环置换操作符
    log_message(log_file, "预计算简盘操作符...")
    plaquette_ops = []

    for i, plaq in enumerate(plaquettes):
        if i % 10 == 0:
            log_message(log_file, f"预计算简盘操作符 {i}/{len(plaquettes)}...")

        # 使用自旋交换算符构建循环置换操作符
        P, P_inv = construct_plaquette_permutation(vqs.hilbert, plaq)
        # 构建操作符 (P + P^-1)
        op = P + P_inv
        plaquette_ops.append(op)

    # 计算所有简盘对之间的位移向量
    log_message(log_file, "计算位移向量...")
    plaquette_positions = np.array(plaquette_positions)
    n_plaq = len(plaquettes)

    # 计算所有简盘对之间的位移向量
    r_vectors = np.zeros((n_plaq, n_plaq, 2))
    for i in range(n_plaq):
        r_vectors[i] = plaquette_positions - plaquette_positions[i]

    # 计算简盘-简盘相关函数
    log_message(log_file, "计算简盘相关函数...")
    plaquette_data = []

    # 分批处理以减少内存使用
    batch_size = 5  # 每批处理的简盘数
    for batch_start in range(0, n_plaq, batch_size):
        batch_end = min(batch_start + batch_size, n_plaq)
        log_message(log_file, f"处理简盘 {batch_start}-{batch_end-1}/{n_plaq}...")

        for i in range(batch_start, batch_end):
            op_i = plaquette_ops[i]

            for j in range(n_plaq):
                # 获取位移向量
                r_ij = r_vectors[i, j]

                # 获取j简盘的操作符
                op_j = plaquette_ops[j]

                # 构建组合操作符
                combined_op = op_i @ op_j

                # 计算期望值
                corr_ij = vqs.expect(combined_op).mean.real / 4.0  # 除以4得到正确的归一化

                # 保存相关函数数据
                plaquette_data.append({
                    'plaq_i': i, 'plaq_j': j,
                    'r_x': r_ij[0], 'r_y': r_ij[1],
                    'corr': corr_ij
                })

    # 向量化计算傅里叶变换
    log_message(log_file, "计算傅里叶变换...")

    # 提取相关数据用于向量化计算
    r_values = np.array([[data['r_x'], data['r_y']] for data in plaquette_data])
    corr_values = np.array([data['corr'] for data in plaquette_data])

    # 创建k网格
    kx_grid, ky_grid = np.meshgrid(k_points, k_points)
    k_grid = np.stack([kx_grid.flatten(), ky_grid.flatten()], axis=1)

    # 向量化计算傅里叶变换
    for idx, k_vec in enumerate(k_grid):
        # 计算所有r的相位因子
        phases = np.exp(1j * np.dot(r_values, k_vec))
        # 计算结构因子
        sf_value = np.sum(corr_values * phases)
        # 存储结果
        kx_idx = idx % n_k
        ky_idx = idx // n_k
        plaq_sf[ky_idx, kx_idx] = sf_value

    # 归一化
    plaq_sf /= n_plaq

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

    # 将所有操作符转换为JAX操作符
    S_a = [op.to_jax_operator() for op in S_a]
    S_b = [op.to_jax_operator() for op in S_b]
    S_c = [op.to_jax_operator() for op in S_c]
    S_d = [op.to_jax_operator() for op in S_d]

    # 构建交换算符 S_{i,j} = (1/2 + 2S_i·S_j)
    def exchange_op(S_i, S_j):
        # 计算 S_i·S_j = S^x_i·S^x_j + S^y_i·S^y_j + S^z_i·S^z_j
        SiSj = S_i[0] @ S_j[0] + S_i[1] @ S_j[1] + S_i[2] @ S_j[2]
        # 返回 1/2 + 2(S_i·S_j)
        constant_op = nk.operator.LocalOperator(hilbert, constant=0.5).to_jax_operator()
        return constant_op + 2.0 * SiSj

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

    # 将操作符转换为JAX操作符
    sx_i = sx_i.to_jax_operator()
    sy_i = sy_i.to_jax_operator()
    sz_i = sz_i.to_jax_operator()

    sx_j = sx_j.to_jax_operator()
    sy_j = sy_j.to_jax_operator()
    sz_j = sz_j.to_jax_operator()

    # 计算 S_i·S_j = S^x_i·S^x_j + S^y_i·S^y_j + S^z_i·S^z_j
    return sx_i @ sx_j + sy_i @ sy_j + sz_i @ sz_j

def calculate_dimer_structure_factor(vqs, lattice, L, save_dir, log_file=None):
    """
    准确计算二聚体-二聚体结构因子
    C_d(r) = <[S(0)·S(0+x)][S(r)·S(r+x)]>
    优化版本：预计算操作符，使用向量化计算加速傅里叶变换
    """
    if log_file is None:
        # 默认日志文件将由调用者提供
        log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L}_J2=0.05_J1=0.05.log")
        # 尝试从目录名称中提取J2和J1值
        try:
            dir_parts = save_dir.split('/')
            for part in dir_parts:
                if part.startswith('J2='):
                    J2_str = part.split('=')[1]
                    if 'J1=' in dir_parts[dir_parts.index(part)+1]:
                        J1_str = dir_parts[dir_parts.index(part)+1].split('=')[1]
                        log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L}_J2={J2_str}_J1={J1_str}.log")
                        break
        except:
            # 如果提取失败，使用默认值
            pass
    log_message(log_file, "="*80)
    log_message(log_file, "开始计算二聚体结构因子...")

    # 创建k点网格
    k_points, _, _ = create_k_mesh(30)
    n_k = len(k_points)

    # 初始化dimer结构因子
    dimer_sf = np.zeros((n_k, n_k), dtype=complex)

    # 收集所有水平方向的二聚体键
    log_message(log_file, "识别水平方向的二聚体...")
    dimers = []
    dimer_positions = []  # 存储每个二聚体的中心位置

    # 预先获取所有边
    edges = list(lattice.edges())

    for x in range(L):
        for y in range(L):
            for unit_idx in range(4):
                site_i = 4 * (y + x * L) + unit_idx

                # 找到水平相邻位点
                for edge in edges:
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
        return k_points, np.zeros((n_k, n_k))

    log_message(log_file, f"找到 {len(dimers)} 个水平方向的二聚体")
    n_dimers = len(dimers)

    # 预计算所有二聚体的自旋点积操作符
    log_message(log_file, "预计算二聚体操作符...")
    dimer_ops = []

    for i, (i1, i2) in enumerate(dimers):
        if i % 10 == 0:
            log_message(log_file, f"预计算二聚体操作符 {i}/{n_dimers}...")

        # 创建二聚体算符 S_i1·S_i2
        S_i1_dot_S_i2 = compute_spin_dot_product(vqs.hilbert, i1, i2)
        dimer_ops.append(S_i1_dot_S_i2)

    # 计算所有二聚体对之间的位移向量
    log_message(log_file, "计算位移向量...")
    dimer_positions = np.array(dimer_positions)

    # 计算所有二聚体对之间的位移向量
    r_vectors = np.zeros((n_dimers, n_dimers, 2))
    for i in range(n_dimers):
        r_vectors[i] = dimer_positions - dimer_positions[i]

    # 计算二聚体-二聚体相关函数
    log_message(log_file, "计算二聚体相关函数...")
    dimer_data = []

    # 分批处理以减少内存使用
    batch_size = 10  # 每批处理的二聚体数
    for batch_start in range(0, n_dimers, batch_size):
        batch_end = min(batch_start + batch_size, n_dimers)
        log_message(log_file, f"处理二聚体 {batch_start}-{batch_end-1}/{n_dimers}...")

        for i in range(batch_start, batch_end):
            op_i = dimer_ops[i]
            dimer_i = dimers[i]

            for j in range(n_dimers):
                # 获取位移向量
                r_ij = r_vectors[i, j]

                # 获取j二聚体的操作符
                op_j = dimer_ops[j]
                dimer_j = dimers[j]

                # 计算二聚体-二聚体相关函数: C_d(r) = <[S(0)·S(0+x)][S(r)·S(r+x)]>
                dimer_corr = vqs.expect(op_i @ op_j).mean.real

                # 保存数据
                dimer_data.append({
                    'dimer_i': dimer_i, 'dimer_j': dimer_j,
                    'r_x': r_ij[0], 'r_y': r_ij[1],
                    'corr': dimer_corr
                })

    # 向量化计算傅里叶变换
    log_message(log_file, "计算傅里叶变换...")

    # 提取相关数据用于向量化计算
    r_values = np.array([[data['r_x'], data['r_y']] for data in dimer_data])
    corr_values = np.array([data['corr'] for data in dimer_data])

    # 创建k网格
    kx_grid, ky_grid = np.meshgrid(k_points, k_points)
    k_grid = np.stack([kx_grid.flatten(), ky_grid.flatten()], axis=1)

    # 向量化计算傅里叶变换
    for idx, k_vec in enumerate(k_grid):
        # 计算所有r的相位因子
        phases = np.exp(1j * np.dot(r_values, k_vec))
        # 计算结构因子
        sf_value = np.sum(corr_values * phases)
        # 存储结果
        kx_idx = idx % n_k
        ky_idx = idx // n_k
        dimer_sf[ky_idx, kx_idx] = sf_value

    # 归一化
    dimer_sf /= n_dimers

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
        # 尝试从目录名称中提取L、J2和J1值
        try:
            dir_parts = save_dir.split('/')
            for part in dir_parts:
                if part.startswith('L='):
                    L_str = part.split('=')[1]
                    if 'J2=' in dir_parts[dir_parts.index(part)+1]:
                        J2_str = dir_parts[dir_parts.index(part)+1].split('=')[1]
                        if 'J1=' in dir_parts[dir_parts.index(part)+2]:
                            J1_str = dir_parts[dir_parts.index(part)+2].split('=')[1]
                            log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L_str}_J2={J2_str}_J1={J1_str}.log")
                            break
        except:
            # 如果提取失败，使用默认值
            pass
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


def calculate_af_order_parameter(k_points, spin_sf, L, save_dir, log_file=None, spin_data=None):
    """
    计算反铁磁序参数 (AF Order Parameter)：m^2(L) = S(π, π)/L^2

    参数:
    - k_points: k点网格
    - spin_sf: 自旋结构因子
    - L: 系统大小
    - save_dir: 保存目录
    - log_file: 日志文件
    - spin_data: 自旋相关函数数据列表（可选，保持与其他序参量函数接口一致，但在此函数中未使用）
    """
    if log_file is None:
        # 默认日志文件将由调用者提供
        log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L}_J2=0.05_J1=0.05.log")
        # 尝试从目录名称中提取J2和J1值
        try:
            dir_parts = save_dir.split('/')
            for part in dir_parts:
                if part.startswith('J2='):
                    J2_str = part.split('=')[1]
                    if 'J1=' in dir_parts[dir_parts.index(part)+1]:
                        J1_str = dir_parts[dir_parts.index(part)+1].split('=')[1]
                        log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L}_J2={J2_str}_J1={J1_str}.log")
                        break
        except:
            # 如果提取失败，使用默认值
            pass

    log_message(log_file, "-"*80)
    log_message(log_file, "计算反铁磁序参数...")

    # 找到最接近 (-π, -π) 的k点，对应反铁磁波矢
    neg_pi_idx = np.argmin(np.abs(k_points + np.pi))

    # 获取 S(π, π) 的值
    # 由于k_points是从-π到π，所以(-π, -π)对应的是反铁磁波矢
    S_pi_pi = spin_sf[neg_pi_idx, neg_pi_idx]

    # 计算反铁磁序参数
    m_squared = S_pi_pi / (L * L)

    # 保存结果
    af_data = {
        'S_pi_pi': S_pi_pi,
        'm_squared': m_squared,
        'L': L
    }

    np.save(os.path.join(save_dir, "af_order_parameter.npy"), af_data)

    # 记录反铁磁序参数信息
    log_message(log_file, f"反铁磁序参数 m^2(L) = {m_squared:.6f}")
    log_message(log_file, f"S(π, π) = {S_pi_pi:.6f}")

    return m_squared


def calculate_plaquette_order_parameter(plaquette_data, L, save_dir, log_file=None):
    """
    计算简盘序参量 (Plaquette Order Parameter)：m_p(L) = |C(L/2, L/2) - C(L/2 - 1, L/2 - 1)|

    参数:
    - plaquette_data: 简盘相关函数数据列表
    - L: 系统大小
    - save_dir: 保存目录
    - log_file: 日志文件
    """
    if log_file is None:
        # 默认日志文件将由调用者提供
        log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L}_J2=0.05_J1=0.05.log")
        # 尝试从目录名称中提取J2和J1值
        try:
            dir_parts = save_dir.split('/')
            for part in dir_parts:
                if part.startswith('J2='):
                    J2_str = part.split('=')[1]
                    if 'J1=' in dir_parts[dir_parts.index(part)+1]:
                        J1_str = dir_parts[dir_parts.index(part)+1].split('=')[1]
                        log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L}_J2={J2_str}_J1={J1_str}.log")
                        break
        except:
            # 如果提取失败，使用默认值
            pass

    log_message(log_file, "-"*80)
    log_message(log_file, "计算简盘序参量...")

    # 从plaquette_data中提取相关函数
    # 寻找位置接近 (L/2, L/2) 和 (L/2-1, L/2-1) 的数据点
    C_L2_L2 = None
    C_L2m1_L2m1 = None

    target_r1 = np.array([L/2, L/2])
    target_r2 = np.array([L/2 - 1, L/2 - 1])

    min_dist1 = float('inf')
    min_dist2 = float('inf')

    for data in plaquette_data:
        r = np.array([data['r_x'], data['r_y']])

        # 计算与目标位置的距离
        dist1 = np.linalg.norm(r - target_r1)
        dist2 = np.linalg.norm(r - target_r2)

        # 更新最接近 (L/2, L/2) 的点
        if dist1 < min_dist1:
            min_dist1 = dist1
            C_L2_L2 = data['corr']

        # 更新最接近 (L/2-1, L/2-1) 的点
        if dist2 < min_dist2:
            min_dist2 = dist2
            C_L2m1_L2m1 = data['corr']

    # 如果找不到合适的点，给出警告
    if C_L2_L2 is None or C_L2m1_L2m1 is None:
        log_message(log_file, "警告: 无法找到合适的相关函数数据点来计算简盘序参量")
        return 0.0

    # 计算简盘序参量
    m_p = abs(C_L2_L2 - C_L2m1_L2m1)

    # 保存结果
    plaq_order_data = {
        'C_L2_L2': C_L2_L2,
        'C_L2m1_L2m1': C_L2m1_L2m1,
        'm_p': m_p,
        'L': L
    }

    np.save(os.path.join(save_dir, "plaquette_order_parameter.npy"), plaq_order_data)

    # 记录简盘序参量信息
    log_message(log_file, f"简盘序参量 m_p(L) = {m_p:.6f}")
    log_message(log_file, f"C(L/2, L/2) = {C_L2_L2:.6f}, C(L/2-1, L/2-1) = {C_L2m1_L2m1:.6f}")

    return m_p


def calculate_dimer_order_parameter(dimer_data, L, save_dir, log_file=None):
    """
    计算二聚体序参量 (Dimer Order Parameter)：D^2 = \frac{1}{N} \sum_r C_d(r)(-1)^{r_x}

    参数:
    - dimer_data: 二聚体相关函数数据列表
    - L: 系统大小
    - save_dir: 保存目录
    - log_file: 日志文件
    """
    if log_file is None:
        # 默认日志文件将由调用者提供
        log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L}_J2=0.05_J1=0.05.log")
        # 尝试从目录名称中提取J2和J1值
        try:
            dir_parts = save_dir.split('/')
            for part in dir_parts:
                if part.startswith('J2='):
                    J2_str = part.split('=')[1]
                    if 'J1=' in dir_parts[dir_parts.index(part)+1]:
                        J1_str = dir_parts[dir_parts.index(part)+1].split('=')[1]
                        log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L}_J2={J2_str}_J1={J1_str}.log")
                        break
        except:
            # 如果提取失败，使用默认值
            pass

    log_message(log_file, "-"*80)
    log_message(log_file, "计算二聚体序参量...")

    # 计算二聚体序参量
    D_squared_sum = 0.0
    count = 0

    for data in dimer_data:
        r_x = data['r_x']
        # 计算 (-1)^{r_x} 的符号
        # 由于r_x可能是浮点数，我们需要四舍五入到最接近的整数
        sign = (-1) ** int(round(r_x))

        # 累加 C_d(r) * (-1)^{r_x}
        D_squared_sum += data['corr'] * sign
        count += 1

    # 归一化
    if count > 0:
        D_squared = D_squared_sum / count
    else:
        D_squared = 0.0
        log_message(log_file, "警告: 没有二聚体相关函数数据来计算二聚体序参量")

    # 保存结果
    dimer_order_data = {
        'D_squared': D_squared,
        'L': L
    }

    np.save(os.path.join(save_dir, "dimer_order_parameter.npy"), dimer_order_data)

    # 记录二聚体序参量信息
    log_message(log_file, f"二聚体序参量 D^2 = {D_squared:.6f}")

    return D_squared
