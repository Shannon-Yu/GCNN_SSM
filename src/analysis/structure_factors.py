import numpy as np
import netket as nk
import os
import jax
import jax.numpy as jnp
from functools import partial
from jax import vmap
from src.utils.logging import log_message

def create_k_mesh(lattice):
    """
    创建k点网格，根据晶格尺寸设置点数，使用2π/L*n的方式生成k点

    参数:
    - lattice: 晶格对象，从中获取Lx和Ly

    返回:
    - k_points_x: x方向的k点
    - k_points_y: y方向的k点
    - kx, ky: 网格化的k点
    """

    # 从晶格获取尺寸
    Lx, Ly = lattice.extent


    # 按照标准定义生成k点：k = 2π/L(n, m)，其中n和m取值从0到L-1
    # 在Shastry-Sutherland模型中，每个简盘在水平和垂直方向各占2个点
    # 所以实际格点数是简盘数的2倍，我们需要使用2*L作为分母

    # 生成k点，范围为[0, 2π]，确保包含端点
    # 使用linspace生成均匀分布的点，包括0和2π
    k_points_x = np.linspace(0, 2*np.pi, 2*Lx+1)
    k_points_y = np.linspace(0, 2*np.pi, 2*Ly+1)
    kx, ky = np.meshgrid(k_points_x, k_points_y)

    return k_points_x, k_points_y, kx, ky

########################################################
# 计算spin structure factor
########################################################
def calculate_spin_structure_factor(vqs, lattice, L, save_dir, log_file=None):
    """
    计算自旋结构因子 S(k) = ∑_r e^(ik·r) <S_0·S_r>
    优化版本：预计算操作符，使用向量化计算加速傅里叶变换
    """
    dir_parts = save_dir.split('/')
    for part in dir_parts:
        if part.startswith('J2='):
            J2_str = part.split('=')[1]
            if 'J1=' in dir_parts[dir_parts.index(part)+1]:
                J1_str = dir_parts[dir_parts.index(part)+1].split('=')[1]
                log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L}_J2={J2_str}_J1={J1_str}.log")

    log_message(log_file, "开始计算自旋结构因子...")

    N = lattice.n_nodes

    # 创建k点网格
    k_points_x, k_points_y, kx_grid, ky_grid = create_k_mesh(lattice)
    n_kx = len(k_points_x)
    n_ky = len(k_points_y)

    # 初始化结构因子
    spin_sf = np.zeros((n_ky, n_kx), dtype=complex)

    # 预计算所有位点的自旋操作符
    log_message(log_file, "预计算自旋操作符...")
    spin_ops = []

    # 计算进度更新间隔
    update_interval = 10  # 每10个记录一次进度

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

    # 只计算位点0与其他位点的相关函数

    # 获取位点0的自旋操作符
    sx_0, sy_0, sz_0 = spin_ops[0]

    # 计算位点0与所有位点的相关函数
    correlation_data = []
    ops_list = []
    for j in range(N):
        sx_j, sy_j, sz_j = spin_ops[j]
        # 构建自旋点积操作符
        spin_dot_op = sx_0 @ sx_j + sy_0 @ sy_j + sz_0 @ sz_j
        ops_list.append(spin_dot_op)

    # 逐个计算操作符的期望值
    log_message(log_file, "使用自旋0作为参考点, 计算自旋相关函数...")
    for j, op in enumerate(ops_list):
        # 计算单个操作符的期望值
        corr = vqs.expect(op)
        r_0j = r_vectors[0, j]
        correlation_data.append({
            'i': 0, 'j': j,
            'r_x': r_0j[0], 'r_y': r_0j[1],
            'corr': corr.mean.real
        })
        if j % update_interval == 0 or j == len(ops_list) - 1:
            log_message(log_file, f"计算相关函数进度: {j+1}/{len(ops_list)}")

    # 将所有结果合并
    all_correlations = correlation_data

    # 将所有结果合并
    correlation_data = all_correlations

    # 向量化计算傅里叶变换
    log_message(log_file, "计算傅里叶变换...")

    # 提取相关数据用于向量化计算
    r_values = np.array([[data['r_x'], data['r_y']] for data in correlation_data])
    corr_values = np.array([data['corr'] for data in correlation_data])

    # 使用已经创建的k网格
    k_grid = np.stack([kx_grid.flatten(), ky_grid.flatten()], axis=1)

    # 使用JAX的向量化功能计算傅里叶变换

    # 将数据转换为JAX数组
    r_values_jax = jnp.array(r_values)
    corr_values_jax = jnp.array(corr_values)
    k_grid_jax = jnp.array(k_grid)

    # 定义计算单个k点的结构因子的函数
    def compute_sf_for_k(k_vec):
        # 计算所有r的相位因子
        phases = jnp.exp(1j * jnp.dot(r_values_jax, k_vec))
        # 计算结构因子
        return jnp.sum(corr_values_jax * phases)

    # 向量化函数以并行计算所有k点
    compute_sf_vmap = vmap(compute_sf_for_k)

    # 并行计算所有k点的结构因子
    sf_values = compute_sf_vmap(k_grid_jax)

    # 将结果重塑为2D网格
    sf_values_2d = sf_values.reshape(n_ky, n_kx)

    # 存储结果
    spin_sf = np.array(sf_values_2d)

    # 归一化
    spin_sf /= N

    # 创建完整的自旋数据存储结构
    spin_data_storage = {
        'k_points': {
            'kx': k_points_x,
            'ky': k_points_y
        },
        'correlation_data': correlation_data,
        'structure_factor': spin_sf.real
    }
    
    # 保存数据
    np.save(os.path.join(save_dir, "spin_data.npy"), spin_data_storage)

    log_message(log_file, "自旋结构因子计算完成")

    return (k_points_x, k_points_y), spin_sf.real


def calculate_dimer_structure_factor(vqs, lattice, L, save_dir, log_file=None):
    """
    准确计算二聚体-二聚体结构因子
    C_d(r) = <[S(0)·S(0+x)][S(r)·S(r+x)]> 和 <[S(0)·S(0+y)][S(r)·S(r+y)]>
    分别处理x和y方向的二聚体，使用总位点数N进行归一化
    优化版本：预计算操作符，使用向量化计算加速傅里叶变换
    """

    dir_parts = save_dir.split('/')
    for part in dir_parts:
        if part.startswith('J2='):
            J2_str = part.split('=')[1]
            if 'J1=' in dir_parts[dir_parts.index(part)+1]:
                J1_str = dir_parts[dir_parts.index(part)+1].split('=')[1]
                log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L}_J2={J2_str}_J1={J1_str}.log")
    log_message(log_file, "开始计算二聚体结构因子...")

    # 获取总位点数
    N = lattice.n_nodes

    # 创建k点网格
    k_points_x, k_points_y, kx_grid, ky_grid = create_k_mesh(lattice)
    n_kx = len(k_points_x)
    n_ky = len(k_points_y)

    # 初始化dimer结构因子
    dimer_sf = np.zeros((n_ky, n_kx), dtype=complex)

    # 收集所有方向的二聚体键（分别处理x和y方向）
    log_message(log_file, "识别x和y方向的二聚体...")
    dimers_x = []  # x方向的二聚体
    dimers_y = []  # y方向的二聚体
    dimer_positions_x = []  # 存储每个x方向二聚体的中心位置
    dimer_positions_y = []  # 存储每个y方向二聚体的中心位置

    # 预先获取所有边
    edges = list(lattice.edges())

    # 获取晶格尺寸
    Lx, Ly = lattice.extent

    for x in range(Lx):
        for y in range(Ly):
            for unit_idx in range(4):
                site_i = 4 * (y + x * Ly) + unit_idx  # 修正：使用Ly而不是L

                # 找到相邻位点
                for edge in edges:
                    if edge[0] == site_i or edge[1] == site_i:
                        site_j = edge[1] if edge[0] == site_i else edge[0]

                        # 获取位置
                        pos_i = lattice.positions[site_i]
                        pos_j = lattice.positions[site_j]

                        # 检查是否为水平边 (x方向)
                        if abs(pos_j[0] - pos_i[0]) > 0 and abs(pos_j[1] - pos_i[1]) < 0.1:
                            # 避免重复计算
                            if site_i < site_j:
                                dimers_x.append((site_i, site_j))
                                # 计算二聚体的中心位置
                                dimer_center = 0.5 * (np.array(pos_i) + np.array(pos_j))
                                dimer_positions_x.append(dimer_center)

                        # 检查是否为垂直边 (y方向)
                        elif abs(pos_j[1] - pos_i[1]) > 0 and abs(pos_j[0] - pos_i[0]) < 0.1:
                            # 避免重复计算
                            if site_i < site_j:
                                dimers_y.append((site_i, site_j))
                                # 计算二聚体的中心位置
                                dimer_center = 0.5 * (np.array(pos_i) + np.array(pos_j))
                                dimer_positions_y.append(dimer_center)

    # 合并所有二聚体
    dimers = dimers_x + dimers_y
    dimer_positions = dimer_positions_x + dimer_positions_y

    # 记录x和y方向二聚体的数量
    n_dimers_x = len(dimers_x)
    n_dimers_y = len(dimers_y)

    # 如果没有找到二聚体，给出警告
    if len(dimers) == 0:
        log_message(log_file, "警告: 没有找到二聚体！")
        return (k_points_x, k_points_y), np.zeros((n_ky, n_kx))

    n_dimers = len(dimers)

    # 预计算所有二聚体的自旋点积操作符
    log_message(log_file, "预计算二聚体操作符...")
    

    spin_ops = []

    # 计算进度更新间隔
    nodes_update_interval = 10  # 每10个记录一次进度

    for i in range(lattice.n_nodes):
        sx_i = nk.operator.spin.sigmax(vqs.hilbert, i) * 0.5
        sy_i = nk.operator.spin.sigmay(vqs.hilbert, i) * 0.5
        sz_i = nk.operator.spin.sigmaz(vqs.hilbert, i) * 0.5

        # 将操作符转换为JAX操作符
        sx_i = sx_i.to_jax_operator()
        sy_i = sy_i.to_jax_operator()
        sz_i = sz_i.to_jax_operator()

        spin_ops.append((sx_i, sy_i, sz_i))


    # 计算进度更新间隔
    
    dimer_ops = []
    for i, (i1, i2) in enumerate(dimers):
        # 创建二聚体算符 S_i1·S_i2，使用预计算的自旋操作符
        sx_i1, sy_i1, sz_i1 = spin_ops[i1]
        sx_i2, sy_i2, sz_i2 = spin_ops[i2]
        # 计算 S_i1·S_i2 = S^x_i1·S^x_i2 + S^y_i1·S^y_i2 + S^z_i1·S^z_i2
        S_i1_dot_S_i2 = sx_i1 @ sx_i2 + sy_i1 @ sy_i2 + sz_i1 @ sz_i2
        dimer_ops.append(S_i1_dot_S_i2)


    # 计算所有二聚体对之间的位移向量
    log_message(log_file, "计算位移向量...")
    dimer_positions = np.array(dimer_positions)

    # 计算所有二聚体对之间的位移向量
    r_vectors = np.zeros((n_dimers, n_dimers, 2))
    for i in range(n_dimers):
        r_vectors[i] = dimer_positions - dimer_positions[i]

    # 计算二聚体-二聚体相关函数，分别处理x方向和y方向
    dimer_data = []
    
    update_interval = 10  # 每10个记录一次进度
    # 分别处理x方向和y方向的二聚体
    if n_dimers_x > 0:
        # 选择第一个x方向二聚体作为x方向参考点
        reference_dimer_x = 0  # dimers_x列表中的第一个
        op_ref_x = dimer_ops[reference_dimer_x]
        dimer_ref_x = dimers[reference_dimer_x]
        log_message(log_file, f"使用二聚体 {dimer_ref_x} 作为参考点，计算二聚体相关函数...")      

        # 预先构建所有x方向二聚体的操作符
        ops_list_x = []
        for j in range(n_dimers_x):
            op_j = dimer_ops[j]
            # 构建组合操作符
            combined_op = op_ref_x @ op_j
            ops_list_x.append((j, combined_op))

        # 直接计算所有x方向操作符的期望值
        for idx, (j, op) in enumerate(ops_list_x):
            # 计算单个操作符的期望值
            corr = vqs.expect(op)
            r_ij = r_vectors[reference_dimer_x, j]
            dimer_j = dimers[j]

            # 保存结果
            dimer_data.append({
                'dimer_i': dimer_ref_x, 'dimer_j': dimer_j,
                'r_x': r_ij[0], 'r_y': r_ij[1],
                'corr': corr.mean.real,
                'direction': 'x'
            })
            if idx % update_interval == 0 or idx == len(ops_list_x) - 1:
                log_message(log_file, f"计算x方向相关函数进度: {idx+1}/{len(ops_list_x)}")

    if n_dimers_y > 0:
        # 选择第一个y方向二聚体作为y方向参考点
        reference_dimer_y = n_dimers_x  # dimers列表中的第一个y方向二聚体
        op_ref_y = dimer_ops[reference_dimer_y]
        dimer_ref_y = dimers[reference_dimer_y]
        log_message(log_file, f"使用二聚体 {dimer_ref_y} 作为参考点，计算二聚体相关函数...")      

        # 预先构建所有y方向二聚体的操作符
        ops_list_y = []
        for j in range(n_dimers_x, n_dimers):
            op_j = dimer_ops[j]
            # 构建组合操作符
            combined_op = op_ref_y @ op_j
            ops_list_y.append((j, combined_op))

        # 直接计算所有y方向操作符的期望值
        for idx, (j, op) in enumerate(ops_list_y):
            # 计算单个操作符的期望值
            corr = vqs.expect(op)
            r_ij = r_vectors[reference_dimer_y, j]
            dimer_j = dimers[j]

            # 保存结果
            dimer_data.append({
                'dimer_i': dimer_ref_y, 'dimer_j': dimer_j,
                'r_x': r_ij[0], 'r_y': r_ij[1],
                'corr': corr.mean.real,
                'direction': 'y'
            })
            if idx % update_interval == 0 or idx == len(ops_list_y) - 1:
                log_message(log_file, f"计算y方向相关函数进度: {idx+1}/{len(ops_list_y)}")

    # 向量化计算傅里叶变换
    log_message(log_file, "计算傅里叶变换...")

    # 提取相关数据用于向量化计算
    r_values = np.array([[data['r_x'], data['r_y']] for data in dimer_data])
    corr_values = np.array([data['corr'] for data in dimer_data])

    # 使用已经创建的k网格
    k_grid = np.stack([kx_grid.flatten(), ky_grid.flatten()], axis=1)

    # 使用JAX的向量化功能计算傅里叶变换

    # 将数据转换为JAX数组
    r_values_jax = jnp.array(r_values)
    corr_values_jax = jnp.array(corr_values)
    k_grid_jax = jnp.array(k_grid)

    # 定义计算单个k点的结构因子的函数
    def compute_sf_for_k(k_vec):
        # 计算所有r的相位因子
        phases = jnp.exp(1j * jnp.dot(r_values_jax, k_vec))
        # 计算结构因子
        return jnp.sum(corr_values_jax * phases)

    # 向量化函数以并行计算所有k点
    compute_sf_vmap = vmap(compute_sf_for_k)

    # 并行计算所有k点的结构因子
    sf_values = compute_sf_vmap(k_grid_jax)

    # 将结果重塑为2D网格
    sf_values_2d = sf_values.reshape(n_ky, n_kx)

    # 存储结果
    dimer_sf = np.array(sf_values_2d)

    # 使用总位点数N进行归一化，而不是二聚体数量
    dimer_sf /= N

    # 创建完整的二聚体数据存储结构
    dimer_data_storage = {
        'k_points': {
            'kx': k_points_x,
            'ky': k_points_y
        },
        'correlation_data': dimer_data,
        'structure_factor': dimer_sf.real
    }
    
    # 保存数据
    np.save(os.path.join(save_dir, "dimer_data.npy"), dimer_data_storage)

    log_message(log_file, "二聚体结构因子计算完成")

    return (k_points_x, k_points_y), dimer_sf.real

########################################################
# 计算plaquette structure factor
########################################################
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

def calculate_plaquette_structure_factor(vqs, lattice, L, save_dir, log_file=None):
    """
    计算简盘结构因子，使用自旋交换算符实现循环置换
    优化版本：预计算操作符，使用向量化计算加速傅里叶变换
    """

    # 尝试从目录名称中提取J2和J1值

    dir_parts = save_dir.split('/')
    for part in dir_parts:
        if part.startswith('J2='):
            J2_str = part.split('=')[1]
            if 'J1=' in dir_parts[dir_parts.index(part)+1]:
                J1_str = dir_parts[dir_parts.index(part)+1].split('=')[1]
                log_file = os.path.join(os.path.dirname(save_dir), f"analyze_L={L}_J2={J2_str}_J1={J1_str}.log")
    log_message(log_file, "开始计算简盘结构因子...")

    # 创建k点网格
    k_points_x, k_points_y, kx_grid, ky_grid = create_k_mesh(lattice)
    n_kx = len(k_points_x)
    n_ky = len(k_points_y)

    # 初始化简盘结构因子
    plaq_sf = np.zeros((n_ky, n_kx), dtype=complex)

    # 识别所有简盘(每个单元格4个位点形成一个简盘)
    log_message(log_file, "识别所有简盘...")
    plaquettes = []
    plaquette_positions = []  # 存储每个简盘的中心位置

    # 获取晶格尺寸
    Lx, Ly = lattice.extent

    for x in range(Lx):
        for y in range(Ly):
            base = 4 * (y + x * Ly)  # 修正：使用Ly而不是L
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

    # 计算进度更新间隔
    update_interval = 10  # 每10个记录一次进度

    for i, plaq in enumerate(plaquettes):
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

    # 计算简盘-简盘相关函数，使用0点位优化

    # 选择第一个简盘作为参考点
    reference_plaq = 0
    op_ref = plaquette_ops[reference_plaq]
    log_message(log_file, f"使用简盘 {reference_plaq} 作为参考点，计算简盘相关函数...")

    # 预先构建所有j的操作符
    ops_list = []
    for j in range(n_plaq):
        op_j = plaquette_ops[j]
        # 构建组合操作符
        combined_op = op_ref @ op_j
        ops_list.append(combined_op)

    # 直接计算所有操作符的期望值
    plaquette_data = []

    # 计算进度更新间隔
    update_interval = 10  # 每10个记录一次进度

    # 直接计算所有操作符的期望值
    for j, op in enumerate(ops_list):
        # 计算单个操作符的期望值
        corr = vqs.expect(op)
        r_ij = r_vectors[reference_plaq, j]

        # 保存结果
        plaquette_data.append({
            'plaq_i': reference_plaq, 'plaq_j': j,
            'r_x': r_ij[0], 'r_y': r_ij[1],
            'corr': corr.mean.real / 4.0  # 除以4得到正确的归一化
        })
        if j % update_interval == 0 or j == len(ops_list) - 1:
            log_message(log_file, f"计算相关函数进度: {j+1}/{len(ops_list)}")

    # 向量化计算傅里叶变换
    log_message(log_file, "计算傅里叶变换...")

    # 提取相关数据用于向量化计算
    r_values = np.array([[data['r_x'], data['r_y']] for data in plaquette_data])
    corr_values = np.array([data['corr'] for data in plaquette_data])

    # 使用已经创建的k网格
    k_grid = np.stack([kx_grid.flatten(), ky_grid.flatten()], axis=1)

    # 使用JAX的向量化功能计算傅里叶变换

    # 将数据转换为JAX数组
    r_values_jax = jnp.array(r_values)
    corr_values_jax = jnp.array(corr_values)
    k_grid_jax = jnp.array(k_grid)

    # 定义计算单个k点的结构因子的函数
    def compute_sf_for_k(k_vec):
        # 计算所有r的相位因子
        phases = jnp.exp(1j * jnp.dot(r_values_jax, k_vec))
        # 计算结构因子
        return jnp.sum(corr_values_jax * phases)

    # 向量化函数以并行计算所有k点
    compute_sf_vmap = vmap(compute_sf_for_k)

    # 并行计算所有k点的结构因子
    sf_values = compute_sf_vmap(k_grid_jax)

    # 将结果重塑为2D网格
    sf_values_2d = sf_values.reshape(n_ky, n_kx)

    # 存储结果
    plaq_sf = np.array(sf_values_2d)

    # 获取总位点数
    N = lattice.n_nodes

    # 归一化，使用总位点数N进行归一化，与自旋和二聚体结构因子保持一致
    plaq_sf /= N

    # 创建完整的简盘数据存储结构
    plaquette_data_storage = {
        'k_points': {
            'kx': k_points_x,
            'ky': k_points_y
        },
        'correlation_data': plaquette_data,
        'structure_factor': plaq_sf.real
    }
    
    # 保存数据
    np.save(os.path.join(save_dir, "plaquette_data.npy"), plaquette_data_storage)

    log_message(log_file, "简盘结构因子计算完成")

    return (k_points_x, k_points_y), plaq_sf.real







