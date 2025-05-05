import numpy as np
import netket as nk
import os
import jax
import jax.numpy as jnp
from functools import partial
from jax import vmap
from src.utils.logging import log_message

def create_k_mesh(lattice=None, L=None):
    """
    创建k点网格，根据晶格尺寸设置点数，使用2π/L*n的方式生成k点

    参数:
    - lattice: 晶格对象，如果提供则从中获取Lx和Ly
    - L: 如果未提供晶格，则使用此参数作为Lx=Ly=L

    返回:
    - k_points_x: x方向的k点
    - k_points_y: y方向的k点
    - kx, ky: 网格化的k点
    """
    if lattice is not None:
        # 从晶格获取尺寸
        Lx, Ly = lattice.extent
    elif L is not None:
        # 使用提供的L值
        Lx, Ly = L, L
    else:
        # 默认值
        Lx, Ly = 20, 20

    # 按照标准定义生成k点：k = 2π/L(n, m)，其中n和m取值从0到L-1
    # 在Shastry-Sutherland模型中，每个简盘在水平和垂直方向各占2个点
    # 所以实际格点数是简盘数的2倍，我们需要使用2*L作为分母

    # 生成k点，范围为[0, 2π]，确保包含端点
    # 使用linspace生成均匀分布的点，包括0和2π
    k_points_x = np.linspace(0, 2*np.pi, 2*Lx+1)
    k_points_y = np.linspace(0, 2*np.pi, 2*Ly+1)

    # 打印k点网格大小，用于调试
    print(f"k点网格大小: x方向 {len(k_points_x)}个点, y方向 {len(k_points_y)}个点")

    kx, ky = np.meshgrid(k_points_x, k_points_y)

    return k_points_x, k_points_y, kx, ky

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
    log_message(log_file, "-"*80)
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

        # 定期更新进度
        if i % update_interval == 0 or i == N - 1:
            log_message(log_file, f"预计算自旋操作符进度: {i+1}/{N}")

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
    log_message(log_file, "计算自旋相关函数...")

    # 获取位点0的自旋操作符
    sx_0, sy_0, sz_0 = spin_ops[0]

    # 计算位点0与所有位点的相关函数
    correlation_data = []

    # 预先构建所有j的操作符
    ops_list = []
    for j in range(N):
        sx_j, sy_j, sz_j = spin_ops[j]
        # 构建自旋点积操作符
        spin_dot_op = sx_0 @ sx_j + sy_0 @ sy_j + sz_0 @ sz_j
        ops_list.append(spin_dot_op)

    # 逐个计算操作符的期望值
    log_message(log_file, f"计算位点0与其他位点的相关函数...")

    # 计算进度更新间隔
    update_interval = 10  # 每10个记录一次进度

    for j, op in enumerate(ops_list):
        # 计算单个操作符的期望值
        corr = vqs.expect(op)
        r_0j = r_vectors[0, j]
        correlation_data.append({
            'i': 0, 'j': j,
            'r_x': r_0j[0], 'r_y': r_0j[1],
            'corr': corr.mean.real
        })

        # 定期更新进度
        if j % update_interval == 0 or j == len(ops_list) - 1:
            log_message(log_file, f"计算进度: {j+1}/{len(ops_list)}")

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
    log_message(log_file, "使用JAX向量化计算傅里叶变换...")
    log_message(log_file, f"处理 {len(k_grid)} 个k点...")

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

    # 保存相关函数数据
    np.save(os.path.join(save_dir, "spin_correlation_data.npy"), correlation_data)
    np.save(os.path.join(save_dir, "spin_structure_factor.npy"), spin_sf.real)
    np.save(os.path.join(save_dir, "k_points_x.npy"), k_points_x)
    np.save(os.path.join(save_dir, "k_points_y.npy"), k_points_y)

    log_message(log_file, "自旋结构因子计算完成")

    return (k_points_x, k_points_y), spin_sf.real

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
    log_message(log_file, "-"*80)
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
    log_message(log_file, f"晶格尺寸: Lx={Lx}, Ly={Ly}")

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

        # 定期更新进度
        if i % update_interval == 0 or i == len(plaquettes) - 1:
            log_message(log_file, f"预计算简盘操作符进度: {i+1}/{len(plaquettes)}")

    # 计算所有简盘对之间的位移向量
    log_message(log_file, "计算位移向量...")
    plaquette_positions = np.array(plaquette_positions)
    n_plaq = len(plaquettes)

    # 计算所有简盘对之间的位移向量
    r_vectors = np.zeros((n_plaq, n_plaq, 2))
    for i in range(n_plaq):
        r_vectors[i] = plaquette_positions - plaquette_positions[i]

    # 计算简盘-简盘相关函数，使用0点位优化
    log_message(log_file, "计算简盘相关函数...")

    # 选择第一个简盘作为参考点
    reference_plaq = 0
    op_ref = plaquette_ops[reference_plaq]

    log_message(log_file, f"使用简盘 {reference_plaq} 作为参考点...")

    # 预先构建所有j的操作符
    ops_list = []
    for j in range(n_plaq):
        op_j = plaquette_ops[j]
        # 构建组合操作符
        combined_op = op_ref @ op_j
        ops_list.append(combined_op)

    # 直接计算所有操作符的期望值
    log_message(log_file, f"计算 {len(ops_list)} 个相关函数...")
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

        # 定期更新进度
        if j % update_interval == 0 or j == len(ops_list) - 1:
            log_message(log_file, f"计算进度: {j+1}/{len(ops_list)}")

    # 向量化计算傅里叶变换
    log_message(log_file, "计算傅里叶变换...")

    # 提取相关数据用于向量化计算
    r_values = np.array([[data['r_x'], data['r_y']] for data in plaquette_data])
    corr_values = np.array([data['corr'] for data in plaquette_data])

    # 使用已经创建的k网格
    k_grid = np.stack([kx_grid.flatten(), ky_grid.flatten()], axis=1)

    # 使用JAX的向量化功能计算傅里叶变换
    log_message(log_file, "使用JAX向量化计算傅里叶变换...")
    log_message(log_file, f"处理 {len(k_grid)} 个k点...")

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

    log_message(log_file, f"使用总位点数N={N}进行归一化，而不是简盘数量n_plaq={n_plaq}")

    # 保存数据
    np.save(os.path.join(save_dir, "plaquette_correlation_data.npy"), plaquette_data)
    np.save(os.path.join(save_dir, "plaquette_structure_factor.npy"), plaq_sf.real)
    np.save(os.path.join(save_dir, "k_points_x.npy"), k_points_x)
    np.save(os.path.join(save_dir, "k_points_y.npy"), k_points_y)

    log_message(log_file, "简盘结构因子计算完成")

    return (k_points_x, k_points_y), plaq_sf.real

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

def compute_spin_dot_product(hilbert, site_i, site_j, spin_ops=None):
    """
    计算两个位点间的自旋点积算符 S_i·S_j

    参数:
    - hilbert: 希尔伯特空间
    - site_i: 第一个位点索引
    - site_j: 第二个位点索引
    - spin_ops: 预计算的自旋操作符列表，如果提供则使用，否则创建新的操作符
    """
    if spin_ops is not None:
        # 使用预计算的操作符
        sx_i, sy_i, sz_i = spin_ops[site_i]
        sx_j, sy_j, sz_j = spin_ops[site_j]
    else:
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
    C_d(r) = <[S(0)·S(0+x)][S(r)·S(r+x)]> 和 <[S(0)·S(0+y)][S(r)·S(r+y)]>
    分别处理x和y方向的二聚体，使用总位点数N进行归一化
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
    log_message(log_file, "-"*80)
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
    log_message(log_file, f"晶格尺寸: Lx={Lx}, Ly={Ly}")

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

    log_message(log_file, f"找到 {n_dimers_x} 个x方向的二聚体和 {n_dimers_y} 个y方向的二聚体，总共 {len(dimers)} 个二聚体")
    n_dimers = len(dimers)

    # 预计算所有二聚体的自旋点积操作符
    log_message(log_file, "预计算二聚体操作符...")
    dimer_ops = []

    # 预计算所有位点的自旋操作符
    log_message(log_file, "预计算位点自旋操作符...")
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

        # 定期更新进度
        if i % nodes_update_interval == 0 or i == lattice.n_nodes - 1:
            log_message(log_file, f"预计算位点自旋操作符进度: {i+1}/{lattice.n_nodes}")

    # 计算进度更新间隔
    update_interval = 10  # 每10个记录一次进度

    for i, (i1, i2) in enumerate(dimers):
        # 创建二聚体算符 S_i1·S_i2，使用预计算的自旋操作符
        S_i1_dot_S_i2 = compute_spin_dot_product(vqs.hilbert, i1, i2, spin_ops)
        dimer_ops.append(S_i1_dot_S_i2)

        # 定期更新进度
        if i % update_interval == 0 or i == n_dimers - 1:
            log_message(log_file, f"预计算二聚体操作符进度: {i+1}/{n_dimers}")

    # 计算所有二聚体对之间的位移向量
    log_message(log_file, "计算位移向量...")
    dimer_positions = np.array(dimer_positions)

    # 计算所有二聚体对之间的位移向量
    r_vectors = np.zeros((n_dimers, n_dimers, 2))
    for i in range(n_dimers):
        r_vectors[i] = dimer_positions - dimer_positions[i]

    # 计算二聚体-二聚体相关函数，分别处理x方向和y方向
    log_message(log_file, "计算二聚体相关函数...")
    dimer_data = []

    # 分别处理x方向和y方向的二聚体
    if n_dimers_x > 0:
        # 选择第一个x方向二聚体作为x方向参考点
        reference_dimer_x = 0  # dimers_x列表中的第一个
        op_ref_x = dimer_ops[reference_dimer_x]
        dimer_ref_x = dimers[reference_dimer_x]

        log_message(log_file, f"使用x方向二聚体 {reference_dimer_x} 作为x方向参考点: {dimer_ref_x}...")

        # 预先构建所有x方向二聚体的操作符
        ops_list_x = []
        for j in range(n_dimers_x):
            op_j = dimer_ops[j]
            # 构建组合操作符
            combined_op = op_ref_x @ op_j
            ops_list_x.append((j, combined_op))

        # 直接计算所有x方向操作符的期望值
        log_message(log_file, f"计算 {len(ops_list_x)} 个x方向相关函数...")

        # 计算进度更新间隔
        update_interval = 10  # 每10个记录一次进度

        # 直接计算所有操作符的期望值
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

            # 定期更新进度
            if idx % update_interval == 0 or idx == len(ops_list_x) - 1:
                log_message(log_file, f"计算x方向进度: {idx+1}/{len(ops_list_x)}")

    if n_dimers_y > 0:
        # 选择第一个y方向二聚体作为y方向参考点
        reference_dimer_y = n_dimers_x  # dimers列表中的第一个y方向二聚体
        op_ref_y = dimer_ops[reference_dimer_y]
        dimer_ref_y = dimers[reference_dimer_y]

        log_message(log_file, f"使用y方向二聚体 {reference_dimer_y} 作为y方向参考点: {dimer_ref_y}...")

        # 预先构建所有y方向二聚体的操作符
        ops_list_y = []
        for j in range(n_dimers_x, n_dimers):
            op_j = dimer_ops[j]
            # 构建组合操作符
            combined_op = op_ref_y @ op_j
            ops_list_y.append((j, combined_op))

        # 直接计算所有y方向操作符的期望值
        log_message(log_file, f"计算 {len(ops_list_y)} 个y方向相关函数...")

        # 计算进度更新间隔
        update_interval = 10  # 每10个记录一次进度

        # 直接计算所有操作符的期望值
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

            # 定期更新进度
            if idx % update_interval == 0 or idx == len(ops_list_y) - 1:
                log_message(log_file, f"计算y方向进度: {idx+1}/{len(ops_list_y)}")

    log_message(log_file, f"总共计算了 {len(dimer_data)} 个二聚体相关函数")

    # 向量化计算傅里叶变换
    log_message(log_file, "计算傅里叶变换...")

    # 提取相关数据用于向量化计算
    r_values = np.array([[data['r_x'], data['r_y']] for data in dimer_data])
    corr_values = np.array([data['corr'] for data in dimer_data])

    # 使用已经创建的k网格
    k_grid = np.stack([kx_grid.flatten(), ky_grid.flatten()], axis=1)

    # 使用JAX的向量化功能计算傅里叶变换
    log_message(log_file, "使用JAX向量化计算傅里叶变换...")
    log_message(log_file, f"处理 {len(k_grid)} 个k点...")

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

    # 保存数据
    np.save(os.path.join(save_dir, "dimer_correlation_data.npy"), dimer_data)
    np.save(os.path.join(save_dir, "dimer_structure_factor.npy"), dimer_sf.real)
    np.save(os.path.join(save_dir, "k_points_x.npy"), k_points_x)
    np.save(os.path.join(save_dir, "k_points_y.npy"), k_points_y)

    log_message(log_file, "二聚体结构因子计算完成")

    return (k_points_x, k_points_y), dimer_sf.real

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

    # 从目录名称中提取L值
    L = None
    try:
        dir_parts = save_dir.split('/')
        for part in dir_parts:
            if part.startswith('L='):
                L = int(part.split('=')[1])
                break
    except:
        # 如果提取失败，使用默认值
        L = 4

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

    # 找到结构因子的最大值位置
    max_idx = np.unravel_index(np.argmax(structure_factor), structure_factor.shape)
    max_idx_y, max_idx_x = max_idx  # 注意：第一个索引是y，第二个是x
    k_max_x = k_points_x[max_idx_x]
    k_max_y = k_points_y[max_idx_y]
    S_max = structure_factor[max_idx].item()

    # 计算δk = 2π/L，正确的相关长度定义
    dk = np.pi / L
    log_message(log_file, f"使用dk = π/L = {dk:.6f}，L={L}")

    # 找到最接近k_max + δk的网格点
    # x方向
    k_plus_dk_x = k_max_x + dk
    idx_x_plus = np.argmin(np.abs(k_points_x - k_plus_dk_x))
    k_x_plus_actual = k_points_x[idx_x_plus]

    # y方向
    k_plus_dk_y = k_max_y + dk
    idx_y_plus = np.argmin(np.abs(k_points_y - k_plus_dk_y))
    k_y_plus_actual = k_points_y[idx_y_plus]

    # 获取对应的结构因子值
    S_kxplus = structure_factor[max_idx_y, idx_x_plus].item()
    S_kyplus = structure_factor[idx_y_plus, max_idx_x].item()

    # 取平均
    S_kplus = 0.5 * (S_kxplus + S_kyplus)

    log_message(log_file, f"S_max = {S_max:.6f} at k=({k_max_x:.4f}, {k_max_y:.4f})")
    log_message(log_file, f"理论上的k+δk点: x=({k_plus_dk_x:.4f}, {k_max_y:.4f}), y=({k_max_x:.4f}, {k_plus_dk_y:.4f})")
    log_message(log_file, f"实际使用的最近网格点: x=({k_x_plus_actual:.4f}, {k_max_y:.4f}), y=({k_max_x:.4f}, {k_y_plus_actual:.4f})")
    log_message(log_file, f"S(k+δk_x) = {S_kxplus:.6f}, S(k+δk_y) = {S_kyplus:.6f}")
    log_message(log_file, f"S_kplus (avg) = {S_kplus:.6f}")

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


def calculate_af_order_parameter(k_points_tuple, spin_sf, L, save_dir, log_file=None, spin_data=None):
    """
    计算反铁磁序参数 (AF Order Parameter)：m^2(L) = S(π, π)/L^2
    同时计算基于实空间的反铁磁序参量（交错磁化率）

    参数:
    - k_points_tuple: 包含k_points_x和k_points_y的元组
    - spin_sf: 自旋结构因子
    - L: 系统大小
    - save_dir: 保存目录
    - log_file: 日志文件
    - spin_data: 自旋相关函数数据列表（可选，用于计算实空间的反铁磁序参量）
    """
    # 解包k点
    k_points_x, k_points_y = k_points_tuple
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

    # 找到最接近 (π, π) 的k点，对应反铁磁波矢
    # 由于我们使用linspace生成k点，应该有一个点正好在π位置
    pi_idx_x = np.argmin(np.abs(k_points_x - np.pi))
    pi_idx_y = np.argmin(np.abs(k_points_y - np.pi))

    # 获取 S(π, π) 的值
    # 在[0, 2π]范围内，(π, π)对应的是反铁磁波矢
    S_pi_pi = spin_sf[pi_idx_y, pi_idx_x]

    # 记录找到的k点位置
    k_pi_x = k_points_x[pi_idx_x]
    k_pi_y = k_points_y[pi_idx_y]
    log_message(log_file, f"找到最接近(π, π)的k点: ({k_pi_x:.4f}, {k_pi_y:.4f})")
    log_message(log_file, f"该点与理论π值的偏差: ({k_pi_x-np.pi:.8f}, {k_pi_y-np.pi:.8f})")

    # 找到结构因子的最大值位置，用于比较
    max_idx = np.unravel_index(np.argmax(spin_sf), spin_sf.shape)
    k_max_x = k_points_x[max_idx[1]]
    k_max_y = k_points_y[max_idx[0]]
    S_max = spin_sf[max_idx].item()
    log_message(log_file, f"结构因子最大值位置: ({k_max_x:.4f}, {k_max_y:.4f}), 值: {S_max:.6f}")
    log_message(log_file, f"最大值与(π, π)的偏差: ({k_max_x-np.pi:.4f}, {k_max_y-np.pi:.4f})")

    # 计算反铁磁序参数（基于k空间）
    # 注意：结构因子已经在calculate_spin_structure_factor函数中通过spin_sf /= N进行了归一化
    # 所以这里直接使用S_pi_pi，而不需要再次除以N
    m_squared_k = S_pi_pi
    log_message(log_file, "结构因子已经在计算时归一化，直接使用S(π,π)作为反铁磁序参量")

    # 计算基于实空间的反铁磁序参量（如果提供了spin_data）
    m_squared_r = None
    if spin_data is not None and len(spin_data) > 0:
        log_message(log_file, "计算基于实空间的反铁磁序参量（交错磁化率）...")

        # 初始化交错磁化率
        s_m = 0.0

        # 计算交错磁化率：∑_r (-1)^(r_x+r_y) <S_0·S_r>
        for data in spin_data:
            r_x = data['r_x']
            r_y = data['r_y']

            # 计算交错因子 (-1)^(r_x+r_y)
            # 由于r_x和r_y可能是浮点数，我们需要四舍五入到最接近的整数
            staggered_factor = (-1) ** (int(round(r_x)) + int(round(r_y)))

            # 累加贡献
            s_m += staggered_factor * data['corr']

        # 归一化，使用总位点数
        N = L * L * 4
        m_squared_r = (s_m + 0.75) / N  # 加0.75是为了包含自相关项

        log_message(log_file, f"基于实空间的反铁磁序参量: m^2_r(L) = {m_squared_r:.6f}")
        log_message(log_file, f"k空间和实空间计算结果的差异: {abs(m_squared_k - m_squared_r):.6f}")

    # 保存结果
    af_data = {
        'S_pi_pi': S_pi_pi,
        'm_squared_k': m_squared_k,
        'm_squared_r': m_squared_r,
        'L': L
    }

    np.save(os.path.join(save_dir, "af_order_parameter.npy"), af_data)

    # 记录反铁磁序参数信息
    log_message(log_file, f"基于k空间的反铁磁序参数 m^2_k(L) = {m_squared_k:.6f}")
    log_message(log_file, f"S(π, π) = {S_pi_pi:.6f}")

    # 返回k空间计算的结果，保持与之前的接口一致
    return m_squared_k


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
    target_r2 = np.array([physical_size_x/2 - 1.0, physical_size_y/2 - 1.0])

    log_message(log_file, f"寻找接近中心位置 ({target_r1[0]:.2f}, {target_r1[1]:.2f}) 和次中心位置 ({target_r2[0]:.2f}, {target_r2[1]:.2f}) 的相关函数")

    min_dist1 = float('inf')
    min_dist2 = float('inf')

    for data in plaquette_data:
        r = np.array([data['r_x'], data['r_y']])

        # 计算与目标位置的距离
        dist1 = np.linalg.norm(r - target_r1)
        dist2 = np.linalg.norm(r - target_r2)

        # 更新最接近系统中心的点
        if dist1 < min_dist1:
            min_dist1 = dist1
            C_L2_L2 = data['corr']

        # 更新最接近次中心位置的点
        if dist2 < min_dist2:
            min_dist2 = dist2
            C_L2m1_L2m1 = data['corr']

    # 记录找到的最接近点的位置
    log_message(log_file, f"找到最接近中心的点，距离为 {min_dist1:.4f}")
    log_message(log_file, f"找到最接近次中心的点，距离为 {min_dist2:.4f}")

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

    log_message(log_file, f"x方向二聚体序参量: D^2_x = {D_squared_x:.6f} (来自{count_x}个数据点)")
    log_message(log_file, f"y方向二聚体序参量: D^2_y = {D_squared_y:.6f} (来自{count_y}个数据点)")

    # 保存结果
    dimer_order_data = {
        'D_squared': D_squared,
        'D_squared_x': D_squared_x,
        'D_squared_y': D_squared_y,
        'count_x': count_x,
        'count_y': count_y,
        'L': L
    }

    np.save(os.path.join(save_dir, "dimer_order_parameter.npy"), dimer_order_data)

    # 记录二聚体序参量信息
    log_message(log_file, f"总二聚体序参量 D^2 = {D_squared:.6f} (x和y方向的平均值)")

    return D_squared
