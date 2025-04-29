import netket as nk
import jax.numpy as jnp
from netket.utils.group import PointGroup, Identity
from netket.utils.group.planar import rotation, reflection_group, D, glide, glide_group
from netket.graph import Graph
from netket.utils.group import PermutationGroup, PointGroup, planar, Identity

def shastry_sutherland_lattice(Lx,Ly):
     # 定义Shastry-Sutherland晶格
    Lx = Lx
    Ly = Ly

    # 定义自定义边 (J1边和J2对角线边)
    custom_edges = [
        (0, 1, [1.0, 0.0], 0),
        (1, 0, [1.0, 0.0], 0),
        (1, 2, [0.0, 1.0], 0),
        (2, 1, [0.0, 1.0], 0),
        (3, 2, [1.0, 0.0], 0),
        (2, 3, [1.0, 0.0], 0),
        (0, 3, [0.0, 1.0], 0),
        (3, 0, [0.0, 1.0], 0),
        (2, 0, [1.0, -1.0], 1),
        (3, 1, [1.0, 1.0], 1),
    ]

    # 创建Shastry-Sutherland晶格
    lattice = nk.graph.Lattice(
        basis_vectors=[[2.0, 0.0], [0.0, 2.0]],
        extent=(Lx, Ly),
        site_offsets=[[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]],
        custom_edges=custom_edges,
        pbc=[True, True]
    )

    return lattice


def shastry_sutherland_hamiltonian(lattice, J1, J2, Q, spin=0.5, total_sz=0):
    """
    创建Shastry-Sutherland模型的哈密顿量，完全支持周期性边界条件
    严格按照论文定义实现Q项，仅包括平行相邻的水平或垂直链接
    """
    hilbert = nk.hilbert.Spin(s=spin, N=lattice.n_nodes, total_sz=total_sz)

    # 自旋-1/2矩阵
    sigmax = jnp.array([[0, 0.5], [0.5, 0]])
    sigmay = jnp.array([[0, -0.5j], [0.5j, 0]])
    sigmaz = jnp.array([[0.5, 0], [0, -0.5]])
    unitm = jnp.array([[1.0, 0.0], [0.0, 1.0]])

    # 自旋-自旋相互作用
    sxsx = jnp.kron(sigmax, sigmax)
    sysy = jnp.kron(sigmay, sigmay)
    szsz = jnp.kron(sigmaz, sigmaz)
    umum = jnp.kron(unitm, unitm)
    SiSj = sxsx + sysy + szsz

    # 定义(Si·Sj - 1/4)算符
    ProjOp = jnp.array(SiSj) - 0.25 * jnp.array(umum)
    ProjOp2 = jnp.kron(ProjOp, ProjOp)

    # 构建J1-J2部分的哈密顿量
    bond_operator = [
        (J1 * jnp.array(SiSj)).tolist(),
        (J2 * jnp.array(SiSj)).tolist(),
    ]
    bond_color = [0, 1]
    H_J = nk.operator.GraphOperator(hilbert, graph=lattice, bond_ops=bond_operator, bond_ops_colors=bond_color)
    
    # 创建Q项哈密顿量
    H_Q = nk.operator.LocalOperator(hilbert, dtype=jnp.complex128)

    # 获取晶格尺寸
    Lx, Ly = lattice.extent[0], lattice.extent[1]
    
    # 遍历所有单元格
    for x in range(Lx):
        for y in range(Ly):
            # 计算当前单元格的基本索引
            base = 4 * (y + x * Ly)
            
            # 当前单元格内的四个格点
            site0 = base      # 左下角 (0.5, 0.5)
            site1 = base + 1  # 右下角 (1.5, 0.5)
            site2 = base + 2  # 右上角 (1.5, 1.5)
            site3 = base + 3  # 左上角 (0.5, 1.5)
            
            # 找到相邻单元格（考虑周期性边界条件）
            right_x = (x + 1) % Lx
            right_base = 4 * (y + right_x * Ly)
            
            left_x = (x - 1 + Lx) % Lx
            left_base = 4 * (y + left_x * Ly)
            
            up_y = (y + 1) % Ly
            up_base = 4 * (up_y + x * Ly)
            
            down_y = (y - 1 + Ly) % Ly
            down_base = 4 * (down_y + x * Ly)
            
            # 1. 单元格内部的水平方向plaquette
            H_Q += nk.operator.LocalOperator(hilbert, [(-Q * ProjOp2).tolist()],
                                         [[site0, site1, site3, site2]])
            
            # 2. 单元格内部的垂直方向plaquette
            H_Q += nk.operator.LocalOperator(hilbert, [(-Q * ProjOp2).tolist()],
                                         [[site0, site3, site1, site2]])
            
            # 3. 与右侧单元格形成的水平plaquette（处理x方向周期性）
            H_Q += nk.operator.LocalOperator(hilbert, [(-Q * ProjOp2).tolist()],
                                         [[site1, right_base, site2, right_base + 3]])
            
            # 4. 与上方单元格形成的垂直plaquette（处理y方向周期性）
            H_Q += nk.operator.LocalOperator(hilbert, [(-Q * ProjOp2).tolist()],
                                         [[site3, up_base, site2, up_base + 1]])
            
            # # 5. 与左侧单元格形成的水平plaquette（处理x方向周期性）
            # H_Q += nk.operator.LocalOperator(hilbert, [(-Q * ProjOp2).tolist()],
            #                              [[site0, left_base + 1, site3, left_base + 2]])
            
            # # 6. 与下方单元格形成的垂直plaquette（处理y方向周期性）
            # H_Q += nk.operator.LocalOperator(hilbert, [(-Q * ProjOp2).tolist()],
            #                              [[site0, down_base + 3, site1, down_base + 2]])
            
    # 合并两部分哈密顿量
    hamiltonian = H_J + 2*H_Q
    hamiltonian = hamiltonian.to_jax_operator()
    return hamiltonian, hilbert




def shastry_sutherland_all_symmetries(lattice):
    """
    为Shastry-Sutherland模型构建对称性
    """
    # 定义晶格对称性，点群对称性C4v + 平移对称性
    nc = 4
    cyclic_4 = PointGroup([Identity()] + [rotation((360 / nc) * i) for i in range(1, nc)],ndim=2,)
    #平移对称性
    C4v = glide_group(trans=(1, 1), origin=(0, 0)) @ cyclic_4
    symmetries = lattice.space_group(C4v)
    
    return symmetries


def auto_symmetries(lattice):
    symmetries = lattice.automorphisms()
    print(f"对称性组大小: {len(symmetries)}")
    return symmetries
    
def shastry_sutherland_point_symmetries(lattice):
    """
    为 Shastry-Sutherland 模型构建对称性，只保留点群对称性和反射对称性。
    这里直接利用晶格提供的点群对称性（通常包含旋转和反射），
    从而简化对称操作的构造。
    """
    # 这里 lattice.point_group() 返回的是晶格的点群对称性（包括反射）
    symmetries = lattice.point_group(D(4))
    
    print(f"对称性组大小: {len(symmetries)}")
    return symmetries

