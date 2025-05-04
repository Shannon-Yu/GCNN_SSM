import matplotlib.pyplot as plt
import numpy as np

def plot_structure_factor(k_points_tuple, sf_data, L, J2, J1, factor_type, save_dir):
    """
    绘制结构因子图，包括3D图和热图

    参数:
    - k_points_tuple: 包含k_points_x和k_points_y的元组
    - sf_data: 结构因子数据
    - L: 系统大小
    - J2: J2耦合强度
    - J1: J1耦合强度
    - factor_type: 结构因子类型名称
    - save_dir: 保存目录
    """
    # 解包k点
    k_points_x, k_points_y = k_points_tuple

    # 创建网格
    kx_grid, ky_grid = np.meshgrid(k_points_x, k_points_y)

    # 创建一个包含两个子图的图形
    fig = plt.figure(figsize=(16, 6))

    # 添加3D图 (左侧)
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(kx_grid, ky_grid, sf_data, cmap='YlGnBu',
                           linewidth=0, antialiased=True)

    # 设置3D图的标签和标题
    ax1.set_xlabel('$k_x$', fontsize=14)
    ax1.set_ylabel('$k_y$', fontsize=14)
    ax1.set_zlabel(f'{factor_type} S(k)', fontsize=14)
    ax1.set_title(f'3D {factor_type} Structure Factor\nL={L}, J2={J2:.2f}, J1={J1:.2f}')

    # 添加颜色条
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, label=f'{factor_type} S(k)')

    # 添加热图 (右侧)
    ax2 = fig.add_subplot(122)
    extent = [k_points_x.min(), k_points_x.max(), k_points_y.min(), k_points_y.max()]

    # 创建热图，使用黄绿蓝配色方案
    im = ax2.imshow(sf_data, extent=extent, origin='lower', cmap='YlGnBu',
                   interpolation='bilinear', aspect='auto')

    # 添加颜色条
    fig.colorbar(im, ax=ax2, label=f'{factor_type} S(k)')

    # 标记高对称点
    high_sym_points = {
        'Γ': (0, 0),
        'X': (np.pi, 0),
        'Y': (0, np.pi),
        'M': (np.pi, np.pi)
    }

    for label, (kx, ky) in high_sym_points.items():
        ax2.plot(kx, ky, 'wo', markersize=5)
        ax2.text(kx+0.1, ky+0.1, label, color='white', fontsize=12)

    # 设置热图的标签和标题
    ax2.set_xlabel('$k_x$', fontsize=14)
    ax2.set_ylabel('$k_y$', fontsize=14)
    ax2.set_title(f'{factor_type} Structure Factor Heatmap\nL={L}, J2={J2:.2f}, J1={J1:.2f}')

    # 保存图片
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{factor_type}_sf_L={L}_J2={J2:.2f}_J1={J1:.2f}.png", dpi=300)
    plt.close()
