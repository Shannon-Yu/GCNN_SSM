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

    # 使用RdBu_r配色方案（红蓝反转）
    # 这是一个从蓝色（低值）到白色（中值）到红色（高值）的渐变
    # 非常适合科学可视化，特别是对于结构因子这类数据
    cmap = plt.cm.RdBu_r

    # 添加3D图 (左侧)
    ax1 = fig.add_subplot(121, projection='3d')

    # 交换kx和ky，使ky从2π到0（从前到后），kx从0到2π（从左到右）
    # 注意：这里我们交换了kx_grid和ky_grid的位置，并反转ky轴
    # 确保3D图中的数据点与热力图中的数据点完全一致
    ax1.plot_surface(ky_grid, kx_grid, sf_data, cmap=cmap,
                    linewidth=0, antialiased=True)

    # 设置3D图的标签和标题，删除z轴标签
    ax1.set_xlabel('$k_y$', fontsize=14)
    ax1.set_ylabel('$k_x$', fontsize=14)
    # 删除z轴标签
    ax1.set_title(f'3D {factor_type} Structure Factor\nL={L}, J2={J2:.2f}, J1={J1:.2f}')

    # 设置x轴和y轴的刻度为0, π, 2π
    ax1.set_xticks([0, np.pi, 2*np.pi])
    ax1.set_yticks([0, np.pi, 2*np.pi])
    ax1.set_xticklabels(['0', '$\pi$', '$2\pi$'])
    ax1.set_yticklabels(['0', '$\pi$', '$2\pi$'])

    # 设置轴的范围
    ax1.set_xlim([0, 2*np.pi])
    ax1.set_ylim([0, 2*np.pi])

    # 反转ky轴（x轴）方向，使其从2π到0
    ax1.invert_xaxis()

    # 根据数据点添加3D网格线
    # 添加与热力图相同的网格线
    for x in k_points_y:  # 注意：这里使用k_points_y是因为我们交换了坐标轴
        ax1.plot([x, x], [0, 2*np.pi], [0, 0], color='gray', linestyle='-', alpha=0.2)

    for y in k_points_x:  # 注意：这里使用k_points_x是因为我们交换了坐标轴
        ax1.plot([0, 2*np.pi], [y, y], [0, 0], color='gray', linestyle='-', alpha=0.2)

    # 调整视角，使ky从2π到0（从前到后），kx从0到2π（从左到右）
    ax1.view_init(elev=30, azim=-45)  # 调整仰角和方位角

    # 添加热图 (右侧)
    ax2 = fig.add_subplot(122)
    extent = [0, 2*np.pi, 0, 2*np.pi]  # 固定范围为[0, 2π]×[0, 2π]

    # 创建热图，使用更漂亮的配色方案
    im = ax2.imshow(sf_data, extent=extent, origin='lower', cmap=cmap,
                   interpolation='bilinear', aspect='equal')

    # 添加颜色条 - 两张图共用一个
    fig.colorbar(im, ax=[ax1, ax2], label=f'{factor_type} S(k)')

    # 根据数据点添加网格线
    # 计算网格线的位置，与数据点对应
    grid_x = k_points_x
    grid_y = k_points_y

    # 添加垂直网格线
    for x in grid_x:
        ax2.axvline(x=x, color='gray', linestyle='-', alpha=0.2)

    # 添加水平网格线
    for y in grid_y:
        ax2.axhline(y=y, color='gray', linestyle='-', alpha=0.2)

    # 设置热图的标签和标题
    ax2.set_xlabel('$k_x$', fontsize=14)
    ax2.set_ylabel('$k_y$', fontsize=14)
    ax2.set_title(f'{factor_type} Structure Factor Heatmap\nL={L}, J2={J2:.2f}, J1={J1:.2f}')

    # 设置x轴和y轴的刻度为0, π, 2π
    ax2.set_xticks([0, np.pi, 2*np.pi])
    ax2.set_yticks([0, np.pi, 2*np.pi])
    ax2.set_xticklabels(['0', '$\pi$', '$2\pi$'])
    ax2.set_yticklabels(['0', '$\pi$', '$2\pi$'])

    # 设置轴的范围
    ax2.set_xlim([0, 2*np.pi])
    ax2.set_ylim([0, 2*np.pi])

    # 不需要默认网格线，因为我们已经添加了自定义网格线

    # 保存图片
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{factor_type}_sf_L={L}_J2={J2:.2f}_J1={J1:.2f}.png", dpi=300)
    plt.close()
