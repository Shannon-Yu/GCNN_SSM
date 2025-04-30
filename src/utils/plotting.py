import matplotlib.pyplot as plt
import numpy as np

def plot_structure_factor(k_points, sf_data, L, J2, J1, factor_type, save_dir):
    """绘制结构因子热图"""
    plt.figure(figsize=(8, 6))
    extent = [-np.pi, np.pi, -np.pi, np.pi]

    # 创建高对比度的热图
    plt.imshow(sf_data, extent=extent, origin='lower', cmap='hot', interpolation='bilinear', aspect='auto')

    # 添加颜色条
    plt.colorbar(label=f'{factor_type} S(k)')

    # 标记高对称点
    high_sym_points = {
        'Γ': (0, 0),
        'X': (np.pi, 0),
        'Y': (0, np.pi),
        'M': (np.pi, np.pi)
    }

    for label, (kx, ky) in high_sym_points.items():
        plt.plot(kx, ky, 'wo', markersize=5)
        plt.text(kx+0.1, ky+0.1, label, color='white', fontsize=12)

    # 设置标签和标题
    plt.xlabel('$k_x$', fontsize=14)
    plt.ylabel('$k_y$', fontsize=14)
    plt.title(f'{factor_type} Structure Factor\nL={L}, J2={J2:.2f}, J1={J1:.2f}')

    # 保存图片
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{factor_type}_sf_L={L}_J2={J2:.2f}_J1={J1:.2f}.png", dpi=300)
    plt.close()
