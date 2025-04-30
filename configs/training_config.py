class ModelConfig:
    # 模型参数
    num_features = 4      # 特征维度
    num_layers = 4        # 编码器层数


class TrainingConfig:
    # 训练参数
    seed = 0               # 随机种子
    diag_shift = 0.15      # 对角线位移
    learning_rate = 0.018  # 学习率
    N_iters = 1000         # 退火迭代次数
    N_samples = 2**12      # 样本数量
    N_discard = 0          # 丢弃的样本数
    chunk_size = 2**10     # 批处理大小
    temperature = 1.0      # 初始温度

class SystemConfig:
    # 系统参数
    J2_LIST = [0.05]                    # J2耦合强度列表

    # 定义要研究的J1值列表
    J1_LIST = [0.06, 0.09, 0.10]        # J1耦合强度列表（变化参数）

    # 定义要研究的系统大小
    L = 5                               # 晶格大小

    # 自旋类型
    spin = 0.5                          # 自旋大小

    # 参考能量（用于计算相对误差）
    # reference_energy = -16.2631116
    reference_energy = None             # 不使用参考能量

    @staticmethod
    def get_size(L):
        return L * L * 4  # 每个单元格有4个站点
