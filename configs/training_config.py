class ModelConfig:
    # 模型参数
    num_features = 4       # 编码器层数
    num_layers = 4           # 特征维度


class TrainingConfig:
    # 训练参数
    seed = 0
    diag_shift = 0.15
    learning_rate = 0.018            # 学习率
    N_iters = 1000           # 退火迭代次数
    N_samples = 2**12      # 样本数量
    N_discard = 0         # 丢弃的样本数
    chunk_size = 2**10     # 批处理大小
    temperature = 1.0     # 初始温度
    

class SystemConfig:
    # 系统参数
    J2_LIST = [0.05] 
    
    # 定义要研究的J1值列表
    J1_LIST = [0.06, 0.09, 0.10]       # J1是变化参数
    
    # 定义要研究的系统大小
    L = 5
    
    # 自旋类型
    spin = 0.5
    # reference_energy = -16.2631116
    reference_energy = None

    @staticmethod
    def get_size(L):
        return L * L * 4  # 每个单元格有4个站点
