class ModelConfig:
    # 模型参数
    f = 6
    heads = 12
    d_model = f * heads
    patch_size = 2
    n_layers = 2

class TrainingConfig:
    # 训练参数
    seed = 0
    diag_shift = 1e-3
    eta = 0.005
    N_opt = 10000
    N_samples = 2**12
    N_discard = 0
    chunk_size=2**10

class SystemConfig:
    # 系统参数
    J1 = 1.0
    # 定义要研究的J2值列表
    J2_LIST = [0.55]
    # 定义要研究的系统大小
    L_LIST = [6]
    
    @staticmethod
    def get_size(L):
        return L * L