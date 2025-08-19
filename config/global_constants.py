from typing import List, Dict

ACTIVATION_FNS: List[str] = ['LeakyReLU', 'ReLU', 'Tanh', 'Sigmoid', 'Identity', 'Square', 'Sine', 'Exp', 'Multiplication', 'DDSR_Add', 'DDSR_Div']
MIN_SUB_MODULES: int = 2
MAX_SUB_MODULES: int = 4
MIN_SUBNET_WIDTH: int = 8
MAX_SUBNET_WIDTH: int = 32
WIDTH_STEP: int = 8
POPULATION_SIZE: int = 20
NUM_GENERATIONS: int = 50
BASE_MUTATION_RATE_STRUCTURE: float = 0.15
EPIGENETIC_MUTATION_RATE: float = 0.1
EPIGENETIC_MUTATION_STRENGTH: float = 0.1
STAGNATION_WINDOW: int = 10
STAGNATION_THRESHOLD: float = 0.0005
ENVIRONMENT_PRESSURE_STRENGTH: float = 0.1
BATCH_SIZE: int = 64
LEVEL_DESCRIPTIONS: Dict[int, str] = {
    0: '基础运算',
    1: '乘法运算', 
    2: '指数对数',
    3: '三角函数',
    4: '微积分',
    5: '线性代数',
    6: 'AGI数学融合'
}

# 模块配置
MODULES_CONFIG = {
    'num_modules': 3,
    'module_widths': [16, 16, 16],
    'activation_functions': ['ReLU', 'Tanh', 'Sigmoid'],
    'epigenetic_markers': True
}

# 数值稳定性配置
MAX_OUTPUT_VALUE: float = 1e6  # 最大输出值
MIN_OUTPUT_VALUE: float = -1e6  # 最小输出值
MAX_WEIGHT_VALUE: float = 10.0  # 最大权重值
MIN_WEIGHT_VALUE: float = -10.0  # 最小权重值
MAX_STD_VALUE: float = 10.0  # 最大标准差
MIN_STD_VALUE: float = 0.01  # 最小标准差
MAX_DIVERSITY_VALUE: float = 10.0  # 最大多样性值
MIN_DIVERSITY_VALUE: float = 0.1  # 最小多样性值

# 评估惩罚配置
NAN_PENALTY: float = 0.0  # NaN值惩罚分数
INF_PENALTY: float = 0.0  # 无穷值惩罚分数
EXTREME_PENALTY: float = 0.0  # 极端值惩罚分数