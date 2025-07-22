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