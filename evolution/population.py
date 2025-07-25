from typing import List
from models.modular_net import ModularMathReasoningNet
from models.epigenetic import EpigeneticMarkers
from config.global_constants import *
from config.logging_setup import setup_logging
import random

logger = setup_logging()

def create_initial_population(pop_size: int = POPULATION_SIZE) -> List[ModularMathReasoningNet]:
    """创建初始种群 - 优化维度匹配"""
    pop = []
    possible_module_types = ['add_sub', 'trig', 'exp_log', 'prod', 'calculus', 'linear_alg', 'agi', 'generic']
    
    for _ in range(pop_size):
        num_sub_modules = random.randint(MIN_SUB_MODULES, MAX_SUB_MODULES)
        modules_config = []
        
        # 优化维度管理
        current_dim = 4  # 初始输入维度
        
        for i in range(num_sub_modules):
            # 确保宽度合理
            widths = [random.randrange(MIN_SUBNET_WIDTH, MAX_SUBNET_WIDTH + 1, WIDTH_STEP) for _ in range(random.randint(1, 2))]
            widths = [max(1, w) for w in widths]  # 确保至少为1
            
            activation = random.choice(ACTIVATION_FNS)
            use_bn = random.choice([True, False])
            module_type = random.choice(possible_module_types)
            
            # 输出维度确保合理
            output_dim = random.randrange(MIN_SUBNET_WIDTH, MAX_SUBNET_WIDTH + 1, WIDTH_STEP)
            output_dim = max(1, output_dim)  # 确保至少为1
            
            # 确定输入源
            if i == 0:
                input_source = 'initial_input'
            else:
                input_source = f'module_{i-1}'
            
            modules_config.append({
                'input_source': input_source,
                'output_dim': output_dim,
                'widths': widths,
                'activation_fn_name': activation,
                'use_batchnorm': use_bn,
                'module_type': module_type
            })

        epigenetic_markers = EpigeneticMarkers()
        individual = ModularMathReasoningNet(modules_config=modules_config, epigenetic_markers=epigenetic_markers)
        pop.append(individual)
    
    logger.info(f"初始种群创建完成，共 {len(pop)} 个个体")
    return pop