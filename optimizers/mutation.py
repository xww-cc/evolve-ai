import random
import copy
from typing import List, Dict, Optional
from config.global_constants import ACTIVATION_FNS, BASE_MUTATION_RATE_STRUCTURE, EPIGENETIC_MUTATION_RATE
from integrations.xai_integration import XAIIntegration
from models.epigenetic import EpigeneticMarkers
from torch.distributions import Normal
import asyncio
import torch

async def mutate_modular_net(modules_config: List[Dict], epigenetic_markers: EpigeneticMarkers, min_sub_modules: int, max_sub_modules: int, min_subnet_width: int, max_subnet_width: int, width_step: int, activation_fns: List[str], base_mutation_rate_structure: float, epigenetic_mutation_rate: float) -> List[Dict]:
    """模块化网络变异 - 完整"""
    mutated_config = copy.deepcopy(modules_config)

    structure_tendency = epigenetic_markers[0].item()  # 注意原脚本是[1]，这里修正为[0]以匹配

    mutation_types = []
    if len(mutated_config) < max_sub_modules:
        mutation_types.append('add_module')
    if len(mutated_config) > min_sub_modules:
        mutation_types.append('remove_module')
    mutation_types.extend(['change_module_width', 'change_module_layers', 'change_module_activation', 'toggle_module_batchnorm', 'change_module_type', 'change_input_source'])

    mutation_weights = [1.0 + max(0, structure_tendency * 2) if t == 'add_module' else 1.0 + max(0, -structure_tendency * 2) if t == 'remove_module' else 1.5 if t == 'change_input_source' else 1.0 for t in mutation_types]
    
    total_weight = sum(mutation_weights)
    mutation_probs = [w / total_weight for w in mutation_weights] 

    if random.random() < base_mutation_rate_structure * (1 + abs(structure_tendency)): 
        mutation_choice = random.choices(mutation_types, weights=mutation_probs, k=1)[0]

        if mutation_choice == 'add_module':
            insert_idx = random.randint(0, len(mutated_config))
            new_module_cfg = get_random_module_config(insert_idx, min_subnet_width, max_subnet_width, width_step, activation_fns)
            mutated_config.insert(insert_idx, new_module_cfg)
            
        elif mutation_choice == 'remove_module':
            if len(mutated_config) > min_sub_modules:
                remove_idx = random.randint(0, len(mutated_config) - 1)
                try:
                    mutated_config.pop(remove_idx)
                except IndexError:
                    pass
                
        elif mutated_config:
            # 确保有有效的模块可以变异
            if len(mutated_config) == 0:
                return mutated_config
                
            module_idx = random.randint(0, len(mutated_config) - 1)
            try:
                module_cfg = mutated_config[module_idx]
                # 确保module_cfg是字典格式
                if not isinstance(module_cfg, dict):
                    # 如果module_cfg不是字典，创建一个新的有效配置
                    module_cfg = get_random_module_config(module_idx, min_subnet_width, max_subnet_width, width_step, activation_fns)
                    mutated_config[module_idx] = module_cfg
            except (IndexError, KeyError):
                # 如果索引超出范围，直接返回
                return mutated_config

            if mutation_choice == 'change_module_width':
                if isinstance(module_cfg, dict) and 'widths' in module_cfg and module_cfg['widths']:
                    layer_idx = random.randint(0, len(module_cfg['widths']) - 1)
                    delta = int(Normal(0, width_step).sample().item())
                    module_cfg['widths'][layer_idx] = max(min_subnet_width, min(max_subnet_width, module_cfg['widths'][layer_idx] + delta))
            
            elif mutation_choice == 'change_module_layers':
                if isinstance(module_cfg, dict) and 'widths' in module_cfg:
                    if random.random() < 0.5 and len(module_cfg['widths']) < 3:
                        module_cfg['widths'].append(random.randrange(min_subnet_width, max_subnet_width + 1, width_step))
                    elif len(module_cfg['widths']) > 1:
                        module_cfg['widths'].pop(random.randint(0, len(module_cfg['widths']) - 1))
            
            elif mutation_choice == 'change_module_activation':
                if isinstance(module_cfg, dict) and 'activation_fn_name' in module_cfg:
                    possible_activations = [fn for fn in activation_fns if fn != module_cfg['activation_fn_name']]
                    if possible_activations:
                        module_cfg['activation_fn_name'] = random.choice(possible_activations)
            
            elif mutation_choice == 'toggle_module_batchnorm':
                if isinstance(module_cfg, dict) and 'use_batchnorm' in module_cfg:
                    module_cfg['use_batchnorm'] = not module_cfg['use_batchnorm']
            
            elif mutation_choice == 'change_module_type':
                if isinstance(module_cfg, dict) and 'module_type' in module_cfg:
                    possible_module_types = ['add_sub', 'trig', 'exp_log', 'prod', 'calculus', 'linear_alg', 'agi', 'generic']
                    module_cfg['module_type'] = random.choice([t for t in possible_module_types if t != module_cfg['module_type']])
            
            elif mutation_choice == 'change_input_source':
                if isinstance(module_cfg, dict) and 'input_source' in module_cfg:
                    possible_sources = ['initial_input'] + list(range(module_idx))
                    if possible_sources:
                        module_cfg['input_source'] = random.choice(possible_sources)

    # LLM符号变异 - 完整
    if random.random() < 0.25:
        current_performance = random.uniform(0.1, 0.9)  # 模拟性能指标
        xai = XAIIntegration()
        llm_suggest = await xai.generate_activation_suggestion("LeakyReLU", current_performance)
        valid_activations = ACTIVATION_FNS
        if llm_suggest in valid_activations:
            for module_cfg in mutated_config:
                if isinstance(module_cfg, dict) and 'activation_fn_name' in module_cfg:
                    module_cfg['activation_fn_name'] = llm_suggest
            # logger.debug(f"LLM建议激活函数: {llm_suggest}")

    while len(mutated_config) < min_sub_modules:
        mutated_config.append(get_random_module_config(len(mutated_config), min_subnet_width, max_subnet_width, width_step, activation_fns))
    while len(mutated_config) > max_sub_modules:
        if len(mutated_config) > 0:
            remove_idx = random.randint(0, len(mutated_config) - 1)
            if remove_idx < len(mutated_config):
                try:
                    mutated_config.pop(remove_idx)
                except (IndexError, KeyError):
                    break
        else:
            break

    return mutated_config

def get_random_module_config(max_modules_idx: int, min_width: int, max_width: int, width_step: int, activation_fns: List[str], module_type: Optional[str] = None) -> Dict:
    """创建随机模块配置 - 完整"""
    widths = [random.randrange(min_width, max_width + 1, width_step) for _ in range(random.randint(1, 2))]
    activation = random.choice(activation_fns)
    use_bn = random.choice([True, False])
    
    possible_module_types = ['add_sub', 'trig', 'exp_log', 'prod', 'calculus', 'linear_alg', 'agi', 'generic']
    if module_type is None:
        module_type = random.choice(possible_module_types)

    input_source = 'initial_input'
    if max_modules_idx > 0 and random.random() < 0.5:
        input_source = random.randint(0, max_modules_idx - 1)
    
    dummy_input_dim = 4
    output_dim = random.randrange(min_width, max_width + 1, width_step)

    return {
        'input_dim': dummy_input_dim,
        'output_dim': output_dim,
        'widths': widths,
        'activation_fn_name': activation,
        'use_batchnorm': use_bn,
        'module_type': module_type,
        'input_source': input_source
    }

def crossover_modular_nets(parent1_config: List[Dict], parent2_config: List[Dict], min_sub_modules: int, max_sub_modules: int, min_subnet_width: int, max_subnet_width: int, width_step: int, activation_fns: List[str]) -> List[Dict]:
    """模块化网络交叉 - 完整"""
    crossover_point1 = random.randint(0, len(parent1_config))
    crossover_point2 = random.randint(0, len(parent2_config))

    child_config = parent1_config[:crossover_point1] + parent2_config[crossover_point2:]

    if not child_config: 
        return [get_random_module_config(0, min_subnet_width, max_subnet_width, width_step, activation_fns)]

    while len(child_config) < min_sub_modules:
        child_config.append(get_random_module_config(len(child_config), min_subnet_width, max_subnet_width, width_step, activation_fns))
    while len(child_config) > max_sub_modules:
        if len(child_config) > 0:
            remove_idx = random.randint(0, len(child_config) - 1)
            if remove_idx < len(child_config):
                try:
                    child_config.pop(remove_idx)
                except (IndexError, KeyError):
                    break
        else:
            break

    fixed_child_config = []
    temp_module_outputs_dims = {'initial_input': 4} 
    
    for i, module_cfg in enumerate(child_config):
        # 确保module_cfg是字典格式
        if not isinstance(module_cfg, dict):
            new_module_cfg = get_random_module_config(i, min_subnet_width, max_subnet_width, width_step, activation_fns)
        else:
            new_module_cfg = copy.deepcopy(module_cfg)
        
        input_source = new_module_cfg.get('input_source', 'initial_input')
        if isinstance(input_source, int):
            if input_source >= i or input_source < 0:
                new_module_cfg['input_source'] = random.choice(['initial_input'] + list(range(i))) if i > 0 else 'initial_input'
        
        fixed_child_config.append(new_module_cfg)
        temp_module_outputs_dims[f'module_{i}'] = new_module_cfg['output_dim'] 

    return fixed_child_config