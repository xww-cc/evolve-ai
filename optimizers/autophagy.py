import torch
from torch.distributions import Normal
from typing import Dict, List
from models.base_module import SubNetModule
from models.modular_net import ModularMathReasoningNet
from utils.parallel_utils import parallel_map
from config.logging_setup import setup_logging
import copy

logger = setup_logging()

def cellular_autophagy_model(model_state: Dict) -> Dict:
    """模型级细胞自噬 - 完整"""
    mutated_modules_states = parallel_map(cellular_autophagy_module, model_state['modules_states'])
    new_model_state = copy.deepcopy(model_state)
    new_model_state['modules_states'] = mutated_modules_states
    return new_model_state

def cellular_autophagy_module(module_state: Dict) -> Dict:
    """模块级细胞自噬 - 完整，参数重构"""
    try:
        temp_module = SubNetModule(
            input_dim=module_state['input_dim'],
            output_dim=module_state['output_dim'],
            widths=module_state['widths'],
            activation_fn_name=module_state['activation_fn_name'],
            use_batchnorm=module_state['use_batchnorm'],
            module_type=module_state['module_type']
        )
        temp_module.load_state_dict(module_state['state_dict'])

        params = list(temp_module.parameters())
        total_params_count = sum(p.numel() for p in params)
        
        adaptive_threshold = 0.05 * (total_params_count**0.25) if total_params_count > 0 else 0.01
        
        logger.debug(f"细胞自噬 - 参数总数: {total_params_count}, 自适应阈值: {adaptive_threshold:.6f}")

        autophagy_count = 0
        for param in params:
            if param.ndim < 2: 
                continue
            
            norm = torch.norm(param.data) / (param.numel()**0.5 + 1e-6) 
            
            if norm < adaptive_threshold:
                q = max(1, min(3, min(param.shape[0], param.shape[1]))) 
                device = param.data.device
                
                try:
                    u, s, v = torch.svd_lowrank(param.data.float().to(device), q=q)
                    reconstructed = (u @ torch.diag(s) @ v.t()).to(device).type_as(param.data)
                    noise = Normal(0, 0.02).sample(reconstructed.shape).to(device)
                    param.data = reconstructed + noise
                    autophagy_count += 1
                except RuntimeError as e:
                    logger.warning(f"SVD重构失败: {e}")

        if autophagy_count > 0:
            logger.debug(f"细胞自噬完成 - 处理了 {autophagy_count} 个参数")

        return temp_module.get_state()
    except Exception as e:
        logger.error(f"细胞自噬失败: {e}")
        return module_state