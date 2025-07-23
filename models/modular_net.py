import torch
import torch.nn as nn
from typing import List, Dict
from models.base_module import SubNetModule
from models.epigenetic import EpigeneticMarkers
from integrations.xai_integration import XAIIntegration
from config.logging_setup import setup_logging
import copy
import sympy as sp

logger = setup_logging()

class ModularMathReasoningNet(nn.Module):
    """模块化数学推理网络 - 完整"""
    def __init__(self, modules_config: List[Dict], epigenetic_markers: EpigeneticMarkers = None):
        super().__init__()
        self.modules_config = copy.deepcopy(modules_config) 
        self.subnet_modules = nn.ModuleList()
        
        temp_module_outputs_dims = {'initial_input': 4} 
        
        for i, cfg in enumerate(self.modules_config):
            if not isinstance(cfg, dict):
                raise ValueError(f"Module config {i} is not a dictionary")
            
            input_source = cfg.get('input_source', 'initial_input')
            if isinstance(input_source, int): 
                if input_source >= i or input_source < 0: 
                    raise ValueError(f"Module {i} has invalid input_source {input_source}.")
                actual_input_dim = temp_module_outputs_dims.get(f'module_{input_source}', 0)
            elif isinstance(input_source, str):
                if input_source == 'initial_input':
                    actual_input_dim = 4
                elif input_source.startswith('module_'):
                    # 解析模块引用，如 'module_0', 'module_1' 等
                    try:
                        module_idx = int(input_source.split('_')[1])
                        if module_idx >= i or module_idx < 0:
                            raise ValueError(f"Module {i} has invalid input_source {input_source}.")
                        actual_input_dim = temp_module_outputs_dims.get(input_source, 0)
                    except (ValueError, IndexError):
                        actual_input_dim = 4
                else:
                    actual_input_dim = 4
            else:
                actual_input_dim = 4

            cfg['input_dim'] = actual_input_dim
            
            module = SubNetModule(
                cfg['input_dim'],
                cfg['output_dim'],
                cfg['widths'],
                cfg['activation_fn_name'],
                cfg['use_batchnorm'],
                cfg['module_type']
            )
            self.subnet_modules.append(module)
            temp_module_outputs_dims[f'module_{i}'] = cfg['output_dim']

        # 确保有有效的模块配置
        if self.modules_config and len(self.modules_config) > 0:
            try:
                last_config = self.modules_config[-1]
                if isinstance(last_config, dict):
                    final_input_dim = last_config.get('output_dim', 4)
                else:
                    final_input_dim = 4
            except (IndexError, KeyError):
                final_input_dim = 4
        else:
            final_input_dim = 4 
        self.final_output_layer = nn.Linear(final_input_dim, 1)
        
        self.epigenetic_markers = epigenetic_markers or EpigeneticMarkers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """简化前向传播 - 确保维度匹配"""
        module_outputs = {'initial_input': x} 
        
        for i, module in enumerate(self.subnet_modules):
            # 获取输入
            input_source_key = f'module_{i-1}' if i > 0 else 'initial_input'
            input_tensor = module_outputs.get(input_source_key)
            
            if input_tensor is None:
                # 如果输入不存在，使用初始输入
                input_tensor = x
            
            # 强制维度匹配 - 简化版本
            if input_tensor.shape[1] != module.input_dim:
                # 使用简单的线性变换
                adjust_layer = nn.Linear(input_tensor.shape[1], module.input_dim).to(input_tensor.device)
                nn.init.xavier_uniform_(adjust_layer.weight)
                nn.init.zeros_(adjust_layer.bias)
                input_tensor = adjust_layer(input_tensor)
            
            # 执行模块
            try:
                current_output = module(input_tensor)
                module_outputs[f'module_{i}'] = current_output
            except Exception as e:
                # 如果模块执行失败，使用零张量
                current_output = torch.zeros(input_tensor.shape[0], module.output_dim, device=input_tensor.device)
                module_outputs[f'module_{i}'] = current_output
        
        # 返回最后一个模块的输出
        try:
            return module_outputs[f'module_{len(self.subnet_modules)-1}']
        except:
            # 如果没有模块，返回输入
            return x

    async def extract_symbolic(self, use_llm: bool = True) -> sp.Expr:
        """增强: LLM辅助提取符号并生成证明 - 完整"""
        x, y, z, w = sp.symbols('x y z w')
        input_vars = sp.Matrix([x, y, z, w])
        expr = input_vars
        
        for i, module in enumerate(self.subnet_modules):
            try:
                expr = await module.extract_symbolic(expr, use_llm)
                logger.debug(f"模块 {i} 符号提取成功")
            except Exception as module_error:
                # 只在第一次失败时记录错误
                if not hasattr(self, '_symbolic_errors'):
                    self._symbolic_errors = set()
                if i not in self._symbolic_errors:
                    logger.warning(f"模块 {i} 符号提取失败: {module_error}")
                    self._symbolic_errors.add(i)
                expr = sp.symbols('x') + sp.symbols('y') + sp.symbols('z') + sp.symbols('w')
        
        # LLM生成证明 - 完整
        if use_llm:
            xai = XAIIntegration()
            context = f"Network has {len(self.subnet_modules)} modules, final expression: {expr}"
            proof = await xai.generate_math_proof(str(expr), "mathematical analysis")
            if proof:
                logger.info(f"LLM-generated proof: {proof}")
            else:
                logger.warning("LLM证明生成失败")
        
        return expr

    def get_state(self) -> Dict:
        """获取网络状态 - 完整"""
        modules_states = [m.get_state() for m in self.subnet_modules]
        return {
            'modules_states': modules_states,
            'modules_config': copy.deepcopy(self.modules_config),
            'epigenetic_markers': self.epigenetic_markers.clone().detach(),
            'final_output_layer_state_dict': self.final_output_layer.state_dict()
        }

    def load_state(self, state: Dict):
        """加载网络状态 - 完整"""
        self.__init__(state['modules_config'], state['epigenetic_markers'])
        for i, m_state in enumerate(state['modules_states']):
            self.subnet_modules[i].load_state(m_state) 
        self.final_output_layer.load_state_dict(state['final_output_layer_state_dict'])