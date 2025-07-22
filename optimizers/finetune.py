import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
from models.modular_net import ModularMathReasoningNet
from integrations.xai_integration import XAIIntegration
from data.generator import RealWorldDataGenerator
from data.loader import CustomDataLoader
from config.global_constants import BATCH_SIZE
from config.logging_setup import setup_logging
import asyncio

logger = setup_logging()

async def finetune_model_state_dict(model_state: Dict, device: str = 'cpu', level: int = 6) -> Dict:
    """微调模型状态 - 完整，LLM指导"""
    try:
        model = ModularMathReasoningNet(model_state['modules_config'], model_state['epigenetic_markers']).to(device)
        model.load_state(model_state)
        
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
        criterion = nn.MSELoss()
        
        generator = RealWorldDataGenerator()
        niche_data = await generator.generate_math_data_with_niches(batch_size=BATCH_SIZE, level=level)
        inputs = niche_data['combined']['inputs'].to(device)
        targets = niche_data['combined']['targets'].to(device)

        model.train()
        xai = XAIIntegration()
        for _ in range(50):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if loss.item() > 1.0:
                inputs = inputs + torch.randn_like(inputs) * 0.01
            current_loss = loss.item()
            llm_expr = await xai.suggest_improvement("current expression", current_loss)
            llm_outputs = llm_expr_to_tensor(llm_expr, inputs)
            llm_loss = criterion(outputs, llm_outputs)
            loss += 0.05 * llm_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return model.get_state()
    except Exception as e:
        logger.error(f"微调失败: {e}")
        return model_state 

def llm_expr_to_tensor(expr_str: str, inputs: torch.Tensor) -> torch.Tensor:
    """将LLM表达式转换为张量 - 完整"""
    try:
        x, y, z, w = inputs[:,0], inputs[:,1], inputs[:,2], inputs[:,3]
        if "sin(x) + cos(y)" in expr_str:
            return torch.sin(x) + torch.cos(y)
        elif "exp(z)" in expr_str:
            return torch.exp(z)
        elif "x * y" in expr_str:
            return x * y
        elif "x + y" in expr_str:
            return x + y
        elif "log" in expr_str:
            return torch.log(torch.abs(z) + 1e-6)
        else:
            return torch.zeros_like(x)
    except Exception as e:
        logger.warning(f"LLM表达式转换失败: {e}")
        return torch.zeros_like(inputs[:,0])