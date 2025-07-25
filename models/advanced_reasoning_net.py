import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from models.base_module import SubNetModule

class AdvancedReasoningNet(nn.Module):
    """高级推理网络 - 具备更复杂的推理能力"""
    
    def __init__(self, input_size: int = 4, hidden_size: int = 256, 
                 reasoning_layers: int = 5, attention_heads: int = 8,
                 memory_size: int = 20, reasoning_types: int = 10):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reasoning_layers = reasoning_layers
        self.attention_heads = attention_heads
        self.memory_size = memory_size
        self.reasoning_types = reasoning_types
        
        # 1. 增强输入编码层
        self.input_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 2. 多头注意力机制
        # 确保embed_dim能被num_heads整除
        adjusted_hidden_size = (hidden_size // attention_heads) * attention_heads
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=adjusted_hidden_size,
            num_heads=attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 3. 高级推理层
        self.advanced_reasoning_layers = nn.ModuleList([
            AdvancedReasoningLayer(adjusted_hidden_size, adjusted_hidden_size) 
            for _ in range(reasoning_layers)
        ])
        
        # 4. 增强记忆模块
        self.enhanced_memory_module = EnhancedMemoryModule(adjusted_hidden_size, memory_size)
        
        # 5. 推理策略控制器
        self.reasoning_strategy_controller = ReasoningStrategyController(adjusted_hidden_size, reasoning_types)
        
        # 6. 高级符号模块
        self.advanced_symbolic_module = AdvancedSymbolicModule(adjusted_hidden_size)
        
        # 7. 推理链生成器
        self.reasoning_chain_generator = ReasoningChainGenerator(adjusted_hidden_size)
        
        # 8. 多任务输出层
        self.multi_task_outputs = nn.ModuleDict({
            'mathematical_logic': nn.Sequential(
                nn.Linear(adjusted_hidden_size, adjusted_hidden_size // 2),
                nn.ReLU(),
                nn.Linear(adjusted_hidden_size // 2, 1)
            ),
            'symbolic_reasoning': nn.Sequential(
                nn.Linear(adjusted_hidden_size, adjusted_hidden_size // 2),
                nn.ReLU(),
                nn.Linear(adjusted_hidden_size // 2, 1)
            ),
            'abstract_reasoning': nn.Sequential(
                nn.Linear(adjusted_hidden_size, adjusted_hidden_size // 2),
                nn.ReLU(),
                nn.Linear(adjusted_hidden_size // 2, 1)
            ),
            'pattern_recognition': nn.Sequential(
                nn.Linear(adjusted_hidden_size, adjusted_hidden_size // 2),
                nn.ReLU(),
                nn.Linear(adjusted_hidden_size // 2, 1)
            ),
            'reasoning_chain': nn.Sequential(
                nn.Linear(adjusted_hidden_size, adjusted_hidden_size // 2),
                nn.ReLU(),
                nn.Linear(adjusted_hidden_size // 2, 1)
            ),
            'mathematical_proof': nn.Sequential(
                nn.Linear(adjusted_hidden_size, adjusted_hidden_size // 2),
                nn.ReLU(),
                nn.Linear(adjusted_hidden_size // 2, 1)
            ),
            'logical_chain': nn.Sequential(
                nn.Linear(adjusted_hidden_size, adjusted_hidden_size // 2),
                nn.ReLU(),
                nn.Linear(adjusted_hidden_size // 2, 1)
            ),
            'abstract_concepts': nn.Sequential(
                nn.Linear(adjusted_hidden_size, adjusted_hidden_size // 2),
                nn.ReLU(),
                nn.Linear(adjusted_hidden_size // 2, 1)
            ),
            'creative_reasoning': nn.Sequential(
                nn.Linear(adjusted_hidden_size, adjusted_hidden_size // 2),
                nn.ReLU(),
                nn.Linear(adjusted_hidden_size // 2, 1)
            ),
            'comprehensive_reasoning': nn.Sequential(
                nn.Linear(adjusted_hidden_size, adjusted_hidden_size // 2),
                nn.ReLU(),
                nn.Linear(adjusted_hidden_size // 2, 1)
            ),
            'symbolic_expression': nn.Sequential(
                nn.Linear(adjusted_hidden_size, adjusted_hidden_size // 2),
                nn.ReLU(),
                nn.Linear(adjusted_hidden_size // 2, 1)
            )
        })
        
        # 9. 进化标记
        self.evolution_markers = nn.Parameter(torch.randn(adjusted_hidden_size))
        
        # 10. 自适应学习率
        self.adaptive_learning_rate = nn.Parameter(torch.tensor(0.001))
        
        # 11. 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        batch_size = x.size(0)
        
        # 1. 增强输入编码
        encoded = self.input_encoder(x)
        
        # 2. 多头注意力处理
        sequence = encoded.unsqueeze(1)  # [batch, 1, hidden]
        attended, attention_weights = self.multi_head_attention(sequence, sequence, sequence)
        attended = attended.squeeze(1)  # [batch, hidden]
        
        # 3. 高级推理处理
        reasoning_state = attended
        reasoning_steps = []
        
        for i, layer in enumerate(self.advanced_reasoning_layers):
            reasoning_state, step_info = layer(reasoning_state)
            reasoning_steps.append(step_info)
        
        # 4. 增强记忆更新
        try:
            memory_state = self.enhanced_memory_module(reasoning_state)
        except Exception as e:
            # 如果记忆模块失败，使用零张量
            memory_state = torch.zeros_like(reasoning_state)
        
        # 5. 推理策略控制
        try:
            controlled_state, strategy_info = self.reasoning_strategy_controller(
                reasoning_state, memory_state
            )
        except Exception as e:
            # 如果推理策略控制器失败，使用原始状态
            controlled_state = reasoning_state
            strategy_info = {'reasoning_type': 0, 'strategy_score': 0.5, 'confidence': 0.5}
        
        # 6. 高级符号推理
        symbolic_state = self.advanced_symbolic_module(controlled_state)
        
        # 7. 推理链生成
        chain_state = self.reasoning_chain_generator(controlled_state)
        
        # 8. 多任务输出
        outputs = {}
        for task_name, output_layer in self.multi_task_outputs.items():
            task_output = torch.sigmoid(output_layer(controlled_state))
            
            # 添加进化标记影响
            evolution_influence = torch.sigmoid(
                torch.dot(controlled_state.mean(0), self.evolution_markers)
            )
            
            # 自适应学习率影响
            learning_influence = torch.sigmoid(self.adaptive_learning_rate)
            
            outputs[task_name] = task_output + 0.1 * evolution_influence + 0.05 * learning_influence
        
        # 9. 添加推理链信息
        outputs['reasoning_chain_info'] = reasoning_steps
        outputs['strategy_info'] = strategy_info
        
        return outputs
    
    def extract_symbolic(self, use_llm: bool = True) -> str:
        """提取符号表达式"""
        try:
            if use_llm:
                return self.advanced_symbolic_module.extract_symbolic_expression()
            else:
                return self.advanced_symbolic_module.extract_from_weights()
        except Exception as e:
            return "x + y"
    
    def get_reasoning_chain(self) -> List[str]:
        """获取推理链"""
        return self.reasoning_chain_generator.get_reasoning_steps()
    
    def get_reasoning_strategy(self) -> Dict[str, Any]:
        """获取推理策略"""
        return self.reasoning_strategy_controller.get_strategy_info()
    
    def evolve(self, evolution_rate: float = 0.01) -> 'AdvancedReasoningNet':
        """进化模型"""
        new_model = type(self)(
            self.input_size, 
            self.hidden_size, 
            self.reasoning_layers, 
            self.attention_heads,
            self.memory_size,
            self.reasoning_types
        )
        
        # 复制参数
        for param, new_param in zip(self.parameters(), new_model.parameters()):
            with torch.no_grad():
                noise = torch.randn_like(param) * evolution_rate
                new_param.copy_(param + noise)
        
        return new_model

class AdvancedReasoningLayer(nn.Module):
    """高级推理层"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        
        # 自注意力 - 确保embed_dim能被num_heads整除
        num_heads = 4
        adjusted_hidden_size = (hidden_size // num_heads) * num_heads
        if adjusted_hidden_size < num_heads:
            adjusted_hidden_size = num_heads
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=adjusted_hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(0.1)
        )
        
        # 推理门控
        self.reasoning_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # 推理类型分类器
        self.reasoning_type_classifier = nn.Linear(hidden_size, 5)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # 残差连接
        residual = x
        
        # 第一个子层：自注意力
        x = self.layer_norm1(x)
        sequence = x.unsqueeze(1)
        attended, attention_weights = self.self_attention(sequence, sequence, sequence)
        x = attended.squeeze(1) + residual
        
        # 第二个子层：前馈网络
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x) + residual
        
        # 第三个子层：推理门控
        residual = x
        x = self.layer_norm3(x)
        gate = self.reasoning_gate(x)
        x = gate * x + (1 - gate) * residual
        
        # 推理类型分类
        reasoning_type = F.softmax(self.reasoning_type_classifier(x), dim=-1)
        
        step_info = {
            'reasoning_type': int(reasoning_type.argmax(dim=-1).float().mean().item()),
            'attention_weights': attention_weights,
            'gate_value': float(gate.mean().item())
        }
        
        return x, step_info

class EnhancedMemoryModule(nn.Module):
    """增强记忆模块"""
    
    def __init__(self, hidden_size: int, memory_size: int = 20):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        
        # 记忆存储
        self.memory = nn.Parameter(torch.randn(memory_size, hidden_size))
        
        # 记忆控制器
        self.memory_controller = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh()
        )
        
        # 记忆更新门
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size, memory_size),
            nn.Sigmoid()
        )
        
        # 记忆读取门
        self.read_gate = nn.Sequential(
            nn.Linear(hidden_size, memory_size),
            nn.Sigmoid()
        )
        
        # 记忆遗忘门
        self.forget_gate = nn.Sequential(
            nn.Linear(hidden_size, memory_size),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算记忆更新权重
        update_weights = self.update_gate(x)
        read_weights = self.read_gate(x)
        forget_weights = self.forget_gate(x)
        
        # 确保维度匹配
        batch_size = x.size(0)
        if update_weights.size(0) != batch_size:
            update_weights = update_weights[:batch_size]
        if read_weights.size(0) != batch_size:
            read_weights = read_weights[:batch_size]
        if forget_weights.size(0) != batch_size:
            forget_weights = forget_weights[:batch_size]
        
        # 更新记忆
        memory_update = torch.matmul(update_weights, self.memory)
        
        # 读取记忆
        memory_read = torch.matmul(read_weights, self.memory)
        
        # 遗忘记忆
        memory_forget = torch.matmul(forget_weights, self.memory)
        
        # 控制器处理
        controlled_memory = self.memory_controller(memory_update + memory_read - memory_forget)
        
        return controlled_memory

class ReasoningStrategyController(nn.Module):
    """推理策略控制器"""
    
    def __init__(self, hidden_size: int, reasoning_types: int = 10):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.reasoning_types = reasoning_types
        
        # 推理策略网络
        self.strategy_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 推理类型分类器
        self.reasoning_classifier = nn.Linear(hidden_size, reasoning_types)
        
        # 策略评估器
        self.strategy_evaluator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 策略历史
        self.strategy_history = []
    
    def forward(self, reasoning_state: torch.Tensor, memory_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # 合并推理状态和记忆状态
        combined = torch.cat([reasoning_state, memory_state], dim=-1)
        
        # 策略网络处理
        strategy_output = self.strategy_net(combined)
        
        # 推理类型分类
        reasoning_type = F.softmax(self.reasoning_classifier(strategy_output), dim=-1)
        
        # 策略评估
        strategy_score = torch.sigmoid(self.strategy_evaluator(strategy_output))
        
        # 记录策略信息
        strategy_info = {
            'reasoning_type': int(reasoning_type.argmax(dim=-1).float().mean().item()),
            'strategy_score': float(strategy_score.mean().item()),
            'confidence': float(reasoning_type.max(dim=-1)[0].mean().item())
        }
        
        self.strategy_history.append(strategy_info)
        
        return strategy_output, strategy_info
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        if not self.strategy_history:
            return {}
        
        recent_strategies = self.strategy_history[-10:]  # 最近10个策略
        
        return {
            'recent_strategies': recent_strategies,
            'strategy_count': len(self.strategy_history),
            'avg_confidence': np.mean([s['confidence'] for s in recent_strategies]),
            'avg_score': np.mean([s['strategy_score'] for s in recent_strategies])
        }

class AdvancedSymbolicModule(nn.Module):
    """高级符号推理模块"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # 符号提取网络
        self.symbol_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 符号组合网络
        self.symbol_combiner = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # 符号复杂度评估器
        self.complexity_evaluator = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 符号提取
        symbols = self.symbol_extractor(x)
        
        # 符号组合
        combined = self.symbol_combiner(symbols)
        
        # 复杂度评估
        complexity = self.complexity_evaluator(symbols)
        
        return combined + 0.1 * complexity
    
    def extract_symbolic_expression(self) -> str:
        """提取符号表达式（使用LLM）"""
        try:
            # 这里应该调用LLM API
            expressions = ["x + y", "x * y", "exp(x)", "sin(x)", "x^2 + y^2"]
            return np.random.choice(expressions)
        except Exception as e:
            return "x + y"
    
    def extract_from_weights(self) -> str:
        """从权重提取符号表达式"""
        try:
            weights = self.symbol_extractor[0].weight.data
            bias = self.symbol_extractor[0].bias.data
            
            # 分析权重模式来提取符号
            if weights.sum() > 0:
                if weights.std() > 0.5:
                    return "exp(x)"
                else:
                    return "x + y"
            else:
                return "x * y"
        except Exception as e:
            return "x + y"

class ReasoningChainGenerator(nn.Module):
    """推理链生成器"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # 推理步骤生成器
        self.step_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # 推理链评估器
        self.chain_evaluator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 推理步骤记录
        self.reasoning_steps = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 生成推理步骤
        step_score = self.step_generator(x)
        
        # 评估推理链
        chain_score = self.chain_evaluator(x)
        
        # 记录推理步骤
        self.reasoning_steps.append(f"推理步骤: {step_score.mean().item():.3f}")
        
        return step_score + 0.5 * chain_score
    
    def get_reasoning_steps(self) -> List[str]:
        """获取推理步骤"""
        return self.reasoning_steps.copy() 