import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch

class EvolutionVisualizer:
    """进化过程可视化器"""
    
    def __init__(self, output_dir: str = "evolution_plots"):
        self.output_dir = output_dir
        self.evolution_history = []
        self.diversity_history = []
        self.fitness_history = []
        self.structure_history = []
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def record_generation(self, generation: int, population: List, 
                         fitness_scores: List[float], diversity: float,
                         best_fitness: float, avg_fitness: float):
        """记录一代的进化数据"""
        # 记录基本指标
        self.evolution_history.append({
            'generation': generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'diversity': diversity,
            'population_size': len(population)
        })
        
        # 记录多样性历史
        self.diversity_history.append(diversity)
        
        # 记录适应度历史
        self.fitness_history.append({
            'best': best_fitness,
            'avg': avg_fitness,
            'all': fitness_scores
        })
        
        # 记录结构多样性
        structure_info = self._analyze_population_structure(population)
        self.structure_history.append(structure_info)
    
    def _analyze_population_structure(self, population: List) -> Dict:
        """分析种群结构多样性"""
        structures = []
        for model in population:
            # 分析模块化网络的结构
            structure = {
                'num_modules': len(model.subnet_modules),
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'module_types': [getattr(module, 'module_type', 'unknown') for module in model.subnet_modules],
                'output_dims': [getattr(module, 'output_dim', 0) for module in model.subnet_modules]
            }
            structures.append(structure)
        
        # 计算结构多样性指标
        num_modules = [s['num_modules'] for s in structures]
        total_parameters = [s['total_parameters'] for s in structures]
        module_types = [s['module_types'] for s in structures]
        
        return {
            'unique_module_counts': len(set(num_modules)),
            'unique_parameter_counts': len(set(total_parameters)),
            'unique_module_type_combinations': len(set(str(mt) for mt in module_types)),
            'structure_diversity': len(set(str(s) for s in structures))
        }
    
    def plot_evolution_curves(self, save_plot: bool = True) -> str:
        """绘制进化曲线"""
        if not self.evolution_history:
            return "无进化数据可绘制"
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        generations = [h['generation'] for h in self.evolution_history]
        best_fitness = [h['best_fitness'] for h in self.evolution_history]
        avg_fitness = [h['avg_fitness'] for h in self.evolution_history]
        diversity = [h['diversity'] for h in self.evolution_history]
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 适应度进化曲线
        ax1.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        ax1.plot(generations, avg_fitness, 'r--', label='Average Fitness', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Evolution Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 多样性变化曲线
        ax2.plot(generations, diversity, 'g-', label='Population Diversity', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Diversity')
        ax2.set_title('Diversity Change Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 结构多样性
        if self.structure_history:
            structure_diversity = [s['structure_diversity'] for s in self.structure_history]
            ax3.plot(generations, structure_diversity, 'm-', label='Structure Diversity', linewidth=2)
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Structure Diversity')
            ax3.set_title('Structure Diversity Change')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 适应度分布箱线图
        if len(self.fitness_history) > 1:
            fitness_data = [h['all'] for h in self.fitness_history]
            ax4.boxplot(fitness_data, tick_labels=[f'G{i+1}' for i in range(len(fitness_data))])
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Fitness Distribution')
            ax4.set_title('Fitness Distribution Boxplot')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evolution_curves_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return filepath
        else:
            plt.show()
            return "显示图表"
    
    def plot_diversity_heatmap(self, save_plot: bool = True) -> str:
        """绘制多样性热力图"""
        if not self.structure_history:
            return "无结构数据可绘制"
        
        # 准备数据
        generations = list(range(len(self.structure_history)))
        metrics = ['unique_hidden_sizes', 'unique_reasoning_layers', 'unique_attention_heads']
        
        data = np.zeros((len(metrics), len(generations)))
        for i, metric in enumerate(metrics):
            for j, history in enumerate(self.structure_history):
                data[i, j] = history[metric]
        
        # 绘制热力图
        plt.figure(figsize=(12, 6))
        im = plt.imshow(data, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im)
        
        plt.yticks(range(len(metrics)), metrics)
        plt.xticks(range(len(generations)), [f'G{i+1}' for i in generations])
        plt.xlabel('Generation')
        plt.ylabel('Diversity Metrics')
        plt.title('Structure Diversity Heatmap')
        
        # 添加数值标签
        for i in range(len(metrics)):
            for j in range(len(generations)):
                plt.text(j, i, f'{data[i, j]:.0f}', ha='center', va='center')
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diversity_heatmap_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return filepath
        else:
            plt.show()
            return "显示图表"
    
    def generate_evolution_report(self) -> str:
        """生成进化报告"""
        if not self.evolution_history:
            return "无进化数据可生成报告"
        
        report = {
            'summary': {
                'total_generations': len(self.evolution_history),
                'final_best_fitness': self.evolution_history[-1]['best_fitness'],
                'final_avg_fitness': self.evolution_history[-1]['avg_fitness'],
                'final_diversity': self.evolution_history[-1]['diversity'],
                'improvement': self.evolution_history[-1]['best_fitness'] - self.evolution_history[0]['best_fitness']
            },
            'evolution_history': self.evolution_history,
            'diversity_history': self.diversity_history,
            'fitness_history': self.fitness_history,
            'structure_history': self.structure_history
        }
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evolution_report_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def save_visualization_data(self) -> str:
        """保存可视化数据"""
        data = {
            'evolution_history': self.evolution_history,
            'diversity_history': self.diversity_history,
            'fitness_history': self.fitness_history,
            'structure_history': self.structure_history
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visualization_data_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath