#!/usr/bin/env python3
"""
优化的可视化系统 - 解决数据质量和存储效率问题
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import gzip
import pickle
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch
from pathlib import Path
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class OptimizedEvolutionVisualizer:
    """优化的进化过程可视化器"""
    
    def __init__(self, output_dir: str = "evolution_plots", 
                 max_files: int = 20,  # 最大保留文件数
                 compression: bool = True,  # 是否启用压缩
                 dpi: int = 150):  # 图片DPI
        self.output_dir = Path(output_dir)
        self.max_files = max_files
        self.compression = compression
        self.dpi = dpi
        
        # 数据存储
        self.evolution_history = []
        self.diversity_history = []
        self.fitness_history = []
        self.structure_history = []
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 清理旧文件
        self._cleanup_old_files()
    
    def _cleanup_old_files(self):
        """清理旧文件，只保留最新的N个文件"""
        try:
            # 获取所有文件
            all_files = list(self.output_dir.glob("*"))
            
            if len(all_files) > self.max_files:
                # 按修改时间排序
                all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                # 删除旧文件
                files_to_delete = all_files[self.max_files:]
                for file_path in files_to_delete:
                    try:
                        file_path.unlink()
                        self.logger.info(f"删除旧文件: {file_path.name}")
                    except Exception as e:
                        self.logger.warning(f"删除文件失败 {file_path.name}: {e}")
                
                self.logger.info(f"清理完成，删除了 {len(files_to_delete)} 个旧文件")
        except Exception as e:
            self.logger.warning(f"清理旧文件失败: {e}")
    
    def _validate_data(self, data: Any) -> Any:
        """验证和清理数据"""
        if isinstance(data, dict):
            return {k: self._validate_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._validate_data(item) for item in data]
        elif isinstance(data, (int, float)):
            # 处理NaN值
            if np.isnan(data) or np.isinf(data):
                return 0.0
            return data
        else:
            return data
    
    def _compress_data(self, data: Dict) -> bytes:
        """压缩数据"""
        if self.compression:
            json_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
            return gzip.compress(json_str.encode('utf-8'))
        else:
            return json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')
    
    def _decompress_data(self, compressed_data: bytes) -> Dict:
        """解压数据"""
        if self.compression:
            json_str = gzip.decompress(compressed_data).decode('utf-8')
            return json.loads(json_str)
        else:
            return json.loads(compressed_data.decode('utf-8'))
    
    def record_generation(self, generation: int, population: List, 
                         fitness_scores: List[float], diversity: float,
                         best_fitness: float, avg_fitness: float):
        """记录一代的进化数据（优化版本）"""
        # 验证和清理数据
        best_fitness = self._validate_data(best_fitness)
        avg_fitness = self._validate_data(avg_fitness)
        diversity = self._validate_data(diversity)
        fitness_scores = [self._validate_data(score) for score in fitness_scores]
        
        # 检查数据是否有效
        if not (np.isfinite(best_fitness) and np.isfinite(avg_fitness) and np.isfinite(diversity)):
            self.logger.warning(f"第{generation}代数据包含无效值，跳过记录")
            return
        
        # 如果best_fitness是NaN，使用avg_fitness替代
        if np.isnan(best_fitness) and np.isfinite(avg_fitness):
            best_fitness = avg_fitness
        
        # 检查是否与上一代数据重复
        if self.evolution_history:
            last_gen = self.evolution_history[-1]
            if (abs(last_gen['best_fitness'] - best_fitness) < 1e-6 and
                abs(last_gen['avg_fitness'] - avg_fitness) < 1e-6 and
                abs(last_gen['diversity'] - diversity) < 1e-6):
                self.logger.info(f"第{generation}代数据与上一代重复，跳过记录")
                return
        
        # 记录基本指标
        self.evolution_history.append({
            'generation': generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'diversity': diversity,
            'population_size': len(population),
            'timestamp': datetime.now().isoformat()
        })
        
        # 记录多样性历史
        self.diversity_history.append(diversity)
        
        # 记录适应度历史（只保留统计信息，不保留所有个体数据）
        self.fitness_history.append({
            'best': best_fitness,
            'avg': avg_fitness,
            'std': np.std(fitness_scores),
            'min': np.min(fitness_scores),
            'max': np.max(fitness_scores),
            'count': len(fitness_scores)
        })
        
        # 记录结构多样性（简化版本）
        structure_info = self._analyze_population_structure_optimized(population)
        self.structure_history.append(structure_info)
        
        self.logger.info(f"记录第{generation}代数据: 最佳={best_fitness:.4f}, 平均={avg_fitness:.4f}, 多样性={diversity:.4f}")
    
    def _analyze_population_structure_optimized(self, population: List) -> Dict:
        """优化的种群结构分析"""
        try:
            # 提取结构参数
            hidden_sizes = []
            reasoning_layers = []
            attention_heads = []
            
            for individual in population:
                if hasattr(individual, 'hidden_size'):
                    hidden_sizes.append(individual.hidden_size)
                if hasattr(individual, 'reasoning_layers'):
                    reasoning_layers.append(individual.reasoning_layers)
                if hasattr(individual, 'attention_heads'):
                    attention_heads.append(individual.attention_heads)
            
            # 计算多样性指标
            structure_info = {
                'unique_hidden_sizes': len(set(hidden_sizes)) if hidden_sizes else 0,
                'unique_reasoning_layers': len(set(reasoning_layers)) if reasoning_layers else 0,
                'unique_attention_heads': len(set(attention_heads)) if attention_heads else 0,
                'structure_diversity': len(set(hidden_sizes)) + len(set(reasoning_layers)) + len(set(attention_heads))
            }
            
            return structure_info
            
        except Exception as e:
            self.logger.warning(f"结构分析失败: {e}")
            return {
                'unique_hidden_sizes': 0,
                'unique_reasoning_layers': 0,
                'unique_attention_heads': 0,
                'structure_diversity': 0
            }
    
    def plot_evolution_curves_optimized(self, save_plot: bool = True) -> str:
        """优化的进化曲线绘制"""
        if not self.evolution_history:
            return "无进化数据可绘制"
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8), dpi=self.dpi)
        
        generations = [h['generation'] for h in self.evolution_history]
        best_fitness = [h['best_fitness'] for h in self.evolution_history]
        avg_fitness = [h['avg_fitness'] for h in self.evolution_history]
        diversity = [h['diversity'] for h in self.evolution_history]
        
        # 1. 适应度进化曲线
        ax1.plot(generations, best_fitness, 'b-', label='最佳适应度', linewidth=2, marker='o', markersize=4)
        ax1.plot(generations, avg_fitness, 'r--', label='平均适应度', linewidth=2, marker='s', markersize=4)
        ax1.set_xlabel('代数')
        ax1.set_ylabel('适应度')
        ax1.set_title('适应度进化曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 多样性变化曲线
        ax2.plot(generations, diversity, 'g-', label='种群多样性', linewidth=2, marker='^', markersize=4)
        ax2.set_xlabel('代数')
        ax2.set_ylabel('多样性')
        ax2.set_title('多样性变化曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 结构多样性
        if self.structure_history:
            structure_diversity = [s['structure_diversity'] for s in self.structure_history]
            ax3.plot(generations, structure_diversity, 'm-', label='结构多样性', linewidth=2, marker='d', markersize=4)
            ax3.set_xlabel('代数')
            ax3.set_ylabel('结构多样性')
            ax3.set_title('结构多样性变化')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 适应度分布统计
        if len(self.fitness_history) > 1:
            fitness_ranges = [h['max'] - h['min'] for h in self.fitness_history]
            ax4.plot(generations, fitness_ranges, 'c-', label='适应度范围', linewidth=2, marker='*', markersize=4)
            ax4.set_xlabel('代数')
            ax4.set_ylabel('适应度范围')
            ax4.set_title('适应度分布范围')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimized_evolution_curves_{timestamp}.png"
            filepath = self.output_dir / filename
            
            # 优化保存参数
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            self.logger.info(f"进化曲线已保存: {filepath}")
            return str(filepath)
        else:
            plt.show()
            return "显示图表"
    
    def plot_diversity_heatmap_optimized(self, save_plot: bool = True) -> str:
        """优化的多样性热力图"""
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
        plt.figure(figsize=(10, 5), dpi=self.dpi)
        im = plt.imshow(data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        plt.colorbar(im)
        
        plt.yticks(range(len(metrics)), ['隐藏层大小', '推理层数', '注意力头数'])
        plt.xticks(range(len(generations)), [f'G{i+1}' for i in generations])
        plt.xlabel('代数')
        plt.ylabel('多样性指标')
        plt.title('结构多样性热力图')
        
        # 添加数值标签
        for i in range(len(metrics)):
            for j in range(len(generations)):
                plt.text(j, i, f'{data[i, j]:.0f}', ha='center', va='center', 
                        fontsize=8, color='black' if data[i, j] < 2 else 'white')
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimized_diversity_heatmap_{timestamp}.png"
            filepath = self.output_dir / filename
            
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            self.logger.info(f"多样性热力图已保存: {filepath}")
            return str(filepath)
        else:
            plt.show()
            return "显示图表"
    
    def generate_optimized_evolution_report(self) -> str:
        """生成优化的进化报告"""
        if not self.evolution_history:
            return "无进化数据可生成报告"
        
        # 计算统计信息
        final_gen = self.evolution_history[-1]
        initial_gen = self.evolution_history[0]
        
        report = {
            'summary': {
                'total_generations': len(self.evolution_history),
                'final_best_fitness': final_gen['best_fitness'],
                'final_avg_fitness': final_gen['avg_fitness'],
                'final_diversity': final_gen['diversity'],
                'improvement': final_gen['best_fitness'] - initial_gen['best_fitness'],
                'improvement_percentage': ((final_gen['best_fitness'] - initial_gen['best_fitness']) / initial_gen['best_fitness']) * 100 if initial_gen['best_fitness'] > 0 else 0,
                'generation_time': final_gen['timestamp']
            },
            'evolution_history': self.evolution_history,
            'diversity_history': self.diversity_history,
            'fitness_history': self.fitness_history,
            'structure_history': self.structure_history,
            'metadata': {
                'compression_used': self.compression,
                'max_files': self.max_files,
                'dpi': self.dpi,
                'generated_at': datetime.now().isoformat()
            }
        }
        
        # 保存报告（压缩版本）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimized_evolution_report_{timestamp}"
        
        if self.compression:
            filepath = self.output_dir / f"{filename}.json.gz"
            compressed_data = self._compress_data(report)
            with open(filepath, 'wb') as f:
                f.write(compressed_data)
        else:
            filepath = self.output_dir / f"{filename}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"优化进化报告已保存: {filepath}")
        return str(filepath)
    
    def save_optimized_visualization_data(self) -> str:
        """保存优化的可视化数据"""
        data = {
            'evolution_history': self.evolution_history,
            'diversity_history': self.diversity_history,
            'fitness_history': self.fitness_history,
            'structure_history': self.structure_history,
            'metadata': {
                'compression_used': self.compression,
                'data_points': len(self.evolution_history),
                'generated_at': datetime.now().isoformat()
            }
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimized_visualization_data_{timestamp}"
        
        if self.compression:
            filepath = self.output_dir / f"{filename}.json.gz"
            compressed_data = self._compress_data(data)
            with open(filepath, 'wb') as f:
                f.write(compressed_data)
        else:
            filepath = self.output_dir / f"{filename}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"优化可视化数据已保存: {filepath}")
        return str(filepath)
    
    def get_storage_statistics(self) -> Dict:
        """获取存储统计信息"""
        total_size = 0
        file_count = 0
        
        for file_path in self.output_dir.glob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        return {
            'total_files': file_count,
            'total_size_mb': total_size / (1024 * 1024),
            'avg_file_size_mb': (total_size / file_count) / (1024 * 1024) if file_count > 0 else 0,
            'compression_enabled': self.compression,
            'max_files_limit': self.max_files
        }

def create_optimized_visualizer(output_dir: str = "evolution_plots", 
                               max_files: int = 20,
                               compression: bool = True,
                               dpi: int = 150) -> OptimizedEvolutionVisualizer:
    """创建优化的可视化器"""
    return OptimizedEvolutionVisualizer(
        output_dir=output_dir,
        max_files=max_files,
        compression=compression,
        dpi=dpi
    ) 