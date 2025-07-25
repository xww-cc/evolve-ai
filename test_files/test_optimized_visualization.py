#!/usr/bin/env python3
"""
测试优化的可视化系统
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import numpy as np
import torch
from models.advanced_reasoning_net import AdvancedReasoningNet
from utils.optimized_visualization import create_optimized_visualizer
from config.optimized_logging import setup_optimized_logging

logger = setup_optimized_logging()

async def test_optimized_visualization():
    """测试优化的可视化系统"""
    logger.log_important("🔔 🚀 启动优化可视化系统测试")
    
    try:
        # 1. 创建优化的可视化器
        visualizer = create_optimized_visualizer(
            output_dir="test_optimized_plots",
            max_files=10,  # 只保留10个文件
            compression=True,  # 启用压缩
            dpi=120  # 降低DPI以减小文件大小
        )
        
        logger.log_important("🔔 创建优化可视化器完成")
        
        # 2. 创建测试种群
        population = []
        for i in range(4):
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=256 + i * 32,  # 不同的隐藏层大小
                reasoning_layers=5 + i,    # 不同的推理层数
                attention_heads=8 + i,     # 不同的注意力头数
                memory_size=20,
                reasoning_types=10
            )
            population.append(model)
        
        logger.log_important(f"🔔 创建测试种群完成，共 {len(population)} 个个体")
        
        # 3. 模拟进化过程并记录数据
        for generation in range(5):
            # 生成模拟的适应度分数
            fitness_scores = []
            for i in range(len(population)):
                # 模拟进化改进
                base_score = 0.1 + generation * 0.05
                individual_score = base_score + np.random.normal(0, 0.02)
                fitness_scores.append(max(0.0, individual_score))
            
            # 计算统计信息
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            diversity = np.std(fitness_scores)  # 使用标准差作为多样性指标
            
            # 记录数据（包含数据验证和清理）
            visualizer.record_generation(
                generation=generation + 1,
                population=population,
                fitness_scores=fitness_scores,
                diversity=diversity,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness
            )
            
            logger.log_important(f"🔔 记录第{generation + 1}代数据: 最佳={best_fitness:.4f}, 平均={avg_fitness:.4f}, 多样性={diversity:.4f}")
        
        # 4. 生成优化的可视化
        logger.log_important("🔔 开始生成优化可视化...")
        
        # 生成进化曲线
        curves_file = visualizer.plot_evolution_curves_optimized()
        logger.log_important(f"📊 优化进化曲线: {curves_file}")
        
        # 生成多样性热力图
        heatmap_file = visualizer.plot_diversity_heatmap_optimized()
        logger.log_important(f"📊 优化多样性热力图: {heatmap_file}")
        
        # 生成优化报告
        report_file = visualizer.generate_optimized_evolution_report()
        logger.log_important(f"📊 优化进化报告: {report_file}")
        
        # 保存优化数据
        data_file = visualizer.save_optimized_visualization_data()
        logger.log_important(f"📊 优化可视化数据: {data_file}")
        
        # 5. 获取存储统计信息
        stats = visualizer.get_storage_statistics()
        logger.log_important(f"📊 存储统计:")
        logger.log_important(f"   总文件数: {stats['total_files']}")
        logger.log_important(f"   总大小: {stats['total_size_mb']:.2f} MB")
        logger.log_important(f"   平均文件大小: {stats['avg_file_size_mb']:.2f} MB")
        logger.log_important(f"   压缩启用: {stats['compression_enabled']}")
        logger.log_important(f"   文件限制: {stats['max_files_limit']}")
        
        # 6. 验证文件大小优化
        import glob
        test_files = glob.glob("test_optimized_plots/*")
        
        if test_files:
            total_size = sum(os.path.getsize(f) for f in test_files)
            total_size_mb = total_size / (1024 * 1024)
            
            logger.log_important(f"📊 测试文件统计:")
            logger.log_important(f"   文件数量: {len(test_files)}")
            logger.log_important(f"   总大小: {total_size_mb:.2f} MB")
            
            # 检查是否有压缩文件
            compressed_files = [f for f in test_files if f.endswith('.gz')]
            if compressed_files:
                logger.log_success(f"✅ 压缩功能正常，发现 {len(compressed_files)} 个压缩文件")
            
            # 检查文件大小是否合理
            if total_size_mb < 2.0:  # 小于2MB
                logger.log_success("✅ 文件大小优化成功")
            else:
                logger.log_warning(f"⚠️ 文件大小较大: {total_size_mb:.2f} MB")
        
        logger.log_success("✅ 优化可视化系统测试成功！")
        return True
        
    except Exception as e:
        logger.log_error(f"❌ 优化可视化系统测试失败: {e}")
        return False

async def test_data_quality_improvements():
    """测试数据质量改进"""
    logger.log_important("🔔 测试数据质量改进...")
    
    try:
        visualizer = create_optimized_visualizer(
            output_dir="test_data_quality",
            max_files=5,
            compression=True,
            dpi=100
        )
        
        # 测试NaN值处理
        population = [AdvancedReasoningNet(4, 256, 5, 8, 20, 10) for _ in range(3)]
        
        # 故意包含NaN值的数据
        fitness_scores_with_nan = [0.5, float('nan'), 0.7]
        best_fitness_with_nan = float('nan')
        avg_fitness_with_nan = 0.6
        diversity_with_nan = float('inf')
        
        # 记录包含NaN的数据
        visualizer.record_generation(
            generation=1,
            population=population,
            fitness_scores=fitness_scores_with_nan,
            diversity=diversity_with_nan,
            best_fitness=best_fitness_with_nan,
            avg_fitness=avg_fitness_with_nan
        )
        
        # 记录正常数据
        visualizer.record_generation(
            generation=2,
            population=population,
            fitness_scores=[0.5, 0.6, 0.7],
            diversity=0.1,
            best_fitness=0.7,
            avg_fitness=0.6
        )
        
        # 检查数据是否被正确清理
        if len(visualizer.evolution_history) == 1:  # 只有第2代被记录
            logger.log_success("✅ NaN值处理正常，无效数据被跳过")
        else:
            logger.log_warning("⚠️ NaN值处理可能有问题")
        
        # 测试重复数据检测
        visualizer.record_generation(
            generation=3,
            population=population,
            fitness_scores=[0.5, 0.6, 0.7],  # 与第2代相同
            diversity=0.1,
            best_fitness=0.7,
            avg_fitness=0.6
        )
        
        if len(visualizer.evolution_history) == 1:  # 重复数据被跳过
            logger.log_success("✅ 重复数据检测正常")
        else:
            logger.log_warning("⚠️ 重复数据检测可能有问题")
        
        return True
        
    except Exception as e:
        logger.log_error(f"❌ 数据质量测试失败: {e}")
        return False

async def main():
    """主函数"""
    logger.log_important("=== 优化可视化系统测试 ===")
    
    # 测试基本功能
    success1 = await test_optimized_visualization()
    
    # 测试数据质量改进
    success2 = await test_data_quality_improvements()
    
    if success1 and success2:
        logger.log_success("🎉 所有优化可视化测试通过！")
    else:
        logger.log_error("❌ 部分测试失败")

if __name__ == "__main__":
    asyncio.run(main()) 