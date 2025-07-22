import asyncio
import torch
from evolution.nsga2 import evolve_population_nsga2
from evolution.population import create_initial_population
from evaluators.realworld_evaluator import RealWorldEvaluator
from evaluators.symbolic_evaluator import SymbolicEvaluator
from utils.visualization import plot_evolution
from config.logging_setup import setup_logging
from config.global_constants import POPULATION_SIZE, NUM_GENERATIONS, LEVEL_DESCRIPTIONS

logger = setup_logging()

async def main():
    """主函数 - 运行完整的进化过程"""
    torch.manual_seed(42)  # 可复现
    
    # 初始化评估器
    realworld_evaluator = RealWorldEvaluator()
    symbolic_evaluator = SymbolicEvaluator()
    
    # 创建初始种群
    logger.info(f"创建初始种群 (大小: {POPULATION_SIZE})")
    initial_population = create_initial_population(POPULATION_SIZE)
    
    all_avg_scores = []
    all_best_scores = []
    population = initial_population
    
    # 运行7个级别的进化
    for level in range(7):
        logger.info(f"\n=== 级别 {level}: {LEVEL_DESCRIPTIONS[level]} 数学进化 ===")
        
        # 运行指定代数的进化
        population, score_history_avg, score_history_best = await evolve_population_nsga2(
            population, 
            NUM_GENERATIONS, 
            level
        )
        
        all_avg_scores.extend(score_history_avg)
        all_best_scores.extend(score_history_best)
        
        logger.info(f"级别 {level} 完成 - 平均得分: {score_history_avg[-1]:.4f}, 最佳得分: {score_history_best[-1]:.4f}")
    
    # 生成进化曲线图
    plot_evolution(all_avg_scores, all_best_scores)
    logger.info(f"\n=== 进化完成! 总共 {len(all_avg_scores)} 个世代 ===")
    logger.info(f"最终平均得分: {all_avg_scores[-1]:.4f}, 最终最佳得分: {all_best_scores[-1]:.4f}")

if __name__ == "__main__":
    asyncio.run(main())