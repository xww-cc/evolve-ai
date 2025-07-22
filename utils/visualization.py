import matplotlib.pyplot as plt
from datetime import datetime
from typing import List
from config.logging_setup import setup_logging

logger = setup_logging()

def plot_evolution(history_avg: List[float], history_best: List[float]):
    """可视化进化曲线 - 完整"""
    plt.figure(figsize=(12, 7))
    plt.plot(range(1, len(history_avg) + 1), history_avg, marker='o', linestyle='-', label='Average Performance Score', alpha=0.7)
    plt.plot(range(1, len(history_best) + 1), history_best, marker='x', linestyle='--', label='Best Performance Score', alpha=0.7)
    plt.title('Evolution Curve: Average and Best Performance Scores per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Performance Score (Negative MSE)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"evolution_curve_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    logger.info(f"可视化图表已保存: {plot_filename}")
    plt.close()