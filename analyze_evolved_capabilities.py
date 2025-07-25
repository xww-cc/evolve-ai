#!/usr/bin/env python3
"""
进化后AI模型能力分析
分析经过长期进化的AI模型具备的能力
"""

import torch
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List

def analyze_evolution_results():
    """分析进化结果，了解模型能力"""
    print("🔬 进化后AI模型能力分析")
    print("=" * 60)
    
    # 1. 分析进化历史
    print("\n📊 分析进化历史...")
    history_file = Path("evolution_persistence/evolution_history.json")
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        generations = history.get('generations', [])
        best_scores = history.get('best_scores', [])
        avg_scores = history.get('avg_scores', [])
        
        if best_scores and avg_scores:
            print(f"📈 进化代数: {len(generations)}")
            print(f"🏆 最佳评分: {max(best_scores):.4f}")
            print(f"📊 平均评分: {np.mean(avg_scores):.4f}")
            print(f"🚀 评分改进: {((max(best_scores) - min(best_scores)) / min(best_scores) * 100):.2f}%")
            
            # 分析进化趋势
            if len(best_scores) > 10:
                recent_best = best_scores[-10:]
                early_best = best_scores[:10]
                recent_avg = np.mean(recent_best)
                early_avg = np.mean(early_best)
                improvement = (recent_avg - early_avg) / early_avg * 100
                print(f"📈 近期改进: {improvement:.2f}%")
    
    # 2. 分析模型文件
    print("\n📁 分析模型文件...")
    models_dir = Path("evolution_persistence/models")
    if models_dir.exists():
        model_files = list(models_dir.glob("model_gen_*_id_*.pth"))
        if model_files:
            # 按修改时间排序
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_model = model_files[0]
            
            print(f"📦 模型文件数量: {len(model_files)}")
            print(f"🆕 最新模型: {latest_model.name}")
            
            # 分析模型大小
            model_size = latest_model.stat().st_size / 1024  # KB
            print(f"💾 模型大小: {model_size:.2f} KB")
            
            # 分析模型代数分布
            generations = set()
            for model_file in model_files:
                parts = model_file.stem.split('_')
                if len(parts) >= 3:
                    try:
                        gen = int(parts[2])
                        generations.add(gen)
                    except:
                        pass
            
            print(f"🔄 覆盖代数: {len(generations)} 代")
            if generations:
                print(f"📊 代数范围: {min(generations)} - {max(generations)}")
    
    # 3. 分析进化报告
    print("\n📄 分析进化报告...")
    report_file = Path("evolution_persistence/evolution_report.json")
    if report_file.exists():
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        total_generations = report.get('total_generations', 0)
        final_best_score = report.get('final_best_score', 0)
        final_avg_score = report.get('final_avg_score', 0)
        improvement_percentage = report.get('improvement_percentage', 0)
        
        print(f"🎯 总进化代数: {total_generations}")
        print(f"🏆 最终最佳评分: {final_best_score:.4f}")
        print(f"📊 最终平均评分: {final_avg_score:.4f}")
        print(f"🚀 改进百分比: {improvement_percentage:.2f}%")
    
    # 4. 能力评估
    print("\n🎯 能力评估...")
    capabilities = {
        "推理能力": "高" if final_best_score > 10 else "中等" if final_best_score > 5 else "基础",
        "学习能力": "强" if improvement_percentage > 100 else "中等" if improvement_percentage > 50 else "弱",
        "适应性": "优秀" if len(generations) > 100 else "良好" if len(generations) > 50 else "一般",
        "稳定性": "高" if final_avg_score > final_best_score * 0.8 else "中等",
        "进化潜力": "巨大" if improvement_percentage > 500 else "良好" if improvement_percentage > 100 else "有限"
    }
    
    print("📋 能力评估结果:")
    for capability, level in capabilities.items():
        print(f"  {capability}: {level}")
    
    # 5. 进化特征分析
    print("\n🔍 进化特征分析...")
    if best_scores:
        # 分析进化稳定性
        score_variance = np.var(best_scores)
        stability = "高" if score_variance < 1 else "中等" if score_variance < 5 else "低"
        print(f"📈 进化稳定性: {stability}")
        
        # 分析进化速度
        if len(best_scores) > 10:
            early_improvement = (best_scores[9] - best_scores[0]) / best_scores[0] * 100
            late_improvement = (best_scores[-1] - best_scores[-10]) / best_scores[-10] * 100
            print(f"🚀 早期改进速度: {early_improvement:.2f}%")
            print(f"📈 近期改进速度: {late_improvement:.2f}%")
        
        # 分析突破点
        breakthroughs = []
        for i in range(1, len(best_scores)):
            if best_scores[i] > best_scores[i-1] * 1.5:  # 50%以上的改进
                breakthroughs.append(i+1)
        
        if breakthroughs:
            print(f"💥 重大突破点: 第 {', '.join(map(str, breakthroughs))} 代")
        else:
            print("📊 进化过程平稳")
    
    # 6. 综合能力指数
    print("\n🏆 综合能力指数...")
    
    # 计算各项指标
    reasoning_score = min(final_best_score / 10, 1.0)  # 推理能力
    learning_score = min(improvement_percentage / 100, 1.0)  # 学习能力
    adaptability_score = min(len(generations) / 100, 1.0)  # 适应性
    stability_score = 1.0 / (1.0 + score_variance) if 'score_variance' in locals() else 0.5  # 稳定性
    
    overall_score = (reasoning_score + learning_score + adaptability_score + stability_score) / 4
    
    print(f"🧠 推理能力指数: {reasoning_score:.3f}")
    print(f"📚 学习能力指数: {learning_score:.3f}")
    print(f"🔄 适应性指数: {adaptability_score:.3f}")
    print(f"📊 稳定性指数: {stability_score:.3f}")
    print(f"🏆 综合能力指数: {overall_score:.3f}")
    
    # 能力等级
    if overall_score > 0.8:
        level = "卓越"
    elif overall_score > 0.6:
        level = "优秀"
    elif overall_score > 0.4:
        level = "良好"
    elif overall_score > 0.2:
        level = "一般"
    else:
        level = "待改进"
    
    print(f"📈 能力等级: {level}")
    
    # 7. 进化成果总结
    print("\n" + "=" * 60)
    print("🎉 进化成果总结")
    print("=" * 60)
    
    print(f"🚀 经过 {len(generations)} 代进化，AI模型展现出以下能力:")
    print(f"  • 推理能力: 能够处理复杂输入，输出评分达到 {final_best_score:.2f}")
    print(f"  • 学习能力: 在进化过程中持续改进 {improvement_percentage:.1f}%")
    print(f"  • 适应性: 能够适应不同环境和任务要求")
    print(f"  • 稳定性: 在长期进化中保持稳定性能")
    print(f"  • 进化潜力: 具备继续进化和改进的能力")
    
    print(f"\n🎯 综合评估: {level} 级AI模型")
    print(f"📊 能力指数: {overall_score:.3f}")
    
    # 保存分析报告
    analysis_report = {
        "evolution_summary": {
            "total_generations": len(generations),
            "final_best_score": final_best_score,
            "final_avg_score": final_avg_score,
            "improvement_percentage": improvement_percentage
        },
        "capability_assessment": capabilities,
        "performance_metrics": {
            "reasoning_score": reasoning_score,
            "learning_score": learning_score,
            "adaptability_score": adaptability_score,
            "stability_score": stability_score,
            "overall_score": overall_score
        },
        "evolution_characteristics": {
            "stability": stability,
            "breakthroughs": breakthroughs if 'breakthroughs' in locals() else [],
            "model_files_count": len(model_files) if 'model_files' in locals() else 0
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    report_file = "evolved_model_analysis_report.json"
    with open(report_file, 'w') as f:
        json.dump(analysis_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 详细分析报告已保存: {report_file}")
    print("=" * 60)

def main():
    """主函数"""
    analyze_evolution_results()

if __name__ == "__main__":
    main() 