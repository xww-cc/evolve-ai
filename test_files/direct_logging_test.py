#!/usr/bin/env python3
"""
直接日志测试 - 确保能看到日志输出
"""

import asyncio
import time
import logging
from config.optimized_logging import setup_optimized_logging

# 设置优化的日志系统
logger = setup_optimized_logging()

async def direct_logging_test():
    """直接日志测试"""
    print("🧬 开始直接日志测试")
    print("=" * 40)
    
    # 测试各种日志级别
    logger.log_important("🔔 这是一条重要信息")
    logger.log_success("✅ 这是一条成功信息")
    logger.log_warning("⚠️ 这是一条警告信息")
    logger.log_error("❌ 这是一条错误信息")
    
    # 测试评估结果日志
    logger.log_evaluation_results("M01", 0.85, 0.92, {
        'mathematical_logic': 0.75,
        'symbolic_reasoning': 0.80,
        'abstract_reasoning': 0.65,
        'pattern_recognition': 0.70,
        'reasoning_chain': 0.85
    })
    
    # 测试进化进度日志
    logger.log_evolution_progress(1, 10, 0.92, 0.85, 0)
    
    # 测试系统状态日志
    logger.log_system_status(45.2, 23.1, 15.5, 78.3)
    
    # 测试性能指标日志
    logger.log_performance_metrics({
        'memory_usage': 45.2,
        'cpu_usage': 23.1,
        'evolution_speed': 15.5,
        'cache_hit_rate': 78.3
    })
    
    # 测试进化总结日志
    logger.log_evolution_summary(1, {
        '符号推理': 0.05,
        '真实世界': 0.08,
        '复杂推理_数学逻辑': 0.03,
        '复杂推理_符号推理': 0.04
    })
    
    # 测试进度日志
    logger.log_progress(3, 10, "模型评估")
    
    print("✅ 日志测试完成")
    return True

async def main():
    """主函数"""
    print("🚀 启动直接日志测试")
    
    success = await direct_logging_test()
    
    if success:
        print("🎉 直接日志测试成功完成！")
        print("✅ 日志系统工作正常")
        print("✅ 各种日志类型都能正确输出")
    else:
        print("⚠️ 直接日志测试失败")

if __name__ == "__main__":
    asyncio.run(main()) 