#!/usr/bin/env python3
"""
系统状态检查脚本
用于检查系统运行状态和健康状况
"""

import asyncio
import time
import psutil
import torch
import platform
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from config.logging_setup import setup_logging

logger = setup_logging()

@dataclass
class SystemStatus:
    """系统状态数据类"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    gpu_memory_percent: float
    gpu_memory_used_mb: float
    network_connections: int
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: str

class SystemStatusChecker:
    """系统状态检查器"""
    
    def __init__(self):
        self.logger = logger
        self.status_history: List[SystemStatus] = []
        
    async def check_system_resources(self) -> Dict[str, Any]:
        """检查系统资源"""
        try:
            self.logger.info("检查系统资源...")
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # GPU使用情况
            gpu_memory_percent = 0
            gpu_memory_used_mb = 0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated()
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_percent = (gpu_memory / gpu_memory_total) * 100
                gpu_memory_used_mb = gpu_memory / (1024 * 1024)
            
            # 网络连接数
            try:
                network_connections = len(psutil.net_connections())
            except psutil.AccessDenied:
                network_connections = 0
            
            # 系统信息
            python_version = sys.version
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else "N/A"
            
            # 创建状态记录
            status = SystemStatus(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                gpu_memory_percent=gpu_memory_percent,
                gpu_memory_used_mb=gpu_memory_used_mb,
                network_connections=network_connections,
                python_version=python_version,
                torch_version=torch_version,
                cuda_available=cuda_available,
                cuda_version=cuda_version
            )
            
            self.status_history.append(status)
            
            # 评估系统健康状态
            health_status = self._evaluate_health_status(status)
            
            return {
                "status": "success",
                "data": status,
                "health": health_status
            }
            
        except Exception as e:
            self.logger.error(f"系统资源检查失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _evaluate_health_status(self, status: SystemStatus) -> Dict[str, Any]:
        """评估系统健康状态"""
        health_issues = []
        warnings = []
        
        # CPU检查
        if status.cpu_percent > 90:
            health_issues.append("CPU使用率过高 (>90%)")
        elif status.cpu_percent > 80:
            warnings.append("CPU使用率较高 (>80%)")
            
        # 内存检查
        if status.memory_percent > 95:
            health_issues.append("内存使用率过高 (>95%)")
        elif status.memory_percent > 85:
            warnings.append("内存使用率较高 (>85%)")
            
        # 磁盘检查
        if status.disk_usage_percent > 95:
            health_issues.append("磁盘使用率过高 (>95%)")
        elif status.disk_usage_percent > 85:
            warnings.append("磁盘使用率较高 (>85%)")
            
        # GPU检查
        if status.cuda_available and status.gpu_memory_percent > 95:
            health_issues.append("GPU内存使用率过高 (>95%)")
        elif status.cuda_available and status.gpu_memory_percent > 85:
            warnings.append("GPU内存使用率较高 (>85%)")
            
        # 确定整体健康状态
        if health_issues:
            overall_status = "critical"
        elif warnings:
            overall_status = "warning"
        else:
            overall_status = "healthy"
            
        return {
            "overall_status": overall_status,
            "health_issues": health_issues,
            "warnings": warnings,
            "recommendations": self._generate_recommendations(status, health_issues, warnings)
        }
    
    def _generate_recommendations(self, status: SystemStatus, 
                                health_issues: List[str], 
                                warnings: List[str]) -> List[str]:
        """生成系统建议"""
        recommendations = []
        
        if status.cpu_percent > 80:
            recommendations.append("考虑减少并行任务数量或优化算法效率")
            
        if status.memory_percent > 85:
            recommendations.append("考虑清理内存缓存或增加系统内存")
            
        if status.disk_usage_percent > 85:
            recommendations.append("考虑清理磁盘空间或扩展存储容量")
            
        if status.cuda_available and status.gpu_memory_percent > 85:
            recommendations.append("考虑清理GPU缓存或减少GPU内存使用")
            
        if not status.cuda_available:
            recommendations.append("考虑启用CUDA以提升计算性能")
            
        if not recommendations:
            recommendations.append("系统运行状态良好，无需特殊操作")
            
        return recommendations
    
    async def check_python_environment(self) -> Dict[str, Any]:
        """检查Python环境"""
        try:
            self.logger.info("检查Python环境...")
            
            # 检查Python版本
            python_version = sys.version_info
            version_ok = python_version.major == 3 and python_version.minor >= 8
            
            # 检查必要的包
            required_packages = {
                'torch': torch.__version__,
                'numpy': 'numpy',  # 需要导入检查
                'asyncio': 'asyncio',
                'psutil': 'psutil'
            }
            
            missing_packages = []
            package_versions = {}
            
            for package, version in required_packages.items():
                try:
                    if package == 'torch':
                        package_versions[package] = version
                    else:
                        module = __import__(package)
                        package_versions[package] = getattr(module, '__version__', 'unknown')
                except ImportError:
                    missing_packages.append(package)
            
            # 检查CUDA可用性
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else None
            
            return {
                "status": "success",
                "python_version": {
                    "version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                    "ok": version_ok
                },
                "packages": {
                    "available": package_versions,
                    "missing": missing_packages
                },
                "cuda": {
                    "available": cuda_available,
                    "version": cuda_version
                },
                "environment_ok": version_ok and len(missing_packages) == 0
            }
            
        except Exception as e:
            self.logger.error(f"Python环境检查失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def check_core_components(self) -> Dict[str, Any]:
        """检查核心组件"""
        try:
            self.logger.info("检查核心组件...")
            
            components = {}
            
            # 检查模型模块
            try:
                from models.modular_net import ModularMathReasoningNet
                components['models'] = {"status": "available", "message": "模型模块正常"}
            except ImportError as e:
                components['models'] = {"status": "error", "message": f"模型模块导入失败: {e}"}
            
            # 检查进化模块
            try:
                from evolution.population import create_initial_population
                from evolution.nsga2 import evolve_population_nsga2_simple
                components['evolution'] = {"status": "available", "message": "进化模块正常"}
            except ImportError as e:
                components['evolution'] = {"status": "error", "message": f"进化模块导入失败: {e}"}
            
            # 检查评估器模块
            try:
                from evaluators.symbolic_evaluator import SymbolicEvaluator
                from evaluators.realworld_evaluator import RealWorldEvaluator
                components['evaluators'] = {"status": "available", "message": "评估器模块正常"}
            except ImportError as e:
                components['evaluators'] = {"status": "error", "message": f"评估器模块导入失败: {e}"}
            
            # 检查数据模块
            try:
                from data.generator import RealWorldDataGenerator
                components['data'] = {"status": "available", "message": "数据模块正常"}
            except ImportError as e:
                components['data'] = {"status": "error", "message": f"数据模块导入失败: {e}"}
            
            # 检查配置模块
            try:
                from config.logging_setup import setup_logging
                components['config'] = {"status": "available", "message": "配置模块正常"}
            except ImportError as e:
                components['config'] = {"status": "error", "message": f"配置模块导入失败: {e}"}
            
            # 统计组件状态
            total_components = len(components)
            available_components = sum(1 for c in components.values() if c["status"] == "available")
            error_components = total_components - available_components
            
            return {
                "status": "success",
                "components": components,
                "summary": {
                    "total": total_components,
                    "available": available_components,
                    "errors": error_components,
                    "all_ok": error_components == 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"核心组件检查失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def check_performance_benchmarks(self) -> Dict[str, Any]:
        """检查性能基准"""
        try:
            self.logger.info("检查性能基准...")
            
            benchmarks = {}
            
            # 模型创建性能
            try:
                from models.modular_net import ModularMathReasoningNet
                start_time = time.time()
                model = ModularMathReasoningNet(modules_config=[])
                creation_time = time.time() - start_time
                benchmarks['model_creation'] = {
                    "time": creation_time,
                    "status": "fast" if creation_time < 0.1 else "normal" if creation_time < 0.5 else "slow"
                }
            except Exception as e:
                benchmarks['model_creation'] = {"error": str(e)}
            
            # 种群创建性能
            try:
                from evolution.population import create_initial_population
                start_time = time.time()
                population = create_initial_population(10)
                creation_time = time.time() - start_time
                benchmarks['population_creation'] = {
                    "time": creation_time,
                    "status": "fast" if creation_time < 0.5 else "normal" if creation_time < 2.0 else "slow"
                }
            except Exception as e:
                benchmarks['population_creation'] = {"error": str(e)}
            
            # 评估性能
            try:
                from evaluators.symbolic_evaluator import SymbolicEvaluator
                from models.modular_net import ModularMathReasoningNet
                
                model = ModularMathReasoningNet(modules_config=[])
                evaluator = SymbolicEvaluator()
                
                start_time = time.time()
                score = await evaluator.evaluate(model, level=0)
                evaluation_time = time.time() - start_time
                
                benchmarks['evaluation'] = {
                    "time": evaluation_time,
                    "score": score,
                    "status": "fast" if evaluation_time < 1.0 else "normal" if evaluation_time < 5.0 else "slow"
                }
            except Exception as e:
                benchmarks['evaluation'] = {"error": str(e)}
            
            # 进化性能
            try:
                from evolution.nsga2 import evolve_population_nsga2_simple
                from evolution.population import create_initial_population
                
                population = create_initial_population(5)
                fitness_scores = [(0.8, 0.7)] * len(population)
                
                start_time = time.time()
                evolved_population = evolve_population_nsga2_simple(
                    population, fitness_scores, 0.8, 0.8
                )
                evolution_time = time.time() - start_time
                
                benchmarks['evolution'] = {
                    "time": evolution_time,
                    "status": "fast" if evolution_time < 0.5 else "normal" if evolution_time < 2.0 else "slow"
                }
            except Exception as e:
                benchmarks['evolution'] = {"error": str(e)}
            
            return {
                "status": "success",
                "benchmarks": benchmarks
            }
            
        except Exception as e:
            self.logger.error(f"性能基准检查失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def generate_status_report(self) -> str:
        """生成状态报告"""
        self.logger.info("生成系统状态报告...")
        
        # 收集所有检查结果
        resources_result = await self.check_system_resources()
        environment_result = await self.check_python_environment()
        components_result = await self.check_core_components()
        benchmarks_result = await self.check_performance_benchmarks()
        
        # 生成报告
        report = f"""
📊 系统状态报告
================

⏰ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🖥️  系统资源状态:
"""
        
        if resources_result["status"] == "success":
            status = resources_result["data"]
            health = resources_result["health"]
            
            report += f"""
   CPU使用率: {status.cpu_percent:.1f}%
   内存使用率: {status.memory_percent:.1f}% ({status.memory_used_mb:.1f} MB)
   可用内存: {status.memory_available_mb:.1f} MB
   磁盘使用率: {status.disk_usage_percent:.1f}%
   GPU内存使用率: {status.gpu_memory_percent:.1f}% ({status.gpu_memory_used_mb:.1f} MB)
   网络连接数: {status.network_connections}
   
   健康状态: {health['overall_status'].upper()}
"""
            
            if health['health_issues']:
                report += "   ⚠️  健康问题:\n"
                for issue in health['health_issues']:
                    report += f"      - {issue}\n"
                    
            if health['warnings']:
                report += "   ⚠️  警告:\n"
                for warning in health['warnings']:
                    report += f"      - {warning}\n"
                    
            if health['recommendations']:
                report += "   💡 建议:\n"
                for rec in health['recommendations']:
                    report += f"      - {rec}\n"
        else:
            report += f"   ❌ 检查失败: {resources_result['message']}\n"
        
        report += f"""
🐍 Python环境状态:
"""
        
        if environment_result["status"] == "success":
            env = environment_result
            report += f"""
   Python版本: {env['python_version']['version']} {'✅' if env['python_version']['ok'] else '❌'}
   CUDA可用: {'✅' if env['cuda']['available'] else '❌'}
   CUDA版本: {env['cuda']['version'] or 'N/A'}
   
   已安装包:
"""
            for package, version in env['packages']['available'].items():
                report += f"      - {package}: {version}\n"
                
            if env['packages']['missing']:
                report += "   缺失包:\n"
                for package in env['packages']['missing']:
                    report += f"      - {package}\n"
        else:
            report += f"   ❌ 检查失败: {environment_result['message']}\n"
        
        report += f"""
🔧 核心组件状态:
"""
        
        if components_result["status"] == "success":
            comps = components_result["components"]
            summary = components_result["summary"]
            
            for name, comp in comps.items():
                status_icon = "✅" if comp["status"] == "available" else "❌"
                report += f"   {status_icon} {name}: {comp['message']}\n"
                
            report += f"\n   组件统计: {summary['available']}/{summary['total']} 正常\n"
        else:
            report += f"   ❌ 检查失败: {components_result['message']}\n"
        
        report += f"""
⚡ 性能基准:
"""
        
        if benchmarks_result["status"] == "success":
            benchs = benchmarks_result["benchmarks"]
            
            for name, bench in benchs.items():
                if "error" in bench:
                    report += f"   ❌ {name}: {bench['error']}\n"
                else:
                    status_icon = "🟢" if bench["status"] == "fast" else "🟡" if bench["status"] == "normal" else "🔴"
                    report += f"   {status_icon} {name}: {bench['time']:.3f}秒 ({bench['status']})\n"
        else:
            report += f"   ❌ 检查失败: {benchmarks_result['message']}\n"
        
        report += f"""
📈 系统评级:
"""
        
        # 综合评级
        overall_score = 0
        total_checks = 0
        
        if resources_result["status"] == "success":
            health = resources_result["health"]
            if health["overall_status"] == "healthy":
                overall_score += 25
            elif health["overall_status"] == "warning":
                overall_score += 15
            total_checks += 1
            
        if environment_result["status"] == "success":
            if environment_result["environment_ok"]:
                overall_score += 25
            total_checks += 1
            
        if components_result["status"] == "success":
            if components_result["summary"]["all_ok"]:
                overall_score += 25
            total_checks += 1
            
        if benchmarks_result["status"] == "success":
            fast_benchmarks = sum(1 for b in benchmarks_result["benchmarks"].values() 
                                if "status" in b and b["status"] == "fast")
            if fast_benchmarks >= 2:
                overall_score += 25
            total_checks += 1
        
        if total_checks > 0:
            final_score = overall_score / total_checks
            if final_score >= 80:
                grade = "🟢 优秀"
            elif final_score >= 60:
                grade = "🟡 良好"
            elif final_score >= 40:
                grade = "🟠 一般"
            else:
                grade = "🔴 需要改进"
                
            report += f"   综合评分: {final_score:.1f}/100 {grade}\n"
        
        report += "\n🎉 系统状态检查完成！"
        
        return report

async def main():
    """主函数"""
    checker = SystemStatusChecker()
    report = await checker.generate_status_report()
    print(report)

if __name__ == "__main__":
    asyncio.run(main()) 