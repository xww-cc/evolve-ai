#!/usr/bin/env python3
"""
可视化数据优化分析脚本
分析当前图表数据的问题并提出优化方案
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
import glob

class VisualizationOptimizationAnalyzer:
    """可视化数据优化分析器"""
    
    def __init__(self):
        self.plots_dir = Path("evolution_plots")
        self.analysis_results = {}
        
    def analyze_current_visualization_data(self):
        """分析当前可视化数据的问题"""
        print("🔍 可视化数据优化分析")
        print("=" * 50)
        
        # 1. 分析文件数量和大小
        self._analyze_file_statistics()
        
        # 2. 分析数据质量问题
        self._analyze_data_quality()
        
        # 3. 分析存储效率
        self._analyze_storage_efficiency()
        
        # 4. 分析生成频率
        self._analyze_generation_frequency()
        
        # 5. 提出优化方案
        self._propose_optimizations()
        
        return self.analysis_results
    
    def _analyze_file_statistics(self):
        """分析文件统计信息"""
        print("\n📊 文件统计分析:")
        
        # 统计文件类型和大小
        file_stats = {
            'png_files': [],
            'json_files': [],
            'total_size': 0
        }
        
        if self.plots_dir.exists():
            for file_path in self.plots_dir.glob("*"):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    file_stats['total_size'] += file_size
                    
                    if file_path.suffix == '.png':
                        file_stats['png_files'].append({
                            'name': file_path.name,
                            'size': file_size,
                            'size_mb': file_size / (1024 * 1024)
                        })
                    elif file_path.suffix == '.json':
                        file_stats['json_files'].append({
                            'name': file_path.name,
                            'size': file_size,
                            'size_mb': file_size / (1024 * 1024)
                        })
        
        # 计算统计信息
        png_count = len(file_stats['png_files'])
        json_count = len(file_stats['json_files'])
        total_size_mb = file_stats['total_size'] / (1024 * 1024)
        
        print(f"   PNG文件数量: {png_count}")
        print(f"   JSON文件数量: {json_count}")
        print(f"   总文件大小: {total_size_mb:.2f} MB")
        
        if png_count > 0:
            avg_png_size = sum(f['size_mb'] for f in file_stats['png_files']) / png_count
            print(f"   平均PNG文件大小: {avg_png_size:.2f} MB")
        
        if json_count > 0:
            avg_json_size = sum(f['size_mb'] for f in file_stats['json_files']) / json_count
            print(f"   平均JSON文件大小: {avg_json_size:.2f} MB")
        
        self.analysis_results['file_statistics'] = {
            'png_count': png_count,
            'json_count': json_count,
            'total_size_mb': total_size_mb,
            'avg_png_size_mb': avg_png_size if png_count > 0 else 0,
            'avg_json_size_mb': avg_json_size if json_count > 0 else 0
        }
    
    def _analyze_data_quality(self):
        """分析数据质量问题"""
        print("\n🔍 数据质量分析:")
        
        quality_issues = []
        
        # 检查JSON文件中的数据质量
        json_files = list(self.plots_dir.glob("*.json"))
        if json_files:
            sample_file = json_files[0]
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查NaN值
                nan_count = 0
                total_values = 0
                
                def count_nans(obj):
                    nonlocal nan_count, total_values
                    if isinstance(obj, dict):
                        for value in obj.values():
                            count_nans(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            count_nans(item)
                    elif isinstance(obj, (int, float)):
                        total_values += 1
                        if np.isnan(obj):
                            nan_count += 1
                
                count_nans(data)
                
                if total_values > 0:
                    nan_percentage = (nan_count / total_values) * 100
                    print(f"   NaN值比例: {nan_percentage:.1f}% ({nan_count}/{total_values})")
                    
                    if nan_percentage > 5:
                        quality_issues.append(f"NaN值过多: {nan_percentage:.1f}%")
                
                # 检查数据重复
                if 'evolution_history' in data:
                    history = data['evolution_history']
                    if len(history) > 1:
                        # 检查连续代数的数据是否重复
                        duplicate_count = 0
                        for i in range(1, len(history)):
                            if (history[i]['best_fitness'] == history[i-1]['best_fitness'] and
                                history[i]['avg_fitness'] == history[i-1]['avg_fitness']):
                                duplicate_count += 1
                        
                        duplicate_percentage = (duplicate_count / (len(history) - 1)) * 100
                        print(f"   数据重复比例: {duplicate_percentage:.1f}%")
                        
                        if duplicate_percentage > 20:
                            quality_issues.append(f"数据重复过多: {duplicate_percentage:.1f}%")
                
                # 检查数据完整性
                required_fields = ['evolution_history', 'diversity_history', 'fitness_history']
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    quality_issues.append(f"缺少必要字段: {missing_fields}")
                
            except Exception as e:
                quality_issues.append(f"JSON解析错误: {e}")
        
        if quality_issues:
            print("   ❌ 发现数据质量问题:")
            for issue in quality_issues:
                print(f"     - {issue}")
        else:
            print("   ✅ 数据质量良好")
        
        self.analysis_results['data_quality'] = {
            'issues': quality_issues,
            'nan_percentage': nan_percentage if 'nan_percentage' in locals() else 0,
            'duplicate_percentage': duplicate_percentage if 'duplicate_percentage' in locals() else 0
        }
    
    def _analyze_storage_efficiency(self):
        """分析存储效率"""
        print("\n💾 存储效率分析:")
        
        efficiency_issues = []
        
        # 检查文件大小是否过大
        png_files = list(self.plots_dir.glob("*.png"))
        if png_files:
            large_files = [f for f in png_files if f.stat().st_size > 200 * 1024]  # 200KB
            if large_files:
                efficiency_issues.append(f"大文件过多: {len(large_files)} 个文件超过200KB")
                print(f"   ⚠️  发现 {len(large_files)} 个大文件 (>200KB)")
        
        # 检查JSON文件压缩
        json_files = list(self.plots_dir.glob("*.json"))
        if json_files:
            total_json_size = sum(f.stat().st_size for f in json_files)
            print(f"   JSON文件总大小: {total_json_size / 1024:.1f} KB")
            
            # 检查是否可以压缩
            if total_json_size > 100 * 1024:  # 100KB
                efficiency_issues.append("JSON文件过大，建议压缩")
        
        # 检查文件命名规范
        files = list(self.plots_dir.glob("*"))
        timestamp_files = [f for f in files if '_20250725_' in f.name]
        if len(timestamp_files) > 50:
            efficiency_issues.append("文件数量过多，建议清理旧文件")
            print(f"   ⚠️  时间戳文件过多: {len(timestamp_files)} 个")
        
        if efficiency_issues:
            print("   ❌ 存储效率问题:")
            for issue in efficiency_issues:
                print(f"     - {issue}")
        else:
            print("   ✅ 存储效率良好")
        
        self.analysis_results['storage_efficiency'] = {
            'issues': efficiency_issues,
            'large_file_count': len(large_files) if 'large_files' in locals() else 0,
            'timestamp_file_count': len(timestamp_files) if 'timestamp_files' in locals() else 0
        }
    
    def _analyze_generation_frequency(self):
        """分析生成频率"""
        print("\n⏱️ 生成频率分析:")
        
        # 分析文件生成时间
        files = list(self.plots_dir.glob("*"))
        if files:
            # 按修改时间排序
            files.sort(key=lambda x: x.stat().st_mtime)
            
            # 计算生成间隔
            intervals = []
            for i in range(1, len(files)):
                interval = files[i].stat().st_mtime - files[i-1].stat().st_mtime
                intervals.append(interval)
            
            if intervals:
                avg_interval = np.mean(intervals)
                min_interval = np.min(intervals)
                max_interval = np.max(intervals)
                
                print(f"   平均生成间隔: {avg_interval:.1f} 秒")
                print(f"   最短生成间隔: {min_interval:.1f} 秒")
                print(f"   最长生成间隔: {max_interval:.1f} 秒")
                
                # 检查是否生成过于频繁
                if min_interval < 10:  # 10秒内
                    print("   ⚠️  生成过于频繁，可能影响性能")
                
                # 检查是否生成过于稀疏
                if avg_interval > 300:  # 5分钟
                    print("   ⚠️  生成间隔过长，可能丢失重要数据")
        
        self.analysis_results['generation_frequency'] = {
            'avg_interval': avg_interval if 'avg_interval' in locals() else 0,
            'min_interval': min_interval if 'min_interval' in locals() else 0,
            'max_interval': max_interval if 'max_interval' in locals() else 0
        }
    
    def _propose_optimizations(self):
        """提出优化方案"""
        print("\n🚀 优化方案建议:")
        
        optimizations = []
        
        # 1. 数据压缩优化
        optimizations.append({
            'category': '数据压缩',
            'description': '压缩JSON数据，减少存储空间',
            'implementation': '使用gzip压缩JSON文件',
            'expected_benefit': '减少50-70%存储空间'
        })
        
        # 2. 图片质量优化
        optimizations.append({
            'category': '图片质量',
            'description': '优化PNG图片质量和大小',
            'implementation': '调整DPI和压缩参数',
            'expected_benefit': '减少30-50%文件大小'
        })
        
        # 3. 数据清理优化
        optimizations.append({
            'category': '数据清理',
            'description': '清理NaN值和重复数据',
            'implementation': '在数据记录前进行验证和清理',
            'expected_benefit': '提高数据质量，减少存储'
        })
        
        # 4. 生成频率优化
        optimizations.append({
            'category': '生成频率',
            'description': '优化图表生成频率',
            'implementation': '实现智能生成策略',
            'expected_benefit': '减少不必要的文件生成'
        })
        
        # 5. 文件管理优化
        optimizations.append({
            'category': '文件管理',
            'description': '实现自动文件清理',
            'implementation': '保留最新的N个文件，自动删除旧文件',
            'expected_benefit': '控制存储空间使用'
        })
        
        # 6. 数据格式优化
        optimizations.append({
            'category': '数据格式',
            'description': '优化数据存储格式',
            'implementation': '使用更高效的二进制格式或数据库',
            'expected_benefit': '提高读写效率'
        })
        
        for i, opt in enumerate(optimizations, 1):
            print(f"   {i}. {opt['category']}: {opt['description']}")
            print(f"      实现: {opt['implementation']}")
            print(f"      预期收益: {opt['expected_benefit']}")
            print()
        
        self.analysis_results['optimizations'] = optimizations
    
    def generate_optimization_report(self):
        """生成优化报告"""
        report = {
            'analysis_timestamp': str(np.datetime64('now')),
            'summary': {
                'total_files': self.analysis_results.get('file_statistics', {}).get('png_count', 0) + 
                              self.analysis_results.get('file_statistics', {}).get('json_count', 0),
                'total_size_mb': self.analysis_results.get('file_statistics', {}).get('total_size_mb', 0),
                'quality_issues': len(self.analysis_results.get('data_quality', {}).get('issues', [])),
                'efficiency_issues': len(self.analysis_results.get('storage_efficiency', {}).get('issues', [])),
                'optimization_count': len(self.analysis_results.get('optimizations', []))
            },
            'detailed_analysis': self.analysis_results
        }
        
        # 保存报告
        report_file = "visualization_optimization_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 优化报告已保存: {report_file}")
        return report_file

def main():
    """主函数"""
    analyzer = VisualizationOptimizationAnalyzer()
    analyzer.analyze_current_visualization_data()
    analyzer.generate_optimization_report()
    
    print("\n✅ 可视化数据优化分析完成！")

if __name__ == "__main__":
    main() 