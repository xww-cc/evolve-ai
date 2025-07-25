#!/usr/bin/env python3
"""
项目清理分析脚本 - 识别可以安全删除的文件
"""
import os
import json
import glob
from pathlib import Path

def analyze_project_structure():
    """分析项目结构，识别可清理的文件"""
    
    # 项目根目录
    project_root = Path(".")
    
    # 定义文件分类
    file_categories = {
        "core_files": [],      # 核心文件，不能删除
        "test_files": [],      # 测试文件，可以清理
        "log_files": [],       # 日志文件，可以清理
        "plot_files": [],      # 图表文件，可以清理
        "temp_files": [],      # 临时文件，可以清理
        "documentation": [],   # 文档文件，保留重要文档
        "json_reports": []     # JSON报告文件，可以清理
    }
    
    # 核心文件列表（不能删除）
    core_files = {
        "main.py", "system_test.py", "system_status.py", "system_optimizer.py",
        "performance_monitor.py", "requirements.txt", "README.md", "LICENSE",
        ".gitignore", "CONTRIBUTING.md", "USAGE_GUIDE.md", "FRAMEWORK_ANALYSIS.md",
        "FUTURE_WORK_PLAN.md"
    }
    
    # 重要文档文件（保留）
    important_docs = {
        "MODEL_MANUAL.md", "TECHNICAL_SPECIFICATIONS.md", "CURRENT_STAGE_REPORT.md",
        "THEORETICAL_RESEARCH.md", "THEORETICAL_ANALYSIS.md", "OPTIMIZATION_ISSUES_ANALYSIS.md"
    }
    
    # 遍历项目文件
    for file_path in project_root.rglob("*"):
        if file_path.is_file() and not str(file_path).startswith(".git"):
            relative_path = str(file_path.relative_to(project_root))
            
            # 分类文件
            if relative_path in core_files:
                file_categories["core_files"].append(relative_path)
            elif relative_path in important_docs:
                file_categories["documentation"].append(relative_path)
            elif relative_path.startswith("test_files/"):
                file_categories["test_files"].append(relative_path)
            elif relative_path.startswith("logs/"):
                file_categories["log_files"].append(relative_path)
            elif relative_path.startswith("evolution_plots/"):
                file_categories["plot_files"].append(relative_path)
            elif relative_path.endswith(".json") and "report" in relative_path.lower():
                file_categories["json_reports"].append(relative_path)
            elif relative_path.endswith((".tmp", ".temp", ".cache")):
                file_categories["temp_files"].append(relative_path)
            else:
                # 检查是否是核心模块文件
                if any(relative_path.startswith(folder) for folder in [
                    "models/", "evolution/", "evaluators/", "config/", 
                    "utils/", "optimizers/", "data/", "integrations/", 
                    "plugins/", "tests/"
                ]):
                    file_categories["core_files"].append(relative_path)
                else:
                    file_categories["temp_files"].append(relative_path)
    
    return file_categories

def analyze_test_files():
    """分析测试文件，识别哪些可以删除"""
    
    test_files_dir = Path("test_files")
    if not test_files_dir.exists():
        return []
    
    # 保留的测试文件（核心功能测试）
    keep_test_files = {
        "quick_evolution_validation.py"  # 核心验证测试
    }
    
    # 可以删除的测试文件
    removable_test_files = []
    
    for test_file in test_files_dir.glob("*.py"):
        if test_file.name not in keep_test_files:
            removable_test_files.append(str(test_file))
    
    return removable_test_files

def analyze_log_files():
    """分析日志文件，识别哪些可以删除"""
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return []
    
    log_files = []
    for log_file in logs_dir.glob("*.log"):
        log_files.append(str(log_file))
    
    return log_files

def analyze_plot_files():
    """分析图表文件，识别哪些可以删除"""
    
    plots_dir = Path("evolution_plots")
    if not plots_dir.exists():
        return []
    
    plot_files = []
    for plot_file in plots_dir.glob("*"):
        if plot_file.is_file():
            plot_files.append(str(plot_file))
    
    return plot_files

def calculate_space_savings(file_categories):
    """计算清理后可以节省的空间"""
    
    total_size = 0
    removable_size = 0
    
    for category, files in file_categories.items():
        for file_path in files:
            try:
                file_size = os.path.getsize(file_path)
                total_size += file_size
                
                if category in ["test_files", "log_files", "plot_files", "temp_files", "json_reports"]:
                    removable_size += file_size
            except OSError:
                continue
    
    return total_size, removable_size

def generate_cleanup_report(file_categories, removable_test_files, log_files, plot_files):
    """生成清理报告"""
    
    total_size, removable_size = calculate_space_savings(file_categories)
    
    report = {
        "summary": {
            "total_files": sum(len(files) for files in file_categories.values()),
            "removable_files": len(removable_test_files) + len(log_files) + len(plot_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "removable_size_mb": round(removable_size / (1024 * 1024), 2),
            "space_saving_percent": round((removable_size / total_size) * 100, 1) if total_size > 0 else 0
        },
        "removable_files": {
            "test_files": removable_test_files,
            "log_files": log_files,
            "plot_files": plot_files
        },
        "file_categories": file_categories
    }
    
    return report

def main():
    """主函数"""
    print("🧹 项目清理分析")
    print("=" * 50)
    
    # 分析项目结构
    file_categories = analyze_project_structure()
    
    # 分析各类文件
    removable_test_files = analyze_test_files()
    log_files = analyze_log_files()
    plot_files = analyze_plot_files()
    
    # 生成报告
    report = generate_cleanup_report(file_categories, removable_test_files, log_files, plot_files)
    
    # 输出报告
    print(f"\n📊 清理分析结果:")
    print(f"   总文件数: {report['summary']['total_files']}")
    print(f"   可删除文件数: {report['summary']['removable_files']}")
    print(f"   总大小: {report['summary']['total_size_mb']} MB")
    print(f"   可节省空间: {report['summary']['removable_size_mb']} MB")
    print(f"   空间节省比例: {report['summary']['space_saving_percent']}%")
    
    print(f"\n🗂️ 文件分类统计:")
    for category, files in file_categories.items():
        print(f"   {category}: {len(files)} 个文件")
    
    print(f"\n🗑️ 可删除文件详情:")
    print(f"   测试文件 ({len(removable_test_files)} 个):")
    for test_file in removable_test_files[:5]:  # 只显示前5个
        print(f"     - {test_file}")
    if len(removable_test_files) > 5:
        print(f"     ... 还有 {len(removable_test_files) - 5} 个文件")
    
    print(f"   日志文件 ({len(log_files)} 个):")
    for log_file in log_files[:5]:
        print(f"     - {log_file}")
    if len(log_files) > 5:
        print(f"     ... 还有 {len(log_files) - 5} 个文件")
    
    print(f"   图表文件 ({len(plot_files)} 个):")
    for plot_file in plot_files[:5]:
        print(f"     - {plot_file}")
    if len(plot_files) > 5:
        print(f"     ... 还有 {len(plot_files) - 5} 个文件")
    
    # 保存详细报告
    with open("cleanup_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细报告已保存到: cleanup_report.json")
    
    # 生成清理脚本
    generate_cleanup_script(report)
    
    return report

def generate_cleanup_script(report):
    """生成清理脚本"""
    
    script_content = '''#!/usr/bin/env python3
"""
自动清理脚本 - 删除可清理的文件
"""
import os
import shutil
from pathlib import Path

def cleanup_project():
    """清理项目文件"""
    
    print("🧹 开始清理项目文件...")
    
    # 要删除的文件列表
    files_to_delete = []
    
    # 添加测试文件
    test_files_dir = Path("test_files")
    if test_files_dir.exists():
        keep_files = {"quick_evolution_validation.py"}
        for test_file in test_files_dir.glob("*.py"):
            if test_file.name not in keep_files:
                files_to_delete.append(test_file)
    
    # 添加日志文件
    logs_dir = Path("logs")
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log"):
            files_to_delete.append(log_file)
    
    # 添加图表文件
    plots_dir = Path("evolution_plots")
    if plots_dir.exists():
        for plot_file in plots_dir.glob("*"):
            if plot_file.is_file():
                files_to_delete.append(plot_file)
    
    # 添加JSON报告文件
    for json_file in Path(".").glob("*report*.json"):
        files_to_delete.append(json_file)
    
    # 执行删除
    deleted_count = 0
    total_size = 0
    
    for file_path in files_to_delete:
        try:
            if file_path.exists():
                file_size = file_path.stat().st_size
                file_path.unlink()
                deleted_count += 1
                total_size += file_size
                print(f"   删除: {file_path}")
        except Exception as e:
            print(f"   删除失败 {file_path}: {e}")
    
    # 清理空目录
    for dir_path in [test_files_dir, logs_dir, plots_dir]:
        if dir_path.exists() and not any(dir_path.iterdir()):
            try:
                dir_path.rmdir()
                print(f"   删除空目录: {dir_path}")
            except Exception as e:
                print(f"   删除目录失败 {dir_path}: {e}")
    
    print(f"\\n✅ 清理完成!")
    print(f"   删除文件数: {deleted_count}")
    print(f"   释放空间: {total_size / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    cleanup_project()
'''
    
    with open("cleanup_project.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print(f"📝 清理脚本已生成: cleanup_project.py")

if __name__ == "__main__":
    main() 