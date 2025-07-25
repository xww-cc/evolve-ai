#!/usr/bin/env python3
"""
安全项目清理分析脚本 - 只清除真正可以删除的文件
"""
import os
import json
from pathlib import Path

def analyze_safe_cleanup():
    """分析可以安全删除的文件"""
    
    print("🧹 安全项目清理分析")
    print("=" * 50)
    
    # 定义可以安全删除的文件类型
    safe_to_delete = {
        "logs": [],           # 日志文件
        "plots": [],          # 图表文件
        "json_reports": [],   # JSON报告文件
        "temp_files": []      # 临时文件
    }
    
    # 1. 分析日志文件
    logs_dir = Path("logs")
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log"):
            safe_to_delete["logs"].append(str(log_file))
    
    # 2. 分析图表文件
    plots_dir = Path("evolution_plots")
    if plots_dir.exists():
        for plot_file in plots_dir.glob("*"):
            if plot_file.is_file():
                safe_to_delete["plots"].append(str(plot_file))
    
    # 3. 分析JSON报告文件
    for json_file in Path(".").glob("*report*.json"):
        safe_to_delete["json_reports"].append(str(json_file))
    
    # 4. 分析临时文件
    for temp_file in Path(".").glob("*.tmp"):
        safe_to_delete["temp_files"].append(str(temp_file))
    for temp_file in Path(".").glob("*.temp"):
        safe_to_delete["temp_files"].append(str(temp_file))
    
    # 计算空间节省
    total_size = 0
    for category, files in safe_to_delete.items():
        for file_path in files:
            try:
                file_size = os.path.getsize(file_path)
                total_size += file_size
            except OSError:
                continue
    
    # 生成报告
    report = {
        "summary": {
            "total_removable_files": sum(len(files) for files in safe_to_delete.values()),
            "space_saving_mb": round(total_size / (1024 * 1024), 2),
            "categories": {k: len(v) for k, v in safe_to_delete.items()}
        },
        "files_to_delete": safe_to_delete
    }
    
    # 输出分析结果
    print(f"\n📊 安全清理分析结果:")
    print(f"   可删除文件总数: {report['summary']['total_removable_files']}")
    print(f"   可节省空间: {report['summary']['space_saving_mb']} MB")
    
    print(f"\n🗂️ 文件分类:")
    for category, count in report['summary']['categories'].items():
        print(f"   {category}: {count} 个文件")
    
    print(f"\n🗑️ 可删除文件详情:")
    
    # 显示日志文件
    if safe_to_delete["logs"]:
        print(f"   日志文件 ({len(safe_to_delete['logs'])} 个):")
        for log_file in safe_to_delete["logs"][:3]:
            print(f"     - {log_file}")
        if len(safe_to_delete["logs"]) > 3:
            print(f"     ... 还有 {len(safe_to_delete['logs']) - 3} 个日志文件")
    
    # 显示图表文件
    if safe_to_delete["plots"]:
        print(f"   图表文件 ({len(safe_to_delete['plots'])} 个):")
        for plot_file in safe_to_delete["plots"][:3]:
            print(f"     - {plot_file}")
        if len(safe_to_delete["plots"]) > 3:
            print(f"     ... 还有 {len(safe_to_delete['plots']) - 3} 个图表文件")
    
    # 显示JSON报告文件
    if safe_to_delete["json_reports"]:
        print(f"   JSON报告文件 ({len(safe_to_delete['json_reports'])} 个):")
        for json_file in safe_to_delete["json_reports"]:
            print(f"     - {json_file}")
    
    # 显示临时文件
    if safe_to_delete["temp_files"]:
        print(f"   临时文件 ({len(safe_to_delete['temp_files'])} 个):")
        for temp_file in safe_to_delete["temp_files"]:
            print(f"     - {temp_file}")
    
    print(f"\n✅ 保留的重要文件:")
    print(f"   - 所有测试文件 (test_files/)")
    print(f"   - 核心模块文件 (models/, evolution/, evaluators/, etc.)")
    print(f"   - 配置文件 (config/)")
    print(f"   - 文档文件 (*.md)")
    print(f"   - 主程序文件 (main.py, system_*.py)")
    
    # 保存详细报告
    with open("safe_cleanup_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细报告已保存到: safe_cleanup_report.json")
    
    # 生成安全清理脚本
    generate_safe_cleanup_script(report)
    
    return report

def generate_safe_cleanup_script(report):
    """生成安全清理脚本"""
    
    script_content = '''#!/usr/bin/env python3
"""
安全项目清理脚本 - 只删除真正可以删除的文件
"""
import os
import shutil
from pathlib import Path

def safe_cleanup_project():
    """安全清理项目文件"""
    
    print("🧹 开始安全清理项目文件...")
    print("⚠️  只删除日志、图表、报告等临时文件")
    print("✅ 保留所有测试文件和核心模块")
    
    # 要删除的文件列表
    files_to_delete = []
    
    # 1. 删除日志文件
    logs_dir = Path("logs")
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log"):
            files_to_delete.append(log_file)
    
    # 2. 删除图表文件
    plots_dir = Path("evolution_plots")
    if plots_dir.exists():
        for plot_file in plots_dir.glob("*"):
            if plot_file.is_file():
                files_to_delete.append(plot_file)
    
    # 3. 删除JSON报告文件
    for json_file in Path(".").glob("*report*.json"):
        files_to_delete.append(json_file)
    
    # 4. 删除临时文件
    for temp_file in Path(".").glob("*.tmp"):
        files_to_delete.append(temp_file)
    for temp_file in Path(".").glob("*.temp"):
        files_to_delete.append(temp_file)
    
    # 执行删除
    deleted_count = 0
    total_size = 0
    
    print(f"\\n🗑️  准备删除 {len(files_to_delete)} 个文件...")
    
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
    for dir_path in [logs_dir, plots_dir]:
        if dir_path.exists() and not any(dir_path.iterdir()):
            try:
                dir_path.rmdir()
                print(f"   删除空目录: {dir_path}")
            except Exception as e:
                print(f"   删除目录失败 {dir_path}: {e}")
    
    print(f"\\n✅ 安全清理完成!")
    print(f"   删除文件数: {deleted_count}")
    print(f"   释放空间: {total_size / (1024 * 1024):.2f} MB")
    print(f"\\n✅ 保留的重要文件:")
    print(f"   - 所有测试文件 (test_files/)")
    print(f"   - 核心模块文件")
    print(f"   - 配置文件")
    print(f"   - 文档文件")

if __name__ == "__main__":
    safe_cleanup_project()
'''
    
    with open("safe_cleanup_project.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print(f"📝 安全清理脚本已生成: safe_cleanup_project.py")

def main():
    """主函数"""
    report = analyze_safe_cleanup()
    
    print(f"\n🎯 总结:")
    print(f"   本次分析只识别了真正可以安全删除的文件")
    print(f"   所有重要的测试文件和核心模块都会被保留")
    print(f"   可以安全释放 {report['summary']['space_saving_mb']} MB 空间")

if __name__ == "__main__":
    main() 