#!/usr/bin/env python3
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
    
    print(f"\n🗑️  准备删除 {len(files_to_delete)} 个文件...")
    
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
    
    print(f"\n✅ 安全清理完成!")
    print(f"   删除文件数: {deleted_count}")
    print(f"   释放空间: {total_size / (1024 * 1024):.2f} MB")
    print(f"\n✅ 保留的重要文件:")
    print(f"   - 所有测试文件 (test_files/)")
    print(f"   - 核心模块文件")
    print(f"   - 配置文件")
    print(f"   - 文档文件")

if __name__ == "__main__":
    safe_cleanup_project()
