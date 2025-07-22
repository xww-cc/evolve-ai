# Evolve-AI 项目结构概览

## 📁 清理后的项目结构

```
evolve-ai/
├── 📄 README.md                    # 项目说明文档
├── 📄 requirements.txt             # Python依赖
├── 📄 LICENSE                      # 许可证文件
├── 📄 CONTRIBUTING.md              # 贡献指南
├── 📄 .gitignore                   # Git忽略文件
├── 📄 main.py                      # 主程序入口
│
├── 📁 config/                      # 配置文件
│   ├── __init__.py
│   ├── env_loader.py
│   ├── global_constants.py
│   ├── logging_setup.py
│
├── 📁 data/                        # 数据生成器
│   ├── __init__.py
│   ├── generator.py
│   ├── loader.py
│
├── 📁 evaluators/                  # 评估器
│   ├── __init__.py
│   ├── realworld_evaluator.py
│   ├── symbolic_evaluator.py
│
├── 📁 evolution/                   # 进化算法
│   ├── __init__.py
│   ├── nsga2.py
│   ├── population.py
│   ├── stagnation_detector.py
│
├── 📁 integrations/                # 外部集成
│   ├── __init__.py
│   ├── external_apis.py
│   ├── xai_integration.py
│
├── 📁 models/                      # AI模型
│   ├── __init__.py
│   ├── base_module.py
│   ├── epigenetic.py
│   ├── modular_net.py
│
├── 📁 optimizers/                  # 优化器
│   ├── __init__.py
│   ├── autophagy.py
│   ├── finetune.py
│   ├── mutation.py
│
├── 📁 plugins/                     # 插件系统
│   ├── __init__.py
│   ├── metrics/
│   ├── modules/
│   ├── tasks/
│
├── 📁 utils/                       # 工具函数
│   ├── __init__.py
│   ├── error_handler.py
│   ├── parallel_utils.py
│   ├── performance_monitor.py
│   ├── visualization.py
│
├── 📁 tests/                       # 测试套件
│   ├── conftest.py
│   ├── functional/
│   ├── integration/
│   ├── performance/
│   ├── unit/
│
├── 📁 reports/                     # 📊 报告文件 (整理后)
│   ├── AI_evolution_final_report.md
│   ├── AI_evolution_validation_report.md
│   ├── comprehensive_test_report.md
│   ├── final_evaluation_summary.md
│   ├── model_evaluation_report.html
│   ├── model_evaluation_report.json
│   ├── test_report.json
│   ├── TEST_STATUS.md
│   ├── GIT_STATUS.md
│
├── 📁 test_files/                  # 🧪 测试文件 (整理后)
│   ├── quick_evolution_validation.py
│   ├── comprehensive_evolution_test.py
│   ├── model_evaluation_report.py
│   ├── test_evolution_simple.py
│   ├── quick_test.py
│   ├── generate_html_report.py
│   ├── system_status.py
│   ├── system_optimizer.py
│   ├── performance_monitor.py
│   ├── system_test.py
│
├── 📁 temp_files/                  # 🗂️ 临时文件 (整理后)
│   ├── .coverage
│   ├── run_tests.py
│
├── 📁 backup/                      # 💾 备用目录
│
└── 📁 logs/                        # 📝 日志目录
```

## 🎯 整理效果

### ✅ 清理前的问题
- 根目录被大量测试文件占满
- 报告文件散落在各处
- 临时文件影响项目整洁度
- 目录结构不清晰

### ✅ 清理后的改进
- **根目录整洁**: 只保留核心项目文件
- **分类整理**: 按功能分类存放文件
- **结构清晰**: 易于理解和维护
- **便于管理**: 相关文件集中存放

## 📊 文件统计

| 目录 | 文件数量 | 主要用途 |
|------|----------|----------|
| `reports/` | 9个 | 存放所有验证报告和状态文件 |
| `test_files/` | 10个 | 存放测试相关脚本 |
| `temp_files/` | 2个 | 存放临时文件 |
| `backup/` | 0个 | 备用目录 |
| 根目录 | 6个 | 核心项目文件 |

## 🚀 使用建议

1. **开发时**: 主要关注 `config/`, `models/`, `evolution/`, `evaluators/` 等核心目录
2. **测试时**: 使用 `tests/` 目录中的测试套件
3. **查看报告**: 查看 `reports/` 目录中的验证报告
4. **临时文件**: 将临时文件放在 `temp_files/` 目录

## 🎉 项目状态

- **目录整洁度**: ✅ 优秀
- **文件组织**: ✅ 清晰
- **维护便利性**: ✅ 高
- **项目可读性**: ✅ 优秀

---

*项目结构已优化完成，现在更加整洁和易于维护！* 