# Evolve AI - 进化人工智能系统

一个基于进化算法的模块化神经网络系统，专门用于数学推理能力的进化。

## 🚀 系统特性

### 核心功能
- **模块化神经网络**: 支持动态模块组合的神经网络架构
- **NSGA-II进化算法**: 多目标优化的进化算法
- **双评估系统**: 符号推理 + 真实世界评估
- **自适应多样性**: 智能多样性维护和停滞检测
- **7级别进化**: 从基础运算到AGI数学融合的渐进式进化

### 性能表现
- ⚡ **评估速度**: 265.8 个体/秒
- 🔄 **进化速度**: 183.4 代/秒
- 🟢 **系统状态**: 优秀
- 💾 **内存效率**: 优化的缓存和内存管理

## 📁 项目结构

```
evolve-ai/
├── main.py                 # 主程序 - 完整进化流程
├── system_test.py          # 系统框架测试
├── performance_monitor.py  # 性能监控
├── system_optimizer.py     # 系统优化
├── system_status.py        # 系统状态检查
├── config/                 # 配置模块
│   ├── global_constants.py # 全局常量
│   ├── logging_setup.py   # 日志配置
│   └── env_loader.py      # 环境加载
├── evolution/              # 进化算法
│   ├── nsga2.py           # NSGA-II算法
│   ├── population.py      # 种群管理
│   └── stagnation_detector.py # 停滞检测
├── evaluators/            # 评估器
│   ├── symbolic_evaluator.py    # 符号推理评估
│   └── realworld_evaluator.py   # 真实世界评估
├── models/                # 神经网络模型
│   ├── modular_net.py    # 模块化网络
│   ├── epigenetic.py     # 表观遗传标记
│   └── base_module.py    # 基础模块
├── optimizers/           # 优化器
│   ├── mutation.py       # 变异操作
│   ├── autophagy.py      # 细胞自噬
│   └── finetune.py       # 微调
├── data/                 # 数据模块
│   ├── generator.py      # 数据生成器
│   └── loader.py         # 数据加载器
├── utils/                # 工具模块
│   ├── visualization.py  # 可视化
│   ├── parallel_utils.py # 并行工具
│   └── error_handler.py  # 错误处理
└── integrations/         # 集成模块
    ├── external_apis.py  # 外部API
    └── xai_integration.py # 可解释AI
```

## 🎯 使用方法

### 快速开始

1. **系统测试**
```bash
python3 system_test.py
```

2. **完整进化**
```bash
python3 main.py
```

3. **性能监控**
```bash
python3 performance_monitor.py
```

4. **系统状态检查**
```bash
python3 system_status.py
```

### 配置参数

在 `config/global_constants.py` 中可以调整：
- `POPULATION_SIZE`: 种群大小 (默认: 20)
- `NUM_GENERATIONS`: 每级别代数 (默认: 50)
- `BASE_MUTATION_RATE_STRUCTURE`: 基础变异率 (默认: 0.15)
- `EPIGENETIC_MUTATION_RATE`: 表观遗传变异率 (默认: 0.1)

## 🔬 进化级别

系统包含7个渐进式进化级别：

1. **基础运算** - 加减乘除
2. **乘法运算** - 复杂乘法
3. **指数对数** - 指数和对数运算
4. **微积分** - 微分和积分
5. **线性代数** - 矩阵运算
6. **三角函数** - 三角函数
7. **AGI数学融合** - 综合数学推理

## 📊 性能指标

### 当前系统性能
- **CPU**: 8核心, 16.3%使用率
- **内存**: 17.2GB, 81.2%使用率
- **评估速度**: 265.8 个体/秒
- **进化速度**: 183.4 代/秒
- **缓存效率**: 99.9%性能提升

### 进化结果示例
- **符号推理得分**: 0.8976 (平均)
- **真实世界得分**: 2.0000 (最佳)
- **种群多样性**: 0.75-0.95 (自适应)

## 🛠️ 技术架构

### 核心算法
- **NSGA-II**: 非支配排序遗传算法
- **模块化网络**: 动态模块组合
- **表观遗传**: 可遗传的调节标记
- **细胞自噬**: 网络结构优化

### 评估系统
- **符号评估**: 数学推理能力测试
- **真实世界评估**: 实际应用能力测试
- **缓存优化**: 智能缓存管理
- **并行评估**: 异步评估支持

## 🎨 可视化

系统自动生成进化曲线图：
- 平均性能曲线
- 最佳性能曲线
- 多样性变化
- 停滞检测

## 🔧 开发工具

### 测试脚本
- `system_test.py`: 核心功能测试
- `performance_monitor.py`: 性能监控
- `system_optimizer.py`: 系统优化
- `system_status.py`: 状态检查

### 日志系统
- 详细的运行日志
- 性能指标记录
- 错误追踪
- 进化过程记录

## 🚀 部署要求

### 系统要求
- Python 3.9+
- PyTorch 2.7+
- NumPy 2.0+
- 8GB+ 内存
- 多核CPU (推荐8核+)

### 安装依赖
```bash
pip install -r requirements.txt
```

## 📈 未来扩展

### 计划功能
- GPU加速支持
- 分布式进化
- 更多数学领域
- 实时可视化
- API接口

### 研究方向
- 大规模种群进化
- 多目标优化改进
- 自适应参数调整
- 跨领域知识迁移

## 🤝 贡献指南

欢迎贡献代码和想法！
- 提交Issue报告问题
- 提交Pull Request贡献代码
- 参与讨论和设计

## 📄 许可证

MIT License - 详见 LICENSE 文件

---

**Evolve AI** - 让AI通过进化变得更智能 🧠✨