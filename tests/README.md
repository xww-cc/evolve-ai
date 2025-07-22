# Evolve AI 测试套件

## 📋 测试概览

本测试套件提供了全面的测试覆盖，确保Evolve AI系统的稳定性、性能和功能正确性。

## 🏗️ 测试结构

```
tests/
├── unit/                    # 单元测试
│   ├── test_models.py      # 模型测试
│   ├── test_evaluators.py  # 评估器测试
│   └── test_evolution.py   # 进化算法测试
├── integration/             # 集成测试
│   └── test_evolution_flow.py
├── performance/             # 性能测试
│   └── test_performance_benchmarks.py
├── functional/              # 功能测试
│   └── test_functional_capabilities.py
├── conftest.py             # pytest配置
└── README.md               # 本文档
```

## 🧪 测试类型

### 1. 单元测试 (`tests/unit/`)

**目标**: 验证单个模块的独立功能

**测试内容**:
- **模型测试**: `ModularMathReasoningNet` 初始化、前向传播、模块操作
- **评估器测试**: `SymbolicEvaluator` 和 `RealWorldEvaluator` 的基本评估功能
- **进化算法测试**: NSGA-II 算法、种群管理、停滞检测

**执行方式**:
```bash
python -m pytest tests/unit/ -v
```

### 2. 集成测试 (`tests/integration/`)

**目标**: 验证模块间的交互和完整流程

**测试内容**:
- 小规模进化流程（种群=10，世代=5）
- 双重评估集成
- 停滞检测集成
- 错误处理集成
- 性能集成

**执行方式**:
```bash
python -m pytest tests/integration/ -v
```

### 3. 性能测试 (`tests/performance/`)

**目标**: 评估系统性能和资源使用

**测试内容**:
- 种群创建性能
- 评估吞吐量
- 进化性能
- 资源消耗监控
- 扩展性测试
- 内存泄漏检测

**执行方式**:
```bash
python -m pytest tests/performance/ -v
```

### 4. 功能测试 (`tests/functional/`)

**目标**: 验证系统的整体能力和7个进化级别

**测试内容**:
- 基本运算级别（1-3）
- 高级数学级别（4-6）
- AGI数学融合级别（7）
- 进化进展
- 可视化生成
- 系统状态检查
- 边界错误处理

**执行方式**:
```bash
python -m pytest tests/functional/ -v
```

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行完整测试套件

```bash
python run_tests.py
```

### 运行特定测试

```bash
# 单元测试
python -m pytest tests/unit/ -v

# 集成测试
python -m pytest tests/integration/ -v

# 性能测试
python -m pytest tests/performance/ -v

# 功能测试
python -m pytest tests/functional/ -v
```

### 运行覆盖率测试

```bash
coverage run -m pytest tests/unit/ tests/integration/
coverage report
coverage html  # 生成HTML报告
```

## 📊 性能基准

### 目标指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 种群创建速度 | > 1000 个体/秒 | 种群初始化性能 |
| 评估吞吐量 | > 200 个体/秒 | 评估器性能 |
| 进化速度 | > 100 代/秒 | 进化算法性能 |
| 系统稳定性 | > 95% 成功率 | 系统可靠性 |
| 内存使用 | < 4GB 峰值 | 资源消耗 |
| CPU使用 | < 80% | 处理器使用率 |

### 测试环境要求

- **Python版本**: 3.9+
- **内存**: 至少8GB RAM
- **存储**: 至少1GB可用空间
- **网络**: 稳定的互联网连接（用于依赖下载）

## 🔧 测试配置

### pytest配置 (`conftest.py`)

- 自动检测异步测试
- 设置事件循环
- 配置测试路径

### 测试夹具

```python
@pytest.fixture
def evaluators():
    """创建评估器实例"""
    return {
        'symbolic': SymbolicEvaluator(),
        'realworld': RealWorldEvaluator()
    }

@pytest.fixture
def test_population():
    """创建测试种群"""
    return create_initial_population(15)
```

## 📈 测试报告

### 自动生成的报告

- **JSON报告**: `test_report.json`
- **覆盖率报告**: `htmlcov/index.html`
- **性能报告**: 控制台输出

### 报告内容

```json
{
  "test_info": {
    "total_tests": 8,
    "passed_tests": 7,
    "failed_tests": 1,
    "success_rate": 0.875,
    "test_date": "2025-01-23T10:30:00",
    "total_duration": 45.2
  },
  "results": {
    "unit_tests": {...},
    "integration_tests": {...},
    "performance_tests": {...}
  },
  "summary": {
    "status": "PASS",
    "message": "测试完成: 7/8 通过"
  }
}
```

## 🐛 故障排除

### 常见问题

1. **导入错误**
   ```bash
   # 确保在项目根目录运行
   cd /path/to/evolve-ai
   python -m pytest tests/
   ```

2. **异步测试失败**
   ```bash
   # 安装pytest-asyncio
   pip install pytest-asyncio
   ```

3. **内存不足**
   ```bash
   # 减少测试种群大小
   export TEST_POPULATION_SIZE=10
   ```

4. **超时错误**
   ```bash
   # 增加超时时间
   python -m pytest tests/ --timeout=600
   ```

### 调试模式

```bash
# 详细输出
python -m pytest tests/ -v -s

# 只运行失败的测试
python -m pytest tests/ --lf

# 运行特定测试
python -m pytest tests/unit/test_models.py::TestModularMathReasoningNet::test_initialization
```

## 📝 添加新测试

### 单元测试模板

```python
def test_function_name():
    """测试描述"""
    # 准备
    input_data = ...
    
    # 执行
    result = function_to_test(input_data)
    
    # 验证
    assert result == expected_value
```

### 异步测试模板

```python
@pytest.mark.asyncio
async def test_async_function():
    """异步测试描述"""
    # 准备
    input_data = ...
    
    # 执行
    result = await async_function_to_test(input_data)
    
    # 验证
    assert result == expected_value
```

## 🎯 测试最佳实践

1. **测试隔离**: 每个测试应该独立运行
2. **快速反馈**: 单元测试应该在几秒内完成
3. **清晰命名**: 测试名称应该描述测试内容
4. **适当断言**: 使用具体的断言而不是通用断言
5. **错误处理**: 测试应该验证错误情况
6. **性能监控**: 性能测试应该监控资源使用

## 📞 获取帮助

如果遇到测试问题：

1. 检查测试日志
2. 查看 `test_report.json` 中的详细错误信息
3. 运行单个测试进行调试
4. 检查系统资源使用情况

---

**注意**: 本测试套件是Evolve AI项目质量保证的重要组成部分。定期运行测试确保系统稳定性和性能。 