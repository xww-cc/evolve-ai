# 贡献指南

感谢您对Evolve AI项目的关注！我们欢迎所有形式的贡献。

## 🚀 快速开始

### 环境设置

1. **Fork仓库**
   ```bash
   git clone https://github.com/your-username/evolve-ai.git
   cd evolve-ai
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **运行测试**
   ```bash
   python system_test.py
   python performance_monitor.py
   ```

## 📋 贡献类型

### 🐛 Bug修复
- 在Issues中搜索现有的Bug报告
- 创建新的Issue（如果不存在）
- 修复Bug并提交Pull Request

### ✨ 新功能
- 在Issues中讨论新功能
- 创建功能分支：`git checkout -b feature/new-feature`
- 实现功能并提交Pull Request

### 📚 文档改进
- 改进README.md
- 添加代码注释
- 创建教程或示例

### 🧪 测试改进
- 添加单元测试
- 改进测试覆盖率
- 创建性能基准

## 🔧 开发流程

### 1. 创建分支
```bash
git checkout -b feature/your-feature-name
```

### 2. 开发代码
- 遵循PEP 8代码风格
- 添加适当的注释和文档字符串
- 确保代码通过所有测试

### 3. 提交更改
```bash
git add .
git commit -m "feat: 添加新功能描述"
```

### 4. 推送分支
```bash
git push origin feature/your-feature-name
```

### 5. 创建Pull Request
- 使用提供的PR模板
- 详细描述更改
- 链接相关Issue

## 📝 代码规范

### Python代码风格
- 遵循PEP 8
- 使用4个空格缩进
- 行长度不超过88个字符
- 使用有意义的变量名

### 文档字符串
```python
def function_name(param1: str, param2: int) -> bool:
    """函数描述。
    
    Args:
        param1: 参数1描述
        param2: 参数2描述
        
    Returns:
        返回值描述
        
    Raises:
        ValueError: 当参数无效时
    """
    pass
```

### 提交消息格式
```
type(scope): 简短描述

详细描述（可选）

BREAKING CHANGE: 破坏性更改描述（可选）
```

类型：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更改
- `style`: 代码风格更改
- `refactor`: 代码重构
- `test`: 测试更改
- `chore`: 构建过程或辅助工具更改

## 🧪 测试指南

### 运行测试
```bash
# 系统测试
python system_test.py

# 性能测试
python performance_monitor.py

# 状态检查
python system_status.py

# 完整评估
python model_evaluation_report.py
```

### 添加新测试
- 为新功能添加相应的测试
- 确保测试覆盖率达到80%以上
- 测试应该独立且可重复

## 📊 性能要求

### 基准测试
- 种群创建：> 1000 个体/秒
- 评估性能：> 200 个体/秒
- 进化性能：> 100 代/秒
- 系统稳定性：> 95% 成功率

### 内存使用
- 避免内存泄漏
- 合理使用缓存
- 及时清理临时对象

## 🔍 代码审查

### 审查清单
- [ ] 代码符合项目风格
- [ ] 功能按预期工作
- [ ] 测试通过
- [ ] 文档已更新
- [ ] 性能影响已评估
- [ ] 安全性已考虑

### 审查流程
1. 自动CI/CD检查
2. 代码审查者审查
3. 维护者最终批准
4. 合并到主分支

## 🏷️ 标签说明

### Issue标签
- `bug`: Bug报告
- `enhancement`: 功能请求
- `documentation`: 文档改进
- `good first issue`: 适合新手的Issue
- `help wanted`: 需要帮助
- `priority: high/medium/low`: 优先级

### PR标签
- `ready for review`: 准备审查
- `work in progress`: 开发中
- `breaking change`: 破坏性更改
- `dependencies`: 依赖更新

## 📞 获取帮助

### 讨论
- 使用GitHub Discussions
- 加入项目社区

### 问题报告
- 使用Issue模板
- 提供详细的错误信息
- 包含环境信息

### 功能请求
- 详细描述需求
- 提供使用场景
- 考虑实现方案

## 🎯 贡献者等级

### 🌱 新手贡献者
- 修复简单的Bug
- 改进文档
- 添加测试

### 🌿 活跃贡献者
- 实现新功能
- 改进核心算法
- 参与代码审查

### 🌳 核心维护者
- 架构决策
- 版本发布
- 社区管理

## 📄 许可证

通过贡献代码，您同意您的贡献将在MIT许可证下发布。

---

感谢您的贡献！🎉 