VLM辅助台球Agent - 使用指南
===============================================

## 架构概述

这个VLM辅助agent结合了视觉语言模型的战略推理和精确的物理计算搜索：

1. drawer.py - 生成台球对局的可视化图片
2. chat.py - 调用VLM API获取战略指导
3. VlmAssistedAgent.py - VLM引导的MCTS搜索agent

## 文件说明

### drawer.py
- BilliardsDrawer类：绘制台球对局俯视图
- 支持标注我方/对方目标球、白球位置、建议shot等
- 生成适合VLM理解的高质量图片

### chat.py
- VLMChat类：VLM API接口
- 支持OpenAI GPT-4V、Claude Vision等
- 自动降级到启发式策略（当VLM不可用时）

### VlmAssistedAgent.py
- VLMAssistedAgent类：主决策agent
- 完整流程：图片生成 → VLM推理 → VLM引导的候选生成 → 并行评估 → BO优化
- 智能时间管理（考虑VLM调用开销）
- 支持纯启发式模式（无需VLM）

## 使用方法

### 方式1: 纯启发式模式（无需VLM API，推荐用于测试）

在agent.py中：
```python
AGENT_TYPE = 'vlm'
```

在NewAgent初始化中：
```python
self.agent = VLMAssistedAgent(
    use_vlm=False,  # 禁用VLM
    n_cores=8
)
```

### 方式2: 使用OpenAI GPT-4V

1. 设置API密钥：
   export OPENAI_API_KEY="your-api-key"

2. 在agent.py中：
```python
AGENT_TYPE = 'vlm'
```

3. 在NewAgent初始化中：
```python
self.agent = VLMAssistedAgent(
    vlm_provider='openai',
    vlm_model='gpt-4-vision-preview',
    use_vlm=True,
    n_cores=8
)
```

### 方式3: 使用Claude Vision

1. 安装anthropic包：
   pip install anthropic

2. 设置API密钥：
   export ANTHROPIC_API_KEY="your-api-key"

3. 在agent.py中：
```python
self.agent = VLMAssistedAgent(
    vlm_provider='claude',
    vlm_model='claude-3-opus-20240229',
    use_vlm=True,
    n_cores=8
)
```

## 运行

```bash
# 激活环境
conda activate poolenv

# 测试VLM agent组件
python test_vlm_agent.py

# 运行评估
python evaluate.py
```

## VLM调用策略

为了平衡性能和成本，VLM不是每个决策都调用：

- 每局游戏的第一个决策：调用VLM
- 之后每4个决策：调用VLM一次
- 其他决策：复用上次的VLM指导

这样可以在保持战略指导的同时控制API调用次数和响应时间。

## 时间管理

- VLM调用预留3-4秒（根据历史动态调整）
- 候选评估使用并行处理（多核加速）
- BO优化使用剩余时间精细调整
- 总时间预算：根据剩余时间和决策数自适应分配

## 性能预期

### 纯启发式模式（use_vlm=False）
- 决策时间：5-8秒/决策
- 预期胜率：60-70%（基于规则的启发式）
- 适合：测试、快速迭代

### VLM辅助模式（use_vlm=True）
- 决策时间：8-12秒/决策（含VLM调用）
- 预期胜率：70-80%（理论上，取决于VLM质量）
- 适合：最终评估、研究VLM能力

## 调试

查看详细日志：
- [VLMAgent] 前缀：agent主流程
- [VLMChat] 前缀：VLM通信
- [VLMTimeManager] 前缀：时间管理

## 注意事项

1. VLM API调用有成本，建议先用use_vlm=False测试
2. 如果网络不稳定，VLM会自动降级到启发式
3. 图片生成无额外成本，即使use_vlm=False也可以保存图片用于分析
4. 建议至少4个CPU核心以获得好的并行加速

## 扩展

### 添加新的VLM提供商

在chat.py的VLMChat类中添加新的provider分支即可。

### 调整VLM提示词

修改chat.py中的_build_strategy_prompt方法。

### 修改候选生成策略

修改VlmAssistedAgent.py中的generate_vlm_guided_candidates函数。

## 故障排查

问题：ImportError: No module named 'openai'
解决：pip install openai

问题：VLM调用超时
解决：检查网络连接，或设置use_vlm=False

问题：图片无法保存
解决：检查/tmp目录权限，或修改drawer.py中的保存路径

