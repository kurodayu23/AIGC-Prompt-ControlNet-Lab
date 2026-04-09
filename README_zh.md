[English](README.md) | [简体中文](README_zh.md) | [日本語](README_ja.md)

---

# AIGC 提示词工程 + ControlNet 工作流

这个仓库用于展示“生成式 AI 工程化能力”，而不是单次出图。

它把两件事放在同一个代码仓库里：
- 提示词模板库管理（可复用、可参数化）
- Stable Diffusion + ControlNet 程序化生成链路

---

## 为什么要合并

很多作品会把 Midjourney 提示词和 Stable Diffusion 脚本分成两个仓库，结果是：
- 复用差
- 评审时不清楚你到底会“提示词设计”还是会“工程落地”

这里直接合并成一个“端到端 AIGC 实验仓库”，面试官更容易评估完整能力。

---

## 核心文件

- `prompts/midjourney_prompt_matrix.json`
: 提示词模板和参数字典，支持固定模板复用。

- `src/prompt_engine.py`
: `PromptComposer` 负责模板加载与参数化拼接。

- `src/sd_diffusers_pipeline.py`
: 把提示词、参考图边缘、ControlNet 推理串成标准流程。

---

## 典型流程

1. 从模板库选一个模板（如 `cinematic_portrait`）
2. 用业务参数填充主题词（subject/scene/style）
3. 对参考图做 Canny 边缘提取
4. 调用 ControlNet 管线生成结果图

---

## 使用建议

- 如果目标是“角色一致性”，优先控制 seed 和模板变量
- 如果目标是“结构一致性”，优先依赖 ControlNet 参考图
- 先固定模板，再微调风格词，避免一次改太多变量

---

## 环境提示

建议：
- Python 3.11+
- CUDA 可用
- 12GB+ 显存（更稳）

该仓库没有强制锁定完整依赖版本，你可以按本地 CUDA 环境选择合适的 `torch` 与 `diffusers` 组合。

## License

MIT
