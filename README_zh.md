[English](README.md) | [简体中文](README_zh.md) | [日本語](README_ja.md)

---

# 🎨 AIGC 提示词工程与 ControlNet 管线实验室

> **一座连接"非结构化艺术灵感"与"确定性工程管线"的生产级桥梁。**

大多数生成式 AI 仓库容易走入两个极端：要么是一堆纯文本的提示词合集，要么是极其混乱的 Jupyter Notebook 代码。本仓库旨在展示纯粹的 **"生成式 AI 工程化能力"**，将面向不同平台（Midjourney V6 与 SDXL）的动态提示词编译系统，与程序化的 Stable Diffusion ControlNet 管线进行了深度统一。

## 🏗️ 核心架构与能力

本实验室由两个相互补充的引擎组成：

### 1. 提示词编译引擎 (`src/prompt_engine.py`)
一个确定性的提示词编译器，抹平了不同生成平台之间的语法差异。

*   **语法动态转换**：自动处理不同平台的 Token 权重语法（例如 SD 的 `(token:1.5)` 对比 Midjourney 的 `token::1.5`）。
*   **全局负面提示注入**：统一管理底层负面提示词，无论如何生成，都能保底控制输出质量。
*   **Midjourney 参数序列化**：针对 V6 版本，自动注入长宽比、艺术风格化及混沌值（例如 `--ar 16:9 --v 6.0 --s 250`）。
*   **JSON 矩阵解耦**：将提示词艺术设计从代码库中抽离。非技术的艺术家可以直接修改 `prompts/midjourney_prompt_matrix.json` 文件，而完全不必触碰 Python 调度代码。

### 2. Vibe AIGC 生图管线 (`src/sd_diffusers_pipeline.py`)
一个完整的 `diffusers` 深度管线，专为本地 GPU 推理展示了正规的 MLOps 处理流程。

*   **极度优化的显存 (VRAM) 占用**：实现了显存极速释放 (`enable_model_cpu_offload()`) 以及基于 `xformers` 的内存高效注意力机制。哪怕是 8GB 显存的显卡也能顺滑运行复杂的 ControlNet 拓扑流。
*   **Canny 边缘智能提取**：使用 `OpenCV` (cv2) 在推理前全自动从参考图中提取边缘轮廓，用于后续结构的精准约束。
*   **高保真自动编码器 (VAE)**：在流水线初始化时，热替换默认 VAE 为 `stabilityai/sd-vae-ft-mse`，彻底消除在确定性种子生成时的色彩偏移或画面发灰问题。

## 🚀 快速启动

### 1. 环境配置
```bash
# 推荐：先创建一个 conda 或 venv 虚拟环境
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate xformers opencv-python
```

### 2. 跨平台提示词编译

```python
from prompt_engine import PromptComposer
import json

composer = PromptComposer()

# 为 Stable Diffusion 编译（自动处理 SD 特有权重引擎和底层结构性负面词）
sd_prompt = composer.compose(
    template_name="cyberpunk_concept", 
    subject="女式仿生人散步", 
    extra_positives=["trending on artstation", composer.apply_weight("镜头光晕", 1.5, "sd")], 
    target_platform="sd"
)
print(json.dumps(sd_prompt.as_sd_payload(), indent=2))

# 为 Midjourney 编译（参数注入系统）
mj_prompt = composer.compose(
    template_name="cinematic_portrait", 
    subject="一位精灵游侠", 
    target_platform="midjourney", 
    mj_param_preset="v6_photoreal"
)
print(mj_prompt.as_midjourney_string()) 
# 输出如: /imagine prompt: Cinematic portrait shot of an elven ranger... --v 6.0 --style raw --ar 16:9 --q 2 --s 250
```

### 3. 程序化生图调用 (SD + ControlNet)

```python
from sd_diffusers_pipeline import VibeAIGCPipeline

pipeline = VibeAIGCPipeline(use_xformers=True)

output_path = pipeline.run_canny_generation(
    template="cyberpunk_concept",
    subject="女式仿生人散步",
    reference_image_path="assets/input.jpg",
    output_path="assets/showcase_out.png",
    seed=8848
)
```

## 🧠 这为什么重要（工程师的视角）
在企业级工作流中，**提示词的生成绝不应该靠手工拼接，图像的生成也绝不能完全依赖各类可视化 UI（如 WebUI/ComfyUI）**——特别是当你想把它部署为一个微服务的时候。

1.  **可复现性（Reproducibility）**：通过强控 Seed、推理步数和纯 Python 类抽象约束的组合生成，我们能够为 AI 结果建立确定的自动化测试。
2.  **高度模块化（Modularity）**：将 JSON 作为映射配置。当你需要改变业务生图的艺术风格风向时，只更新配置 JSON 即可，生图的微服务架构可以实现零停机 (Zero-downtime) 部署。
3.  **硬件感知（Hardware Awareness）**：管线具备底层的 OS 硬件感知逻辑（CUDA 退回机制），强制进行了显存的溢出管理。

*(该仓库直接证明了你在多模态大模型 AI 调度协同模式 (Multimodal Orchestration) 上的系统级架构设计深度)。*
