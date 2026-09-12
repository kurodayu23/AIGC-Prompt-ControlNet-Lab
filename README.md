# AIGC Prompt / ControlNet Lab

**[下载 Windows x64 运行包](https://github.com/kurodayu23/AIGC-Prompt-ControlNet-Lab/releases/latest)**：下载 `AIGCPromptStudio-v1.0.0-Windows-x64.zip`，完整解压后双击 `AIGCPromptStudio.exe`，无需安装 Python。此包提供提示词组合窗口，不包含 ControlNet 生图模型。

一个可测试的提示词组合器，以及 Stable Diffusion 1.5 + Canny ControlNet 调用示例。项目关注参数组织、图像预处理和模型调用边界，不宣称生产级 MLOps 或固定显存下的性能保证。

## Vibe Coding / AI 辅助开发

本项目采用 AI 辅助开发，通过需求描述与迭代反馈，使用 AI 辅助编写和修改代码。Vibe Coding 在这里描述开发方式；项目的具体功能与完成度以源码、运行说明和验证记录为准。

项目本身提供提示词组合及 ControlNet 模型调用示例。展示重点是参数组织、图像预处理、CPU 路径与可复现测试。

维护时以明确需求、审查代码改动和可复现验证为准；具体测试及尚未验证的部分见下方说明。

## 提示词组合：无需模型

Python 3.11，在仓库根目录运行：

```bash
python -m src.prompt_engine
```

```python
from src.prompt_engine import PromptComposer

composer = PromptComposer()
print(composer.compose("cinematic_portrait", "a small robot").as_sd_payload())
print(composer.compose("cinematic_portrait", "a small robot",
    target_platform="midjourney", mj_param_preset="v6_photoreal").as_midjourney_string())
```

默认模板相对于源码定位，不依赖当前工作目录。可传入自定义 JSON 路径。Midjourney 使用独立的负面提示词，避免把 Stable Diffusion 的括号权重语法混入 `--no`。

## ControlNet 示例

先按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/) 安装与硬件匹配的 PyTorch，再安装项目依赖：

```bash
python -m pip install -r requirements.txt
python -m src.sd_diffusers_pipeline --help
python -m src.sd_diffusers_pipeline path/to/reference.png --subject "a small robot" --template cinematic_portrait --output output/robot.png --seed 42
```

`path/to/reference.png` 需要替换为自己的参考图。首次运行下载基础模型、VAE 与 ControlNet 权重；模型使用需遵守各自条款。

实现顺序：参考图转 RGB/灰度 → Canny 边缘 → PIL 控制图 → 提示词组合 → 模型推理 → 保存输出。CPU 使用 float32，CUDA 使用 float16 和 model CPU offload。`xformers` 默认关闭，可在 Python 接口中显式启用。

基础管线是 **SD 1.5，不是 SDXL**。`apply_weight()` 生成的 SD 字符串采用 A1111 风格，原生 Diffusers 不会自动解释该权重语法，不能据此声称完成了权重嵌入控制。

## 测试

```bash
python -m pip install pytest
python -m pytest -q
```

测试覆盖模板定位、参数错误、平台负面提示词、CPU dtype 和 Canny 控制图传递。模型下载及推理入口在测试中替换为测试对象，因此无需下载权重。这些检查不能证明真实生成效果、显存占用或推理速度；固定 seed 也不保证跨硬件、跨库版本的逐像素一致。

## 构建 Windows 下载包

使用 Python 3.11 x64，建议在独立虚拟环境中运行：

```powershell
python -m pip install -r scripts/build-requirements.txt
.\scripts\package_windows.ps1
```

输出在 `dist/`。脚本会包含运行时和所需资源，生成包含 EXE 与 `_internal` 的 ZIP；发布前需验证解压后的程序。
