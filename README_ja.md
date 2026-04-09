[English](README.md) | [简体中文](README_zh.md) | [日本語](README_ja.md)

---

# 🎨 AIGC プロンプトエンジニアリング & ControlNet MLOps ラボ

> **「非構造化されたアートプロンプト」と「決定論的なエンジニアリングパイプライン」をつなぐ本番グレードの架け橋。**

ほとんどの生成 AI リポジトリは、純粋なプロンプトのテキストファイルであるか、混沌とした Jupyter Notebook になっているかのどちらかです。本リポジトリは、Midjourney V6 と SDXL のプラットフォームにまたがる動的なプロンプトコンパイルシステムと、プログラムによる Stable Diffusion ControlNet パイプラインを統合することで、純粋な **「生成 AI エンジニアリング能力」** を示します。

## 🏗️ コアアーキテクチャと機能

このラボは2つの補完的なエンジンに分かれています：

### 1. プロンプト構成エンジン (`src/prompt_engine.py`)
生成プラットフォーム間の構文の違いを抽象化する、決定論的なプロンプトコンパイラ。

*   **構文の動的変換**: トークンの重み付け構文を動的にコンパイルします（SDの `(token:1.5)` とMidjourneyの `token::1.5` の違いを吸収）。
*   **ユニバーサルなネガティブプロンプト**: 一元化されたネガティブプロンプトを自動注入し、生成プラットフォームに関わらずベースライン品質を確保します。
*   **Midjourney パラメータのシリアライズ**: アスペクト比、スタイル、カオス値などの V6 固有パラメータを自動的に追加します（例: `--ar 16:9 --v 6.0 --s 250`）。
*   **JSON マトリックスによる分離**: プロンプトアートのデザインをコードベースから分離します。非技術系のアーティストが Python コードを壊すことなく `prompts/midjourney_prompt_matrix.json` を更新できます。

### 2. Vibe AIGC 画像生成パイプライン (`src/sd_diffusers_pipeline.py`)
ローカル GPU 推論に合わせた本格的な MLOps ワークフローを示す `diffusers` パイプライン。

*   **VRAM フットプリントの最適化**: 逐次 CPU オフロード（`enable_model_cpu_offload()`）と `xformers` による効率的なメモリアテンションを実装。8GB の VRAM 搭載カードでも複雑な ControlNet グラフを処理可能です。
*   **Canny エッジ抽出**: 推論前に `OpenCV` (cv2) を使用して、参照画像から輪郭を全自動で抽出し、強力な構造的条件付けを行います。
*   **高忠実度オートエンコーダ (VAE)**: 決定論的シードでの生成中の不快な色ずれを防ぐために、デフォルトのVAEを `stabilityai/sd-vae-ft-mse` にホットスワップします。

## 🚀 クイックスタート

### 1. 環境構築
```bash
# 推奨: conda/venv 環境を最初に作成
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate xformers opencv-python
```

### 2. プロンプトのコンパイル (クロスプラットフォーム)

```python
from prompt_engine import PromptComposer
import json

composer = PromptComposer()

# Stable Diffusion 用にコンパイル (トークンの重み付けと構造的なネガティブワードを自動処理)
sd_prompt = composer.compose(
    template_name="cyberpunk_concept", 
    subject="散歩する女性アンドロイド", 
    extra_positives=["trending on artstation", composer.apply_weight("レンズフレア", 1.5, "sd")], 
    target_platform="sd"
)
print(json.dumps(sd_prompt.as_sd_payload(), indent=2))

# Midjourney 用にコンパイル (パラメータ注入システム)
mj_prompt = composer.compose(
    template_name="cinematic_portrait", 
    subject="エルフのレンジャー", 
    target_platform="midjourney", 
    mj_param_preset="v6_photoreal"
)
print(mj_prompt.as_midjourney_string()) 
# 出力例: /imagine prompt: Cinematic portrait shot of an elven ranger... --v 6.0 --style raw --ar 16:9 --q 2 --s 250
```

### 3. プログラムによる画像生成呼び出し (SD + ControlNet)

```python
from sd_diffusers_pipeline import VibeAIGCPipeline

pipeline = VibeAIGCPipeline(use_xformers=True)

output_path = pipeline.run_canny_generation(
    template="cyberpunk_concept",
    subject="散歩する女性アンドロイド",
    reference_image_path="assets/input.jpg",
    output_path="assets/showcase_out.png",
    seed=8848
)
```

## 🧠 なぜこれが重要なのか（エンジニアの視点）
エンタープライズのワークフローにおいて、プロンプトの生成は手動であってはならず、画像をマイクロサービスとしてデプロイしたい場合、生成プロセスを完全に UI（WebUI/ComfyUIなど）に依存するべきではありません。

1.  **再現性 (Reproducibility)**: シード値、生成パラメータ、および Python クラスによる動的プロンプトを強力に制御することで、AI 出力に確固たる自動テストを保証します。
2.  **モジュール性 (Modularity)**: JSON をマッピング設定として機能させます。アートディレクションを更新する場合、マイクロサービスの再デプロイ（ゼロダウンタイム）を要求することなく、JSON を変更するだけで済みます。
3.  **ハードウェアアウェアネス (Hardware Awareness)**: パイプラインは OS レベルのデバイス機能を明示的にチェックし（CUDA フォールバック）、RAM/VRAM のオーバーフロー制約を注入しています。

*(このリポジトリは、マルチモーダル AI オーケストレーションパターンにおけるシステムレベルの深いアーキテクチャ設計能力を証明しています)。*
