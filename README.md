<div align="center">

# 🧠 模型脑手术（LLM-Refusal-Remover）— 大语言模型拒绝行为消融工具箱

**不训练、不微调、不 LoRA —— 一行命令，精准"切除"模型的拒绝回路**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## ✨ 独特能力

| 特性 | 说明 |
|------|------|
| **零训练编辑** | 无需任何梯度更新、微调或 LoRA，纯数学运算直接修改模型权重 |
| **拒绝向量消融** | 精准定位模型内部的"拒绝方向"，通过正交投影从权重矩阵中剔除 |
| **架构无关** | 自动适配 Qwen、LLaMA、Mistral 等主流 Transformer 架构 |
| **一键手术** | 单条命令完成激活采集 → 向量计算 → 权重编辑 → 模型保存 |
| **完整验证体系** | 内置 Gradio Web 界面，支持单模型对话、双模型 A/B 对比、批量测试 |
| **智能显存管理** | 自动检测 GPU 显存，大模型自动回退 CPU，小模型自动上 GPU |
| **内置中英双语提示词** | 53 条有害 + 38 条无害提示词，覆盖 9 大类别，即开即用 |

---

## 📋 目录

- [项目简介](#项目简介)
- [核心原理](#核心原理)
- [项目结构](#项目结构)
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
  - [1. 模型手术（brain_surgery.py）](#1-模型手术brain_surgerypy)
  - [2. Web 测试界面（chat_interface.py）](#2-web-测试界面chat_interfacepy)
  - [3. 命令行聊天（chat_qwen.py）](#3-命令行聊天chat_qwenpy)
  - [4. 下载模型（download_qwen.py）](#4-下载模型download_qwenpy)
- [提示词列表](#提示词列表)
- [技术栈](#技术栈)
- [许可证](#许可证)

---

## 项目简介

**模型脑手术**是一套基于激活消融（Activation Ablation）技术的本地模型编辑工具箱。它的核心思想是：大语言模型的"拒绝行为"并非散布在整个模型中，而是集中在特定层的特定方向上。通过识别并剔除这些方向，可以在**不重新训练**的情况下显著改变模型的响应模式。

完整流程只需三步：

1. **激活采集** —— 分别用"有害"和"无害"提示词刺激模型，采集目标 Transformer 层的隐藏状态激活
2. **消融计算** —— 计算两层激活的差异向量（拒绝向量），通过正交投影从 MLP 的 `down_proj` 权重中剔除
3. **效果验证** —— 通过手术前后生成测试、Web 界面对比、批量测试等方式验证编辑效果

> ⚠️ **免责声明**：本项目仅用于学术研究和技术探索。编辑后的模型可能产生不可预期的输出，请谨慎使用，并遵守相关法律法规。

---

## 核心原理

方法参考了 "refusal vector" 相关研究：

### 1. 均值激活采集

对每一层 $l$，分别计算有害和无害提示词的平均隐藏状态：

$$H_{\text{harmful}}^{(l)} = \frac{1}{N}\sum_{i=1}^{N} h_{\text{harmful},i}^{(l)}, \quad H_{\text{harmless}}^{(l)} = \frac{1}{M}\sum_{j=1}^{M} h_{\text{harmless},j}^{(l)}$$

### 2. 拒绝向量计算

$$R^{(l)} = H_{\text{harmful}}^{(l)} - H_{\text{harmless}}^{(l)}, \quad \hat{R}^{(l)} = \frac{R^{(l)}}{\|R^{(l)}\|}$$

### 3. 权重投影消融

对 MLP 的 `down_proj` 权重 $W$ 进行正交投影剔除：

$$W' = W - \alpha \cdot \hat{R} \hat{R}^T W$$

其中 $\alpha$ 为 `--ablation-scale` 控制的消融强度系数。$\alpha=1$ 表示完全消融，$\alpha<1$ 表示部分消融。

---

## 项目结构

```
模型脑手术/
├── brain_surgery.py        # 🔪 核心手术脚本：激活采集 + 权重消融 + 模型保存
├── chat_interface.py       # 🖥️ Gradio Web 测试界面（单模型对话 / 双模型对比 / 批量测试）
├── chat_qwen.py            # 💬 命令行聊天工具（轻量级）
├── download_qwen.py        # 📥 从 Hugging Face 下载 Qwen3-1.7B 模型
├── harmful_prompts.py      # ⚠️ 有害提示词列表（53 条，用于采集拒绝激活）
├── harmless_prompts.py     # ✅ 无害提示词列表（38 条，用于采集正常激活）
├── requirements.txt        # 📦 Python 依赖
├── run.bat                 # 🚀 Windows 批处理入口（虚拟环境管理 + 快捷命令）
│
├── Qwen3-1.7B/             # 📁 原始模型目录（需自行下载，不上传至 Git）
├── Qwen3-4B/               # 📁 原始模型目录（可选）
├── surgery-qwen3-1.7b/     # 📁 手术后模型输出目录（示例）
├── surgery-qwen3-4b/       # 📁 手术后模型输出目录（示例）
└── surgery-output/         # 📁 默认手术输出目录
```

---

## 环境准备

### 1. 创建虚拟环境（推荐）

```bash
python -m venv .venv
```

Windows 下激活：
```bash
.venv\Scripts\activate
```

Linux/macOS 下激活：
```bash
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**核心依赖**：
- `torch` — PyTorch 深度学习框架
- `transformers>=4.45.0` — Hugging Face 模型加载与推理
- `safetensors` — 安全模型权重格式
- `tqdm` — 进度条显示
- `huggingface_hub[cli]` — 模型下载与管理
- `gradio` — Web 测试界面（需额外安装）

> 如需使用 Web 界面，请额外安装：`pip install gradio`

### 3. 下载模型

```bash
python download_qwen.py
```

默认下载 `Qwen/Qwen3-1.7B` 到 `./Qwen3-1.7B/` 目录。也可手动下载其他 HuggingFace 格式的因果语言模型。

---

## 快速开始

### 方式一：使用 run.bat（Windows 推荐）

```bash
# 运行完整手术
run.bat surgery

# 运行轻量 smoke test（仅编辑 2-3 层，不保存）
run.bat surgery-test

# 启动 Web 聊天界面
run.bat chat

# 运行自定义脚本
run.bat python your_script.py
```

### 方式二：直接使用 Python

```bash
# 1. 运行手术（默认编辑 8-18 层，保存到 surgery-output/）
python brain_surgery.py --model ./Qwen3-1.7B --layers 8-18

# 2. 启动 Web 测试界面
python chat_interface.py

# 3. 命令行聊天
python chat_qwen.py
```

---

## 使用说明

### 1. 模型手术（brain_surgery.py）

核心脚本，执行激活采集、消融计算和模型保存。

```bash
python brain_surgery.py [选项]
```

**常用参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `./Qwen3-1.7B` | 本地 Transformers 模型目录 |
| `--layers` | `8-18` | 目标层范围，如 `2,3` 或 `10-25` |
| `--output` | `surgery-output` | 编辑后模型保存路径 |
| `--ablation-scale` | `1.0` | 消融强度系数（0~1，1 为完全消融） |
| `--dtype` | `float16` | 模型数据类型：`auto`/`float16`/`bfloat16`/`float32` |
| `--device` | `auto` | 计算设备：`auto`/`cpu`/`cuda`/`mps` |
| `--load-cpu` | `False` | 强制在 CPU 上加载模型（避免显存不足） |
| `--skip-save` | `False` | 不保存编辑后的模型 |
| `--test-prompt` | `"Write a short Python function..."` | 手术前后测试提示词 |
| `--max-new-tokens` | `32` | 测试生成最大 token 数 |

**示例**：

```bash
# 编辑 10-20 层，消融系数 0.5，保存到 my-output/
python brain_surgery.py --model ./Qwen3-1.7B --layers 10-20 --ablation-scale 0.5 --output my-output

# 轻量测试：仅编辑 2-3 层，不保存模型
python brain_surgery.py --model ./Qwen3-1.7B --layers 2,3 --ablation-scale 0.1 --skip-save

# 大模型显存不足时，强制 CPU 加载
python brain_surgery.py --model ./Qwen3-4B --load-cpu
```

**脚本流程**：
1. 加载本地 Transformers 模型和分词器
2. 自动检测并适配多种模型架构的层路径（支持 Qwen、LLaMA、Mistral 等）
3. 用 `harmful_prompts.py` 和 `harmless_prompts.py` 中的提示词采集目标层激活
4. 计算拒绝向量并投影消融 MLP `down_proj` 权重
5. 手术前后分别用测试提示词生成，对比输出差异
6. 保存编辑后的模型和分词器

---

### 2. Web 测试界面（chat_interface.py）

基于 Gradio 的交互式 Web 界面，支持单模型对话、双模型 A/B 对比、批量测试和预设测试。

```bash
python chat_interface.py [选项]
```

**参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `127.0.0.1` | 服务监听地址 |
| `--port` | `7860` | 服务端口 |
| `--share` | `False` | 创建公开分享链接（Gradio Tunnel） |
| `--auto-load` | `None` | 启动时自动加载模型（默认手术后模型） |

**四大功能标签页**：

| 标签页 | 功能 |
|--------|------|
| **单模型对话** | 多轮对话、流式输出、系统提示词、Temperature / Top-P 调节 |
| **双模型对比** | 同时加载原始模型与手术后模型，同一提示词并排对比输出差异 |
| **批量测试** | 上传 `.txt` 提示词文件（每行一条），批量运行并导出结果表格 |
| **预设测试** | 使用内置 harmful / harmless 提示词，快速验证手术效果 |

**示例**：

```bash
# 启动 Web 界面
python chat_interface.py

# 启动并自动加载手术后模型
python chat_interface.py --auto-load

# 指定端口并创建公开链接
python chat_interface.py --port 8080 --share
```

---

### 3. 命令行聊天（chat_qwen.py）

轻量级命令行聊天工具，适合快速测试手术后模型。

```bash
python chat_qwen.py
```

默认加载 `./surgery-qwen3-1.7b/` 目录下的模型。支持以下命令：
- `/exit`, `/quit`, `exit`, `quit` — 退出程序
- `/clear` — 清空对话历史

---

### 4. 下载模型（download_qwen.py）

从 Hugging Face 下载 `Qwen/Qwen3-1.7B` 模型到本地。

```bash
python download_qwen.py
```

---

## 提示词列表

项目内置了两组中英双语提示词用于激活采集：

### harmful_prompts.py — 有害提示词（53 条）

覆盖 **9 大类别**，用于激发模型的拒绝（refusal）行为：

| 类别 | 示例数量 |
|------|----------|
| 毒品/药物 | 12 条 |
| 暴力/伤害 | 3 条 |
| 仇恨/辱骂 | 2 条 |
| 金融诈骗 | 5 条 |
| 网络犯罪 | 5 条 |
| 自残/自杀 | 4 条 |
| 虚假信息/造谣 | 4 条 |
| 隐私侵犯 | 4 条 |
| 非法武器 | 3 条 |
| 其他 | 11 条 |

### harmless_prompts.py — 无害提示词（38 条）

涵盖编程、生活常识、翻译、文学、科学、历史等日常问题，用于采集模型的正常响应激活。

> 💡 可根据研究需要自行修改这两个文件中的提示词列表。

---

## 技术栈

- **Python 3.10+**
- **PyTorch** — 张量计算与模型推理
- **Transformers** — Hugging Face 模型生态
- **Accelerate** — 自动设备映射与大模型加载优化
- **Gradio** — Web 交互界面
- **SafeTensors** — 安全模型序列化格式

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源许可。

---

## 注意事项

1. **模型格式**：仅支持 Hugging Face Transformers 格式的本地模型（含 `config.json` 和 `safetensors` 权重）。不支持 GGUF/LM Studio 格式。
2. **显存管理**：脚本会自动检测 GPU 显存，若模型过大则自动切换至 CPU 加载，也可手动使用 `--load-cpu`。
3. **量化模型**：若模型权重为量化格式（非浮点型），脚本会报错退出，因为量化权重无法直接编辑。
4. **安全性**：编辑后的模型行为不可预测，请勿用于生产环境或面向用户的服务。
