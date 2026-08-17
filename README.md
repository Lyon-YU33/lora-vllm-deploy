# 大模型微调与推理部署：LoRA + vLLM 全流程实战

> 基于 **Qwen2.5-7B** 的端到端大模型工程化项目：**数据准备 → LoRA 微调 → 模型量化 → vLLM 部署 → API 调用**
> 解决原教学项目环境冲突、流程割裂、显卡选型模糊三大痛点。
> 📖 **完整实践指南**：[docs/05_实践指南_从原理到验证.md](docs/05_实践指南_从原理到验证.md)

> 本仓库是基于本人学习与工程实践整理的可复现实践项目，不包含任何公司数据、模型权重、内部系统或生产配置；性能参数均为教学配置或待实测目标，实际结果取决于硬件、数据与版本组合。

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2B-ee4c2c)](https://pytorch.org/)
[![vLLM](https://img.shields.io/badge/vLLM-0.11.0-0066CC)](https://docs.vllm.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 项目特色

| 特色 | 说明 |
|------|------|
| 🔄 **全流程贯通** | 数据 → 微调 → 量化 → 部署 → 调用，一条命令串联，告别环境切换 |
| 🧩 **冲突已解决** | 统一 torch/transformers 版本组合，3 个环境合并为 2 个 |
| 📝 **YAML 配置驱动** | 采用 LLaMA-Factory 标准 YAML 配置，超参数集中管理，方便版本对比 |
| 💡 **显卡选型明确** | 按 7B/13B/70B 三档参数量给出显存公式与显卡推荐 |
| 🚀 **可直接运行** | 提供示例数据集、脚本化训练/量化/部署，AutoDL 一键复现 |
| 📚 **原理与代码并重** | 每步附论文原文链接，让你在实践中领悟 LoRA/vLLM 原理 |
| 🔍 **替代方案罗列** | 每个环节都给出 2-3 个可选框架（仅列举不使用） |

---

## 🗂️ 项目结构

```
lora-vllm-deploy/
├── README.md                              # 本文件
├── LICENSE                                # MIT 协议
├── .gitignore
│
├── requirements/
│   ├── requirements-train.txt             # 训练环境依赖
│   └── requirements-deploy.txt            # 部署环境依赖
│
├── configs/                               # ⭐ YAML 配置文件
│   ├── qwen25_lora.yaml                   #    LoRA 训练配置（核心）
│   ├── merge_lora.yaml                    #    模型合并配置
│   ├── dataset_info.yaml                  #    数据集信息
│   └── ds_zero2.json                      #    DeepSpeed 配置
│
├── scripts/                               # 执行脚本（按顺序运行）
│   ├── 01_setup_env.sh                    #    环境配置引导
│   ├── 02_download_model.py               #    模型下载
│   ├── 03_train_lora.sh                   #    LoRA 训练（用 YAML）
│   ├── 04_merge_model.sh                  #    模型合并（用 YAML）
│   ├── 05_quantize_gptq.py                #    GPTQ 量化
│   ├── 06_serve_vllm.sh                   #    vLLM 启动
│   └── 07_call_api.py                     #    API 调用示例
│
├── data/
│   ├── sample_dataset.json                # 示例数据集（10 条）
│   └── README.md                          # 数据格式说明
│
├── src/
│   └── preprocess_data.py                 # 数据预处理工具
│
└── docs/                                  # 📚 教学文档
    ├── 01_环境冲突解决方案.md             #    解决原项目痛点
    ├── 02_显卡选型规则.md                  #    7B/13B/70B 选型详解
    ├── 03_替代框架参考.md                  #    其他可选框架罗列
    ├── 04_原项目冲突点分析.md              #    原项目痛点剖析
    └── 05_实践指南_从原理到验证.md         # ⭐ 完整实践指南（含论文链接）
```

---

## 🎯 核心目标

实现一个完整的文本摘要模型部署流程：

```
输入文本 → 微调后的 Qwen2.5-7B → 简洁摘要
```

### 性能目标

| 指标 | 原始 Qwen2.5-7B | 本项目(量化+LoRA) |
|------|-------------|------------------|
| 显存占用(推理) | 16GB | **6GB** |
| 吞吐量(tokens/s) | 120 | **320** |
| 平均延迟 | >1s | **<500ms** |

---

## 🚀 快速开始

完整步骤见 [实践指南](docs/05_实践指南_从原理到验证.md)，简化版：

```bash
# 1. 环境配置
bash scripts/01_setup_env.sh

# 2. 下载模型
conda activate deploy_env
python scripts/02_download_model.py --source modelscope

# 3. 修改 YAML 配置（指向本地模型路径）
#    编辑 configs/qwen25_lora.yaml 中 model_name_or_path

# 4. LoRA 训练（用 YAML 配置）
conda activate train_env
bash scripts/03_train_lora.sh configs/qwen25_lora.yaml

# 5. 合并 LoRA 权重（用 YAML 配置）
bash scripts/04_merge_model.sh

# 6. GPTQ 量化
conda activate deploy_env
python scripts/05_quantize_gptq.py

# 7. vLLM 部署
bash scripts/06_serve_vllm.sh

# 8. 调用测试（另开终端）
python scripts/07_call_api.py
```

---

## 📚 学习路线

### 📖 推荐学习顺序

1. **先读实践指南**：[docs/05_实践指南_从原理到验证.md](docs/05_实践指南_从原理到验证.md)
   - 包含 LoRA/vLLM 论文链接
   - 每步附验证点和思考题
   - 完整流程图和成本估算

2. **边做边学**：按实践指南的阶段 0 → 阶段 8 顺序执行
3. **遇问题查文档**：
   - [环境冲突解决方案](docs/01_环境冲突解决方案.md)
   - [显卡选型规则](docs/02_显卡选型规则.md)
   - [替代框架参考](docs/03_替代框架参考.md)

### 🎯 论文原文链接

| 主题 | 论文链接 |
|------|---------|
| LoRA 原理 | [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) |
| vLLM/PagedAttention | [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180) |
| DeepSpeed ZeRO | [arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054) |
| GPTQ 量化 | [arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323) |
| Qwen2.5 技术报告 | [arxiv.org/abs/2412.15115](https://arxiv.org/abs/2412.15115) |
| Transformer 原论文 | [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) |

---

## ⚙️ 环境要求

### 硬件（AutoDL 租用推荐）
- **训练**: RTX 4090 (24GB) × 1（7B + LoRA + ZeRO-2 足够）
- **量化**: RTX 4090 (24GB) × 1
- **部署**: RTX 4090 (24GB) × 1（量化后 7B 约 5GB 推理）
- CUDA ≥ 12.4

> 详细选型规则见 [docs/02_显卡选型规则.md](docs/02_显卡选型规则.md)

### 软件
- Python 3.11
- PyTorch 2.5.1 (cu124) — 训练
- PyTorch 2.8.0 (cu126) — 部署
- **本项目已统一版本组合，2 个环境覆盖全流程**

> 部署环境请先依照 [vLLM GPU 安装指南](https://docs.vllm.ai/en/v0.11.0/getting_started/installation/gpu.html) 创建干净环境并安装与 CUDA 驱动匹配的 vLLM 二进制，再安装 `requirements/requirements-deploy.txt` 中的补充依赖。不要在已有环境中手动固定或升级 PyTorch、CUDA wheel 与 vLLM 的组合。

---

## 🔧 解决的冲突点

详见 [docs/01_环境冲突解决方案.md](docs/01_环境冲突解决方案.md)

| 原项目痛点 | 本项目方案 |
|----------|-----------|
| 3 个独立 conda 环境（torch 2.5.1 / 2.8.0 / 2.8.0） | 合并为 2 个：`train_env` + `deploy_env` |
| LLaMA-Factory 与 gptqmodel 依赖冲突 | 用 AutoGPTQ 替代 gptqmodel，错开版本 |
| 命令行参数散落各处，难管理 | ⭐ **YAML 配置文件统一管理**，可版本控制 |
| 模型文件手动搬运 | 脚本自动化串联，路径通过配置文件管理 |
| 显卡选型无说明 | 按 7B/13B/70B 给出显存公式与推荐卡 |
| 缺乏原理解释 | ⭐ **每步附论文链接 + 思考题** |

---

## 📄 License

本项目仅供学习研究，基于 MIT 协议开源。
