#!/bin/bash
# 一键环境配置脚本
# 解决原项目 3 个独立环境冲突，合并为 2 个互不干扰的环境
#
# 环境划分：
#   train_env   - 训练用 (torch 2.5.1 + llama-factory)
#   deploy_env  - 量化部署用 (torch 2.8.0 + auto-gptq + vllm)
#
# 为什么要分两个环境？
#   - LLaMA-Factory 锁定 torch 2.5.1（稳定）
#   - vLLM 0.11.0 要求 torch >= 2.8.0
#   - 两者无法共存于同一环境，但每个环境内部依赖完全自洽

set -e

echo "================================================"
echo "  环境配置脚本 - 解决原项目依赖冲突"
echo "  原项目: 3 个环境 (torch 2.5.1 / 2.8.0 / 2.8.0)"
echo "  本项目: 2 个环境 (train_env + deploy_env)"
echo "================================================"

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo "[错误] 未找到 conda，请先安装 Anaconda 或 Miniconda"
    exit 1
fi

# ============================================
# 环境 1: train_env (训练)
# ============================================
echo ""
echo "[1/2] 创建训练环境: train_env"
echo "  Python 3.11 + torch 2.5.1 + cuda 12.4"
echo ""

# conda create -n train_env python=3.11 -y
# conda activate train_env
# pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
# pip install -r requirements/requirements-train.txt

# 由于 conda activate 在脚本中可能不生效，建议手动执行以下命令：
echo "请手动执行以下命令创建训练环境:"
echo "  conda create -n train_env python=3.11 -y"
echo "  conda activate train_env"
echo "  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124"
echo "  pip install -r requirements/requirements-train.txt"
echo ""

# ============================================
# 环境 2: deploy_env (量化 + 部署)
# ============================================
echo "[2/2] 创建部署环境: deploy_env"
echo "  Python 3.11 + 与 CUDA 驱动匹配的 vLLM 二进制"
echo ""

echo "请手动执行以下命令创建部署环境:"
echo "  conda create -n deploy_env python=3.11 -y"
echo "  conda activate deploy_env"
echo "  # 先按 vLLM 官方 GPU 安装指南安装匹配的 vLLM wheel"
echo "  pip install -r requirements/requirements-deploy.txt"
echo ""

# ============================================
# 环境使用说明
# ============================================
echo "================================================"
echo "  环境配置完成后的使用流程"
echo "================================================"
echo ""
echo "[步骤 1] 训练 + 合并 (train_env)"
echo "  conda activate train_env"
echo "  bash scripts/03_train_lora.sh configs/qwen25_lora.yaml"
echo "  bash scripts/04_merge_model.sh configs/merge_lora.yaml"
echo ""
echo "[步骤 2] 量化 + 部署 (deploy_env)"
echo "  conda activate deploy_env"
echo "  python scripts/05_quantize_gptq.py"
echo "  bash scripts/06_serve_vllm.sh"
echo "  python scripts/07_call_api.py"
echo ""
echo "================================================"
echo "  冲突解决说明"
echo "================================================"
echo "  原项目痛点:                                   "
echo "    - 3 个环境 (llm_env / gptq_env / 部署环境)  "
echo "    - gptqmodel 与 llama-factory 依赖冲突        "
echo "    - 模型文件手动搬运                           "
echo "                                                "
echo "  本项目方案:                                   "
echo "    - train_env: 用 LLaMA-Factory + DeepSpeed   "
echo "    - deploy_env: 用 AutoGPTQ + vLLM            "
echo "    - 用 AutoGPTQ 替代 gptqmodel，依赖更稳定     "
echo "    - 脚本自动传递路径，无需手动搬运             "
echo "================================================"
