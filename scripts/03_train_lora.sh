#!/bin/bash
# ============================================================
#  LoRA 微调训练启动脚本
#  使用 LLaMA-Factory 标准 YAML 配置文件（推荐方式）
#  环境: train_env (torch 2.5.1 + llama-factory)
# ============================================================
#
# 使用方法:
#   bash scripts/03_train_lora.sh                 # 用默认配置
#   bash scripts/03_train_lora.sh configs/your.yaml  # 用自定义配置
#
# 验证训练是否成功启动:
#   1. 终端输出 "***** Running training *****" 表示开始训练
#   2. 每隔 10 步会打印 loss（应小于 5.0 并逐步下降）
#   3. 训练完成后 outputs/qwen25-7b-lora/ 下会有 adapter_model.safetensors
#   4. 用 nvidia-smi 检查 GPU 占用应在 80% 以上
# ============================================================

set -e

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 激活环境（手动取消注释）
# conda activate train_env

# 配置文件（支持参数传入）
YAML_CONFIG=${1:-configs/qwen25_lora.yaml}

echo "================================================"
echo "  LoRA 微调训练"
echo "  框架: LLaMA-Factory + DeepSpeed ZeRO-2"
echo "  模型: Qwen2.5-7B-Instruct"
echo "  配置: $YAML_CONFIG"
echo "================================================"

# 校验配置文件
if [ ! -f "$YAML_CONFIG" ]; then
    echo "[错误] 配置文件不存在: $YAML_CONFIG"
    exit 1
fi

# 校验数据文件
DATA_FILE="data/sample_dataset.json"
if [ ! -f "$DATA_FILE" ]; then
    echo "[错误] 数据文件不存在: $DATA_FILE"
    exit 1
fi
echo "[校验] 数据文件存在: $DATA_FILE"

# 校验 GPU 可用性
echo "[校验] GPU 状态:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || {
    echo "[错误] 无法访问 GPU，请检查 CUDA 驱动"
    exit 1
}

echo ""
echo "[启动] 开始训练..."
echo "  - 训练日志将输出到终端"
echo "  - TensorBoard 日志: outputs/qwen25-7b-lora/runs/"
echo "  - 训练 loss 曲线: outputs/qwen25-7b-lora/training_loss.png"
echo ""

# 核心: 使用 YAML 配置文件训练（LLaMA-Factory 标准用法）
llamafactory-cli train "$YAML_CONFIG"

echo ""
echo "================================================"
echo "  训练完成！验证清单:"
echo "================================================"
echo ""
echo "[1] 检查 LoRA 权重文件:"
ls -lh outputs/qwen25-7b-lora/adapter_model.safetensors 2>/dev/null && echo "  ✓ 权重文件存在" || echo "  ✗ 权重文件缺失"

echo ""
echo "[2] 检查 checkpoint 目录:"
ls outputs/qwen25-7b-lora/ 2>/dev/null

echo ""
echo "[3] 查看训练 loss 曲线:"
echo "  tensorboard --logdir outputs/qwen25-7b-lora/runs/ --port 6006"

echo ""
echo "[4] 下一步: 合并 LoRA 权重"
echo "  bash scripts/04_merge_model.sh"
echo "================================================"
