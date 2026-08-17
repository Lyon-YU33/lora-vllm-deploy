#!/bin/bash
# ============================================================
#  模型合并脚本: 将 LoRA 适配器合并到基座模型
#  使用 LLaMA-Factory 标准 YAML 配置文件（merge_lora.yaml）
#  环境: train_env
# ============================================================
#
# 使用方法:
#   bash scripts/04_merge_model.sh
#
# 验证合并是否成功:
#   1. merged_model/qwen25-7b-lora-merged/ 下应有完整的模型文件
#   2. 文件总大小约 14GB（7B 模型 FP16）
#   3. 用 transformers 加载测试能正常初始化
# ============================================================

set -e

cd "$(dirname "$0")/.."
# conda activate train_env

YAML_CONFIG=${1:-configs/merge_lora.yaml}

echo "================================================"
echo "  合并 LoRA 权重到基座模型"
echo "  配置: $YAML_CONFIG"
echo "================================================"

# 校验 LoRA 权重存在
LORA_DIR="outputs/qwen25-7b-lora"
if [ ! -f "$LORA_DIR/adapter_model.safetensors" ]; then
    echo "[错误] LoRA 权重不存在: $LORA_DIR/adapter_model.safetensors"
    echo "       请先运行 bash scripts/03_train_lora.sh"
    exit 1
fi
echo "[校验] LoRA 权重存在 ✓"

# 执行合并（用 CPU 导出避免显存占用）
echo ""
echo "[启动] 开始合并..."
llamafactory-cli export "$YAML_CONFIG"

echo ""
echo "================================================"
echo "  合并完成！验证清单:"
echo "================================================"

MERGED_DIR="merged_model/qwen25-7b-lora-merged"

echo ""
echo "[1] 检查模型文件:"
ls -lh "$MERGED_DIR" 2>/dev/null

echo ""
echo "[2] 检查文件大小（预期约 14GB）:"
TOTAL_SIZE=$(du -sh "$MERGED_DIR" 2>/dev/null | cut -f1)
echo "  总大小: $TOTAL_SIZE"

echo ""
echo "[3] 加载测试（验证模型可用性）:"
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
print('  [加载] tokenizer...')
tok = AutoTokenizer.from_pretrained('$MERGED_DIR', trust_remote_code=True)
print('  [加载] model...')
model = AutoModelForCausalLM.from_pretrained('$MERGED_DIR', torch_dtype=torch.float16, trust_remote_code=True)
print('  ✓ 模型加载成功!')
print(f'  - 词表大小: {tok.vocab_size}')
print(f'  - 模型参数量: {sum(p.numel() for p in model.parameters())/1e9:.2f}B')
" || echo "  ✗ 加载测试失败，请检查模型文件完整性"

echo ""
echo "================================================"
echo "  下一步: GPTQ 量化"
echo "  conda activate deploy_env"
echo "  python scripts/05_quantize_gptq.py"
echo "================================================"
