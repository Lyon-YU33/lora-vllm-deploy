#!/bin/bash
# vLLM 推理服务启动脚本
# 使用 deploy_env 环境 (torch 2.8.0 + vllm 0.11.0)

set -e

echo "================================================"
echo "  vLLM 推理服务启动"
echo "  框架: vLLM 0.11.0 + PagedAttention"
echo "  模型: Qwen2.5-7B GPTQ 量化版"
echo "================================================"

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 激活部署环境
# conda activate deploy_env

# 默认参数
MODEL_PATH=${1:-quantized_model/qwen25-7b-gptq}
PORT=${2:-8000}
GPU_MEMORY_UTILIZATION=${3:-0.85}
MAX_MODEL_LEN=${4:-4096}
TENSOR_PARALLEL_SIZE=${5:-1}

echo "[信息] 模型路径:        $MODEL_PATH"
echo "[信息] 端口:            $PORT"
echo "[信息] GPU 显存利用率:  $GPU_MEMORY_UTILIZATION"
echo "[信息] 最大序列长度:    $MAX_MODEL_LEN"
echo "[信息] 张量并行大小:    $TENSOR_PARALLEL_SIZE"
echo ""

# 启动 vLLM 服务（OpenAI 兼容 API）
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "qwen25-7b-summarizer" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --quantization "gptq" \
    --trust-remote-code \
    --dtype half

# 启动后可访问:
#   - 健康检查: http://localhost:$PORT/health
#   - 模型列表: http://localhost:$PORT/v1/models
#   - 推理 API: http://localhost:$PORT/v1/chat/completions
