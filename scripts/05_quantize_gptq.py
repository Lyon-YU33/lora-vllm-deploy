"""
GPTQ 模型量化脚本

使用 AutoGPTQ 将合并后的模型量化为 4-bit，大幅降低显存占用和推理延迟
注意: 本脚本在 deploy_env (torch 2.8.0) 环境下运行

量化效果（以 Qwen2.5-7B 为例）:
    - 模型大小: 14GB → 4GB
    - 推理显存: 16GB → 6GB
    - 推理速度: 提升 2-3 倍

使用方法:
    python scripts/05_quantize_gptq.py \
        --model merged_model/qwen25-7b-lora-merged \
        --output quantized_model/qwen25-7b-gptq

解决原项目冲突点:
    - 原项目使用 gptqmodel (与 llama-factory 冲突)
    - 本项目改用 AutoGPTQ，在 deploy_env 中独立运行
"""

import argparse
import os
from pathlib import Path


def quantize_model(
    model_path: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    calibration_samples: int = 128,
    max_seq_length: int = 1024,
):
    """
    使用 AutoGPTQ 进行 4-bit 量化

    Args:
        model_path: 待量化的模型路径
        output_dir: 量化后模型保存路径
        bits: 量化位数（4 或 8）
        group_size: 量化组大小，影响精度
        calibration_samples: 校准样本数量
        max_seq_length: 校准样本最大长度
    """
    from transformers import AutoTokenizer
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    # 1. 准备校准数据
    print(f"[步骤 1/4] 准备校准数据 ({calibration_samples} 样本)")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 使用内置示例数据作为校准集
    data_path = os.path.join("data", "sample_dataset.json")
    if os.path.exists(data_path):
        import json
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        calibration_data = []
        for item in dataset[:calibration_samples]:
            text = f"{item['instruction']}\n{item['input']}"
            tokenized = tokenizer(text, return_tensors="pt", max_length=max_seq_length, truncation=True)
            calibration_data.append({"input_ids": tokenized["input_ids"]})
    else:
        print(f"[警告] 未找到校准数据 {data_path}，使用默认示例")
        calibration_data = [
            {"input_ids": tokenizer("测试文本用于模型量化校准", return_tensors="pt")["input_ids"]}
            for _ in range(8)
        ]

    # 2. 配置量化参数
    print(f"[步骤 2/4] 配置量化: bits={bits}, group_size={group_size}")
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
    )

    # 3. 加载模型并量化
    print(f"[步骤 3/4] 加载并量化模型: {model_path}")
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config,
        trust_remote_code=True,
    )

    print("[信息] 开始量化，可能需要 10-30 分钟...")
    model.quantize(calibration_data)

    # 4. 保存量化模型
    print(f"[步骤 4/4] 保存量化模型到: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_quantized(output_dir, use_safetensors=True)
    tokenizer.save_pretrained(output_dir)

    # 打印模型大小对比
    original_size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(model_path) for f in fs) / 1024**3
    quantized_size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(output_dir) for f in fs) / 1024**3

    print("\n" + "=" * 60)
    print("  量化完成")
    print("=" * 60)
    print(f"原始模型大小: {original_size:.2f} GB")
    print(f"量化模型大小: {quantized_size:.2f} GB")
    print(f"压缩比:       {original_size / quantized_size:.1f}x")
    print("=" * 60)
    print(f"\n下一步: vLLM 部署")
    print(f"  bash scripts/06_serve_vllm.sh {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="GPTQ 模型量化")
    parser.add_argument(
        "--model",
        type=str,
        default="merged_model/qwen25-7b-lora-merged",
        help="待量化的模型路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="quantized_model/qwen25-7b-gptq",
        help="量化后模型保存路径",
    )
    parser.add_argument("--bits", type=int, default=4, help="量化位数(4 或 8)")
    parser.add_argument("--group-size", type=int, default=128, help="量化组大小")
    parser.add_argument("--calibration-samples", type=int, default=128, help="校准样本数")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="校准样本最大长度")
    args = parser.parse_args()

    print("=" * 60)
    print("  GPTQ 模型量化")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"输出: {args.output}")
    print(f"参数: bits={args.bits}, group_size={args.group_size}")
    print("=" * 60)

    quantize_model(
        model_path=args.model,
        output_dir=args.output,
        bits=args.bits,
        group_size=args.group_size,
        calibration_samples=args.calibration_samples,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
