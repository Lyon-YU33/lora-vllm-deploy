"""
模型下载脚本

从 HuggingFace 或 ModelScope 下载 Qwen2.5-7B-Instruct 基座模型
解决原项目硬编码本地路径问题

使用方法:
    # 下载到默认目录
    python scripts/02_download_model.py

    # 指定保存目录
    python scripts/02_download_model.py --output /root/autodl-tmp/models/Qwen2.5-7B-Instruct

    # 使用 ModelScope（国内更快）
    python scripts/02_download_model.py --source modelscope
"""

import argparse
import os
from pathlib import Path


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_DIR = "models/Qwen2.5-7B-Instruct"


def download_from_huggingface(model_name: str, output_dir: str):
    """从 HuggingFace 下载模型"""
    from huggingface_hub import snapshot_download

    print(f"[HuggingFace] 下载 {model_name} -> {output_dir}")
    snapshot_download(
        repo_id=model_name,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[完成] 模型已保存到: {output_dir}")


def download_from_modelscope(model_name: str, output_dir: str):
    """从 ModelScope 下载模型（国内更快）"""
    from modelscope.hub.snapshot_download import snapshot_download

    print(f"[ModelScope] 下载 {model_name} -> {output_dir}")
    snapshot_download(
        model_id=model_name,
        local_dir=output_dir,
    )
    print(f"[完成] 模型已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="下载基座模型")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="模型名称 (HuggingFace 或 ModelScope ID)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="本地保存目录",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["huggingface", "modelscope"],
        default="modelscope",
        help="下载源 (modelscope 国内更快)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  下载基座模型")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"输出: {args.output}")
    print(f"源:   {args.source}")
    print("=" * 60)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.source == "huggingface":
        download_from_huggingface(args.model, args.output)
    else:
        download_from_modelscope(args.model, args.output)

    print(f"\n下一步: 修改 configs/qwen25_lora.yaml 中的 model_name_or_path 为: {args.output}")
    print(f"        修改 configs/merge_lora.yaml 中的 model_name_or_path 为: {args.output}")


if __name__ == "__main__":
    main()
