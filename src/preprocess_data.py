"""
数据预处理工具

功能:
    1. 将 CSV/TXT 等原始数据转换为指令微调格式 (instruction/input/output)
    2. 检查数据集格式是否合法
    3. 数据集统计分析

使用方法:
    # 转换 CSV 数据
    python src/preprocess_data.py --input raw.csv --output data/train.json

    # 检查数据集
    python src/preprocess_data.py --check data/sample_dataset.json

    # 统计数据集
    python src/preprocess_data.py --stats data/sample_dataset.json
"""

import argparse
import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any


# 默认指令（文本摘要任务）
DEFAULT_INSTRUCTION = "请提取以下内容中的摘要信息"


def convert_csv_to_json(
    csv_path: str,
    output_path: str,
    text_col: str = "text",
    summary_col: str = "summary",
    instruction: str = DEFAULT_INSTRUCTION,
) -> int:
    """
    将 CSV 文件转换为指令微调 JSON 格式

    Args:
        csv_path: CSV 文件路径
        output_path: 输出 JSON 路径
        text_col: 原文所在列名
        summary_col: 摘要所在列名
        instruction: 固定指令

    Returns:
        转换后的样本数量
    """
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(text_col, "").strip()
            summary = row.get(summary_col, "").strip()
            if not text or not summary:
                continue
            samples.append({
                "instruction": instruction,
                "input": text,
                "output": summary,
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[完成] 转换 {len(samples)} 条样本 -> {output_path}")
    return len(samples)


def check_dataset(data_path: str) -> bool:
    """
    检查数据集格式是否合法

    Args:
        data_path: JSON 数据集路径

    Returns:
        是否合法
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("[错误] 数据集必须是 JSON 数组")
        return False

    required_fields = {"instruction", "input", "output"}
    empty_count = 0
    for i, sample in enumerate(data):
        if not isinstance(sample, dict):
            print(f"[错误] 第 {i} 条样本不是字典")
            return False
        missing = required_fields - set(sample.keys())
        if missing:
            print(f"[错误] 第 {i} 条样本缺少字段: {missing}")
            return False
        for field in required_fields:
            if not sample[field].strip():
                empty_count += 1

    print(f"[通过] 共 {len(data)} 条样本，{empty_count} 个空字段")
    return True


def stats_dataset(data_path: str) -> Dict[str, Any]:
    """
    统计数据集信息

    Args:
        data_path: JSON 数据集路径

    Returns:
        统计信息字典
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    input_lens = [len(s["input"]) for s in data]
    output_lens = [len(s["output"]) for s in data]

    stats = {
        "样本总数": len(data),
        "input 平均长度": sum(input_lens) / len(input_lens),
        "input 最大长度": max(input_lens),
        "input 最小长度": min(input_lens),
        "output 平均长度": sum(output_lens) / len(output_lens),
        "output 最大长度": max(output_lens),
        "output 最小长度": min(output_lens),
        "指令种类": len(set(s["instruction"] for s in data)),
    }

    print("=" * 50)
    print(f"数据集统计: {data_path}")
    print("=" * 50)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="数据预处理工具")
    parser.add_argument("--input", type=str, help="输入文件路径(CSV)")
    parser.add_argument("--output", type=str, help="输出文件路径(JSON)")
    parser.add_argument("--check", type=str, help="检查数据集格式")
    parser.add_argument("--stats", type=str, help="统计数据集")
    parser.add_argument("--text-col", type=str, default="text", help="CSV 文本列名")
    parser.add_argument("--summary-col", type=str, default="summary", help="CSV 摘要列名")
    args = parser.parse_args()

    if args.check:
        check_dataset(args.check)
    elif args.stats:
        stats_dataset(args.stats)
    elif args.input and args.output:
        convert_csv_to_json(args.input, args.output, args.text_col, args.summary_col)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
