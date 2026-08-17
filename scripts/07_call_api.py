"""
vLLM API 调用示例

演示如何调用部署好的 vLLM 服务进行文本摘要
支持: 同步调用、流式调用、批量调用

使用方法:
    # 先启动 vLLM 服务
    bash scripts/06_serve_vllm.sh

    # 再运行本脚本
    python scripts/07_call_api.py
"""

import argparse
import json
import time
import requests
from typing import List, Dict


# vLLM 服务默认地址
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL_NAME = "qwen25-7b-summarizer"


def check_service(base_url: str) -> bool:
    """检查 vLLM 服务是否可用"""
    try:
        resp = requests.get(f"{base_url.replace('/v1', '')}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def list_models(base_url: str) -> List[str]:
    """获取可用模型列表"""
    resp = requests.get(f"{base_url}/models", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return [m["id"] for m in data.get("data", [])]


def summarize_sync(
    text: str,
    base_url: str = DEFAULT_BASE_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.3,
    max_tokens: int = 256,
) -> str:
    """
    同步调用：生成单条文本的摘要

    Args:
        text: 待摘要的原文
        base_url: vLLM 服务地址
        model_name: 模型名称
        temperature: 采样温度（0.3 适合摘要任务，平衡多样性和稳定性）
        max_tokens: 最大生成 token 数

    Returns:
        生成的摘要文本
    """
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "你是一个文本摘要助手，请简洁准确地提取关键信息。"
            },
            {
                "role": "user",
                "content": f"请提取以下内容中的摘要信息\n\n{text}"
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def summarize_stream(
    text: str,
    base_url: str = DEFAULT_BASE_URL,
    model_name: str = DEFAULT_MODEL_NAME,
) -> str:
    """
    流式调用：逐 token 返回结果，适合交互式场景

    Args:
        text: 待摘要的原文

    Returns:
        完整的摘要文本
    """
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": f"请提取以下内容中的摘要信息\n\n{text}"}
        ],
        "stream": True,
        "temperature": 0.3,
        "max_tokens": 256,
    }

    full_text = ""
    print("[流式输出] ", end="", flush=True)
    with requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        stream=True,
        timeout=60,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    print(delta, end="", flush=True)
                    full_text += delta
                except json.JSONDecodeError:
                    continue
    print()
    return full_text


def summarize_batch(
    texts: List[str],
    base_url: str = DEFAULT_BASE_URL,
    model_name: str = DEFAULT_MODEL_NAME,
) -> List[str]:
    """
    批量调用：同时处理多条文本，利用 vLLM 的连续批处理优势

    Args:
        texts: 原文列表

    Returns:
        摘要列表
    """
    print(f"[批量调用] 共 {len(texts)} 条文本，并发处理...")
    results = []
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": "请提取以下内容中的摘要信息\n\n" + "\n---\n".join(texts)
            }
        ],
        "temperature": 0.3,
        "max_tokens": 512,
    }

    resp = requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()["choices"][0]["message"]["content"]
    return result


def demo():
    """完整演示：健康检查 → 模型列表 → 同步/流式/批量调用"""

    base_url = DEFAULT_BASE_URL
    model_name = DEFAULT_MODEL_NAME

    # 1. 健康检查
    print("=" * 60)
    print("  vLLM API 调用演示")
    print("=" * 60)

    print("\n[1] 健康检查...")
    if not check_service(base_url):
        print(f"[错误] 无法连接到 vLLM 服务 ({base_url})")
        print("请先启动服务: bash scripts/06_serve_vllm.sh")
        return
    print("    服务正常 ✓")

    # 2. 模型列表
    print("\n[2] 可用模型列表...")
    models = list_models(base_url)
    print(f"    {models}")

    # 3. 同步调用
    print("\n[3] 同步调用测试...")
    test_text = "健康饮食三原则:\n1. 早餐吃好:蛋白质+碳水化合物+维生素\n2. 午餐吃饱:主食+蔬菜+优质蛋白\n3. 晚餐吃少:以蔬菜和少量蛋白为主"
    print(f"    输入: {test_text[:50]}...")

    start = time.time()
    summary = summarize_sync(test_text, base_url, model_name)
    elapsed = time.time() - start
    print(f"    输出: {summary}")
    print(f"    耗时: {elapsed:.2f}s")

    # 4. 流式调用
    print("\n[4] 流式调用测试...")
    test_text_2 = "提高学习效率的三个技巧:\n1. 使用番茄工作法\n2. 建立思维导图\n3. 睡前复习"
    summarize_stream(test_text_2, base_url, model_name)

    # 5. 批量调用
    print("\n[5] 批量调用测试...")
    texts = [
        "冬季护肤五大步骤:温和洁面、保湿面霜、定期面膜、日常防晒、颈部防护",
        "Python入门要点:基础语法、数据结构、面向对象、项目实战",
    ]
    start = time.time()
    batch_result = summarize_batch(texts, base_url, model_name)
    elapsed = time.time() - start
    print(f"    批量结果: {batch_result[:80]}...")
    print(f"    耗时: {elapsed:.2f}s")

    print("\n" + "=" * 60)
    print("  演示完成")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="vLLM API 调用示例")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="vLLM 服务地址")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="模型名称")
    parser.add_argument("--text", type=str, help="待摘要的文本（不传则运行演示）")
    args = parser.parse_args()

    if args.text:
        result = summarize_sync(args.text, args.base_url, args.model)
        print(f"摘要: {result}")
    else:
        demo()


if __name__ == "__main__":
    main()
