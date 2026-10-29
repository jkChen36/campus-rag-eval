# -*- coding: utf-8 -*-
"""
step2_embed.py  ——  第 3 天用：把文字块变成向量，存成 vectors.npy

做什么：
  1. 读 chunks.json
  2. 调 embedding 接口，把每个片段转成一个数字数组（向量）
  3. 存成 vectors.npy，供检索时使用

为什么要这一步：
  电脑不懂文字，只懂数字。向量就是"把一段话翻译成一组数字"，
  意思相近的两段话，向量也相近（用余弦相似度算距离）。

怎么用：
  python step2_embed.py

怎么算成功：
  看到 "已保存 vectors.npy，形状 (N, 1024)"，N 等于片段数量
"""
import os
import json
import time

import numpy as np
from openai import OpenAI

# ---- 向量接口配置（用环境变量，不要把密钥写进代码！）----
EMB_API_KEY = os.environ.get("EMB_API_KEY", "")
EMB_BASE_URL = os.environ.get("EMB_BASE_URL", "https://api.siliconflow.cn/v1")
EMB_MODEL = os.environ.get("EMB_MODEL", "BAAI/bge-m3")

BATCH = 16  # 一次最多送 16 条，太多会被接口拒绝


def main():
    if not EMB_API_KEY:
        print("请先设置环境变量 EMB_API_KEY（见手册第 0 天）")
        return

    data = json.load(open("chunks.json", encoding="utf-8"))
    chunks = data["chunks"]
    texts = [c["text"] for c in chunks]
    print(f"共 {len(texts)} 个片段待向量化，模型 {EMB_MODEL}")

    client = OpenAI(api_key=EMB_API_KEY, base_url=EMB_BASE_URL)

    vectors = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        # 失败自动重试 3 次
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=EMB_MODEL, input=batch)
                vectors.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                print(f"  第 {i//BATCH+1} 批第 {attempt+1} 次失败：{e}")
                time.sleep(3)
        else:
            raise RuntimeError(f"第 {i//BATCH+1} 批连续失败，请检查密钥和额度")

        done = min(i + BATCH, len(texts))
        print(f"  进度 {done}/{len(texts)}")

    arr = np.array(vectors, dtype="float32")
    np.save("vectors.npy", arr)
    print("-" * 50)
    print(f"已保存 vectors.npy，形状 {arr.shape}")


if __name__ == "__main__":
    main()
