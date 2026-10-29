# -*- coding: utf-8 -*-
"""
step3_ask.py  ——  第 3 天用：检索 + 生成，搭通整条链路

做什么：
  1. 把问题也变成向量
  2. 在所有片段里找最相似的 K 个（这就是"检索"，RAG 里的 R）
  3. 把这 K 个片段塞进提示词，让大模型根据资料回答（这就是"生成"，RAG 里的 G）

怎么用：
  python step3_ask.py "大三下学期什么时候开始选课"

怎么算成功：
  能看到检索到的片段编号 + 一个像样的回答；
  问一个资料里没有的问题，它应该说"资料中没有相关信息"
"""
import os
import sys
import json

import numpy as np
from openai import OpenAI

# ---- 接口配置 ----
EMB_API_KEY = os.environ.get("EMB_API_KEY", "")
EMB_BASE_URL = os.environ.get("EMB_BASE_URL", "https://api.siliconflow.cn/v1")
EMB_MODEL = os.environ.get("EMB_MODEL", "BAAI/bge-m3")

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

emb_client = OpenAI(api_key=EMB_API_KEY, base_url=EMB_BASE_URL)
llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

# ---- 加载索引 ----
_index = json.load(open("chunks.json", encoding="utf-8"))
chunks = [c["text"] for c in _index["chunks"]]
sources = [c["source"] for c in _index["chunks"]]
V = np.load("vectors.npy")


def embed(texts):
    """把一批文字变成向量"""
    out = []
    for i in range(0, len(texts), 16):
        resp = emb_client.embeddings.create(model=EMB_MODEL, input=texts[i:i + 16])
        out.extend([d.embedding for d in resp.data])
    return np.array(out, dtype="float32")


def retrieve(query, k=5):
    """
    检索：算问题向量和所有片段向量的余弦相似度，取最像的 k 个
    返回：(片段下标列表, 相似度列表)
    """
    qv = embed([query])[0]
    norm_v = np.linalg.norm(V, axis=1)
    norm_q = np.linalg.norm(qv)
    sims = V @ qv / (norm_v * norm_q + 1e-9)
    idx = np.argsort(-sims)[:k]
    return [int(i) for i in idx], [float(sims[i]) for i in idx]


PROMPT_V1 = """你是学校教务咨询助手。只能依据【资料】回答，不允许使用资料以外的知识。

规则：
1. 资料里有答案 → 简洁回答，并在句末标注引用编号，例如 [3][7]
2. 资料里没有 → 只回答："资料中没有相关信息，我无法确定。" 不要猜测，不要编造
3. 资料里有多份文件冲突 → 以最新年份的文件为准，并说明你依据的是哪一份

【资料】
{context}

【问题】
{question}

请回答："""

# v2：针对"时效冲突""对抗诱导"两类弱项强化（基线实验显示这两类准确率仅 46.7% / 40.0%）
PROMPT_V2 = PROMPT_V1 + """
4. 【核对前提】如果问题里含有某个说法或前提（例如"听说……""是不是……吧"），
   你必须先拿资料核对，不能顺着用户的前提回答：
   - 资料不支持该说法 → 先明确说"资料中没有这一规定"，再给出资料里的真实规定
   - 资料中的数字/条件与问题的说法不一致 → 直接指出不一致，并给出正确数字
5. 【冲突取新】检索到多份不同年份文件时，必须比较后作答：
   明确写出"以××××年文件为准"，并说明旧版的规定是什么、为什么不再适用
6. 【不确定就拒答】资料只支持部分结论时，只回答有依据的部分，
   剩余部分明确说"资料未涉及"，严禁补全或推测
"""

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v1")
PROMPT = PROMPTS.get(PROMPT_VERSION, PROMPT_V1)


def ask(question, k=5, use_rerank=False, rerank_pool=10, return_detail=False):
    """
    完整问答
    use_rerank=True 时：先取 rerank_pool 个候选，再让大模型挑最相关的 k 个
    """
    pool = rerank_pool if use_rerank else k
    idx, sims = retrieve(question, pool)

    if use_rerank:
        idx = llm_rerank(question, idx, k)

    # 是否在片段前带上来源文件名（含年份）
    # 时效冲突类题的失败根因就是模型看不到年份，无法判断哪份文件最新
    include_source = os.environ.get("INCLUDE_SOURCE", "0") == "1"
    if include_source:
        context = "\n\n".join(f"[{i}] 来源：{sources[i]}\n{chunks[i]}" for i in idx)
    else:
        context = "\n\n".join(f"[{i}] {chunks[i]}" for i in idx)
    resp = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}],
        temperature=0,
    )
    answer = resp.choices[0].message.content

    if return_detail:
        return answer, idx, [sources[i] for i in idx]
    return answer, idx


RERANK_PROMPT = """下面有若干资料片段和一个问题。请按"对回答该问题的有用程度"从高到低排序，
只输出最有用的 {k} 个片段的编号，用逗号分隔，例如：7,3,12
不要输出任何其他内容。

【问题】{question}

【资料片段】
{context}
"""


def llm_rerank(question, idx, k):
    """用大模型做重排（第 9 天的对照实验用）"""
    context = "\n\n".join(f"[{i}] {chunks[i][:200]}" for i in idx)
    resp = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": RERANK_PROMPT.format(question=question, context=context, k=k)}],
        temperature=0,
    )
    nums = []
    for part in resp.choices[0].message.content.replace("，", ",").split(","):
        part = part.strip()
        if part.isdigit() and int(part) in idx:
            nums.append(int(part))
    for i in idx:  # 兜底：模型漏了就补上
        if len(nums) >= k:
            break
        if i not in nums:
            nums.append(i)
    return nums[:k]


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "请介绍这个文档的主要内容"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    answer, idx = ask(q, k=k)
    print("=" * 60)
    print("问题：", q)
    print("-" * 60)
    for i in idx:
        print(f"[片段{i}] 来源={sources[i]}  {chunks[i][:120]}...")
    print("-" * 60)
    print("回答：", answer)
