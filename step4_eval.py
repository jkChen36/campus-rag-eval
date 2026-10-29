# -*- coding: utf-8 -*-
"""
step4_eval.py  ——  第 8~11 天用：跑评测，算出指标，产出结果表

做什么：
  1. 读 eval_set.csv（你自己造的 100 条评测集）
  2. 每条问题跑一遍问答，记录：检索到了什么、回答了什么
  3. 算 4 个指标：召回命中率、答案准确率、幻觉率、拒答正确率
  4. 输出到 result_<实验名>.csv + result_<实验名>.json

怎么用：
  python step4_eval.py exp1        # 实验名随便起，用于区分不同配置
  注意：换配置（切分策略 / Top-K / rerank）前，要重新跑 step1 + step2

怎么算成功：
  跑完看到 4 个指标的数字，并且 result_*.csv 里能看到每条的判定明细
"""
import os
import re
import sys
import csv
import json
import time

import numpy as np
from openai import OpenAI

import step3_ask as RAG  # 复用检索和问答

# ---- 本实验的配置 ----
# 手动跑：python step4_eval.py E2 semantic 5 0
# 批量跑：交给 step5_run_all.py，它会自动重跑 step1/step2 并逐个调用本脚本
_name = sys.argv[1] if len(sys.argv) > 1 else "exp1"
_strategy = sys.argv[2] if len(sys.argv) > 2 else "semantic"
_top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
_rerank = sys.argv[4] == "1" if len(sys.argv) > 4 else False

CONFIG = {
    "split_strategy": _strategy,   # fixed / section / semantic
    "top_k": _top_k,
    "use_rerank": _rerank,
    "note": "由命令行参数指定；换策略前 step5 会自动重跑 step1 + step2",
}

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
judge_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


# ---------------- 判分：让大模型当裁判 ----------------
JUDGE_PROMPT = """你是评分裁判。请判断【模型回答】相对【标准答案】是否正确。

【问题】{question}
【标准答案】{golden}
【模型回答】{answer}

判定规则：
1. 核心事实与标准答案一致（说法不同但意思相同也算）→ correct=1
2. 标准答案是"应当拒答"，模型明确表示"没有相关信息/无法确定" → correct=1, hallucination=0
3. 模型编造了标准答案之外的事实，或把没说的事说成有 → correct=0, hallucination=1
4. 模型该回答却拒答了 → correct=0, hallucination=0

只输出一行 JSON，不要任何其他文字：
{{"correct": 0或1, "hallucination": 0或1, "reason": "不超过30字的原因"}}"""


def judge(question, golden, answer):
    for _ in range(3):
        try:
            r = judge_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    question=question, golden=golden, answer=answer)}],
                temperature=0,
            )
            txt = r.choices[0].message.content.strip()
            txt = txt.replace("```json", "").replace("```", "").strip()
            start, end = txt.find("{"), txt.rfind("}") + 1
            obj = json.loads(txt[start:end])
            return int(obj.get("correct", 0)), int(obj.get("hallucination", 0)), obj.get("reason", "")
        except Exception:
            time.sleep(2)
    return 0, 0, "判分失败"


def _norm(s):
    """去掉所有空白（含换行），避免原文被切分换行后匹配不上"""
    return re.sub(r"\s+", "", s or "")


def hit_evidence(evidence, idx):
    """
    召回命中判定：标准答案的依据原文，是否出现在检索到的片段里
    三级判定，从严格到宽松：
      1. 去空白后整句子串命中
      2. 关键词 60% 以上命中
      3. 前 12 字子串命中
    """
    if not evidence:
        return None          # 拒答题没有依据，不参与召回统计（返回 None）
    ev = _norm(evidence)
    pool = _norm(" ".join(RAG.chunks[i] for i in idx))
    if not ev:
        return None
    if ev in pool:
        return 1
    keys = [w for w in re.split(r"[，。；、（）()\s]", evidence) if len(w) >= 2]
    if keys:
        hit = sum(1 for w in keys if _norm(w) in pool)
        if hit >= max(1, int(len(keys) * 0.6)):
            return 1
    if ev[:12] in pool:
        return 1
    return 0


def main():
    name = _name
    if not os.path.exists("eval_set.csv"):
        print("缺少 eval_set.csv，请先按手册第 4~7 天造评测集")
        return

    # 安全检查：chunks.json 的策略必须和 CONFIG 一致，否则跑的是错的索引
    try:
        cur = json.load(open("chunks.json", encoding="utf-8")).get("strategy")
        if cur != CONFIG["split_strategy"]:
            print("=" * 60)
            print(f"✗ 配置不一致！chunks.json 当前是 '{cur}'，但本实验要跑 '{CONFIG['split_strategy']}'")
            print(f"  请先执行：python step1_load.py {CONFIG['split_strategy']}  然后  python step2_embed.py")
            print("  或直接用 python step5_run_all.py，它会自动处理")
            return
    except FileNotFoundError:
        print("缺少 chunks.json，请先跑 step1_load.py")
        return

    rows = list(csv.DictReader(open("eval_set.csv", encoding="utf-8")))
    rows = [r for r in rows if r.get("question", "").strip()]
    print(f"评测集 {len(rows)} 条，配置：{CONFIG}")

    out = []
    for n, r in enumerate(rows, 1):
        q = r["question"].strip()
        t0 = time.time()
        try:
            answer, idx = RAG.ask(
                q,
                k=CONFIG["top_k"],
                use_rerank=CONFIG["use_rerank"],
            )
        except Exception as e:
            print(f"  [{n}] 调用失败：{e}")
            continue
        latency = round(time.time() - t0, 2)

        recall_hit = hit_evidence(r.get("golden_evidence", ""), idx)
        correct, halluc, reason = judge(q, r.get("golden_answer", ""), answer)

        out.append({
            "id": r.get("id", n),
            "type": r.get("type", ""),
            "question": q,
            "golden_answer": r.get("golden_answer", ""),
            "model_answer": answer,
            "recall_hit": recall_hit,
            "correct": correct,
            "hallucination": halluc,
            "reason": reason,
            "latency_s": latency,
            "retrieved_ids": "|".join(str(i) for i in idx),
        })
        print(f"  [{n}/{len(rows)}] 命中={recall_hit} 正确={correct} 幻觉={halluc} ({latency}s)")

    # ---------------- 汇总指标 ----------------
    total = len(out)
    if total == 0:
        print("没有有效结果")
        return
    # 召回命中率只在"有依据"的题上算（拒答题无依据，计入会把指标拉低）
    valid = [o for o in out if o["recall_hit"] is not None]
    recall = (sum(o["recall_hit"] for o in valid) / len(valid)) if valid else None
    acc = sum(o["correct"] for o in out) / total
    halluc_rate = sum(o["hallucination"] for o in out) / total
    refuse_rows = [o for o in out if o["type"] == "拒答"]
    refuse_acc = (sum(o["correct"] for o in refuse_rows) / len(refuse_rows)) if refuse_rows else None
    adv_rows = [o for o in out if o["type"] == "对抗诱导"]
    adv_halluc = (sum(o["hallucination"] for o in adv_rows) / len(adv_rows)) if adv_rows else None
    latency = sum(o["latency_s"] for o in out) / total

    # 分题型统计（报告里最有价值的一张表）
    by_type = {}
    for o in out:
        t = o["type"] or "未分类"
        d = by_type.setdefault(t, {"n": 0, "acc": 0, "halluc": 0, "recall_n": 0, "recall_hit": 0})
        d["n"] += 1
        d["acc"] += o["correct"]
        d["halluc"] += o["hallucination"]
        if o["recall_hit"] is not None:
            d["recall_n"] += 1
            d["recall_hit"] += o["recall_hit"]
    type_stat = {}
    for t, d in by_type.items():
        type_stat[t] = {
            "题数": d["n"],
            "准确率": round(d["acc"] / d["n"], 4),
            "幻觉率": round(d["halluc"] / d["n"], 4),
            "召回命中率": round(d["recall_hit"] / d["recall_n"], 4) if d["recall_n"] else None,
        }

    summary = {
        "实验名": name,
        "配置": CONFIG,
        "样本数": total,
        "参与召回统计题数": len(valid),
        "召回命中率": round(recall, 4) if recall is not None else None,
        "答案准确率": round(acc, 4),
        "幻觉率": round(halluc_rate, 4),
        "拒答正确率": round(refuse_acc, 4) if refuse_acc is not None else None,
        "对抗诱导幻觉率": round(adv_halluc, 4) if adv_halluc is not None else None,
        "平均时延s": round(latency, 2),
        "分题型": type_stat,
    }

    with open(f"result_{name}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    with open(f"result_{name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    for k, v in summary.items():
        if k == "分题型":
            print("分题型：")
            for t, d in v.items():
                print(f"   {t:<8} 题数{d['题数']:>3}  准确率{d['准确率']}  幻觉率{d['幻觉率']}  召回{d['召回命中率']}")
        else:
            print(f"{k}: {v}")
    print(f"明细已写入 result_{name}.csv，汇总已写入 result_{name}.json")


if __name__ == "__main__":
    main()
