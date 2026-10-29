# -*- coding: utf-8 -*-
"""
step6_report.py  —— 把实验结果自动拼成一份可交付的评测报告

怎么用：
  python step6_report.py

产出：
  评测报告.md   含实验配置表、指标对比表、分题型表、自动挑出的 badcase
  （这份 md 直接就是简历里"输出完整归因报告"的实物）
"""
import os
import json
import glob
import csv

ORDER = ["E1_fixed_top5", "E2_semantic_top5", "E3_semantic_top3", "E4_semantic_top10",
         "E5_semantic_top5_rerank", "E6_section_top5", "E7_semantic_top10_rerank"]
TYPE_ORDER = ["直接检索", "多跳推理", "拒答", "时效冲突", "对抗诱导"]


def pct(v):
    return f"{v*100:.1f}%" if isinstance(v, (int, float)) else "—"


def load():
    data = {}
    for p in glob.glob("result_*.json"):
        name = os.path.basename(p)[len("result_"):-len(".json")]
        try:
            data[name] = json.load(open(p, encoding="utf-8"))
        except Exception:
            pass
    return data


def load_detail(name):
    p = f"result_{name}.csv"
    if not os.path.exists(p):
        return []
    return list(csv.DictReader(open(p, encoding="utf-8")))


def main():
    data = load()
    if not data:
        print("没有找到 result_*.json，请先跑 step5_run_all.py")
        return

    names = [n for n in ORDER if n in data] + [n for n in data if n not in ORDER]
    base = next((n for n in names if n.startswith("E2")), names[0])
    best = max(names, key=lambda n: (data[n].get("答案准确率") or 0))

    L = []
    L.append("# 校园教务 RAG 问答系统 —— 效果评测报告\n")
    L.append(f"- 评测集：100 条（直接检索 30 / 多跳推理 20 / 拒答 20 / 时效冲突 15 / 对抗诱导 15）")
    L.append(f"- 资料：安阳师范学院教务制度文件 50 份，切分后 {_chunk_count()} 个文本片段")
    L.append(f"- 向量模型：BAAI/bge-m3（1024 维）　生成模型：deepseek-chat　判分：LLM-as-Judge")
    L.append(f"- 基线：{base}　最佳：{best}\n")

    # ---- 1 指标对比 ----
    L.append("## 一、实验配置与总体指标\n")
    L.append("| 实验 | 切分策略 | Top-K | 重排 | 召回命中率 | 答案准确率 | 幻觉率 | 拒答正确率 | 平均时延 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for n in names:
        c = data[n].get("配置", {})
        s = data[n]
        L.append(f"| {n} | {c.get('split_strategy','-')} | {c.get('top_k','-')} | "
                 f"{'是' if c.get('use_rerank') else '否'} | {pct(s.get('召回命中率'))} | "
                 f"{pct(s.get('答案准确率'))} | {pct(s.get('幻觉率'))} | "
                 f"{pct(s.get('拒答正确率'))} | {s.get('平均时延s','-')}s |")

    b0, b1 = data[base], data[best]
    if b0.get("答案准确率") and b1.get("答案准确率"):
        L.append(f"\n**结论**：相对基线，最佳配置答案准确率 "
                 f"{pct(b0.get('答案准确率'))} → {pct(b1.get('答案准确率'))}，"
                 f"幻觉率 {pct(b0.get('幻觉率'))} → {pct(b1.get('幻觉率'))}。\n")

    # ---- 2 分题型 ----
    L.append("\n## 二、分题型表现（最佳配置 vs 基线）\n")
    L.append("| 题型 | 题数 | 基线准确率 | 最佳准确率 | 基线幻觉率 | 最佳幻觉率 | 最佳召回 |")
    L.append("|---|---|---|---|---|---|---|")
    t0 = b0.get("分题型", {})
    t1 = b1.get("分题型", {})
    for t in TYPE_ORDER:
        if t not in t0 and t not in t1:
            continue
        a, c = t0.get(t, {}), t1.get(t, {})
        L.append(f"| {t} | {c.get('题数', a.get('题数','-'))} | {pct(a.get('准确率'))} | "
                 f"{pct(c.get('准确率'))} | {pct(a.get('幻觉率'))} | {pct(c.get('幻觉率'))} | "
                 f"{pct(c.get('召回命中率'))} |")

    # ---- 3 归因 ----
    L.append("\n## 三、瓶颈归因\n")
    L.append("判据：召回命中率高但准确率低 → 问题在生成/Prompt；召回低 → 问题在切分/检索。\n")
    for n in names:
        s = data[n]
        r, a = s.get("召回命中率"), s.get("答案准确率")
        if not isinstance(r, (int, float)) or not isinstance(a, (int, float)):
            continue
        gap = r - a
        if gap > 0.15:
            judge = "召回明显高于准确率 → 检索能找到，但生成环节没用好（Prompt / 上下文组织问题）"
        elif r < 0.7:
            judge = "召回偏低 → 切分或检索是瓶颈（换策略 / 加大 Top-K / 加重排）"
        else:
            judge = "召回与准确率接近 → 检索是主要瓶颈，提升召回即可带动准确率"
        L.append(f"- **{n}**：召回 {pct(r)}，准确率 {pct(a)}，差值 {gap*100:+.1f} 个百分点。{judge}")

    # ---- 4 badcase ----
    detail = load_detail(best)
    bads = [d for d in detail if d.get("correct") == "0" or d.get("hallucination") == "1"]
    L.append(f"\n## 四、典型 badcase（取自 {best}，共 {len(bads)} 条失败）\n")
    if bads:
        L.append("| 题型 | 问题 | 标准答案 | 模型回答 | 召回 | 幻觉 | 归因 |")
        L.append("|---|---|---|---|---|---|---|")
        for d in bads[:12]:
            ans = (d.get("model_answer") or "").replace("\n", " ")[:60]
            gold = (d.get("golden_answer") or "").replace("\n", " ")[:50]
            q = (d.get("question") or "")[:40]
            rh = d.get("recall_hit")
            if rh in (None, "", "None"):
                cause = "无依据（拒答题）→ 判定模型是否错误作答"
            elif rh == "0":
                cause = "召回失败 → 切分/检索问题"
            else:
                cause = "召回成功但答错 → 生成/Prompt 问题"
            L.append(f"| {d.get('type','')} | {q} | {gold} | {ans} | {rh} | {d.get('hallucination')} | {cause} |")
    else:
        L.append("（无失败样本）")

    # ---- 5 优化建议 ----
    L.append("\n## 五、优化建议\n")
    weakest = None
    if b1.get("分题型"):
        cand = [(t, v["准确率"]) for t, v in b1["分题型"].items() if isinstance(v.get("准确率"), (int, float))]
        if cand:
            weakest = min(cand, key=lambda x: x[1])
    if weakest:
        L.append(f"- 最弱题型是「{weakest[0]}」（准确率 {pct(weakest[1])}），优先针对它优化")
    L.append("- 召回仍是瓶颈时：优先换语义切分 + 加大 Top-K + 引入 rerank")
    L.append("- 召回已高但准确率低时：改 Prompt（明确「只依据资料」、要求给引用编号、冲突取最新版本）")
    L.append("- 幻觉集中在对旧政策的误用：在 Prompt 中加入「多份文件冲突时以最新年份为准」")
    L.append("- 对抗诱导题的改进：在 Prompt 中增加「用户说法与资料冲突时，必须指出并纠正」")

    out = "评测报告.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"已生成 {out}（{len(L)} 行）")
    print(f"基线={base}  最佳={best}")
    if b0.get("答案准确率") and b1.get("答案准确率"):
        print(f"准确率 {pct(b0['答案准确率'])} → {pct(b1['答案准确率'])}")


def _chunk_count():
    try:
        return len(json.load(open("chunks.json", encoding="utf-8"))["chunks"])
    except Exception:
        return "?"


if __name__ == "__main__":
    main()
