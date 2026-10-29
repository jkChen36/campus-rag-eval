# -*- coding: utf-8 -*-
"""
step5_run_all.py  —— 一键跑完 7 组对比实验

为什么要有这个脚本：
  换切分策略时必须重跑 step1 + step2，只改 Top-K / rerank 才不用。
  手跑很容易忘，一忘就是"拿旧的向量跑新配置"，数据全错。
  本脚本自动判断、自动重跑、自动收集结果。

怎么用：
  python step5_run_all.py              # 跑全部 7 组（约 20~40 分钟）
  python step5_run_all.py E1 E5        # 只跑指定几组
  python step5_run_all.py --force      # 已有的也重跑

跑完得到：
  result_<实验名>.csv / .json  每组明细
  实验对比表.csv                所有组汇总，直接贴报告
"""
import os
import sys
import json
import time
import subprocess

PY = sys.executable

# (实验名, 切分策略, Top-K, 是否重排)
EXPERIMENTS = [
    ("E1_fixed_top5",          "fixed",    5,  False, "v1"),
    ("E2_semantic_top5",       "semantic", 5,  False, "v1"),   # 基线
    ("E3_semantic_top3",       "semantic", 3,  False, "v1"),
    ("E4_semantic_top10",      "semantic", 10, False, "v1"),
    ("E5_semantic_top5_rerank","semantic", 5,  True,  "v1"),
    ("E6_section_top5",        "section",  5,  False, "v1"),
    ("E7_semantic_top10_rerank","semantic",10, True,  "v1"),
    # 下面两组换 Prompt v2：专门打"时效冲突""对抗诱导"两个弱项
    ("E8_semantic_top5_promptv2",  "semantic", 5,  False, "v2"),
    ("E9_semantic_top10_rerank_promptv2", "semantic", 10, True, "v2"),
    # E10：在 E9 基础上给片段加来源文件名（含年份），专打时效冲突
    ("E10_E9_plus_source", "semantic", 10, True, "v2"),
]

# 需要额外环境变量的实验
EXTRA_ENV = {
    "E10_E9_plus_source": {"INCLUDE_SOURCE": "1"},
}


def current_strategy():
    if not os.path.exists("chunks.json"):
        return None
    try:
        return json.load(open("chunks.json", encoding="utf-8")).get("strategy")
    except Exception:
        return None


def run(cmd, prompt_version=None, extra_env=None):
    env = os.environ.copy()
    if prompt_version:
        env["PROMPT_VERSION"] = prompt_version
    if extra_env:
        env.update(extra_env)
    tag = []
    if prompt_version and prompt_version != "v1":
        tag.append(f"PROMPT_VERSION={prompt_version}")
    for k, v in (extra_env or {}).items():
        tag.append(f"{k}={v}")
    print(f"    $ {' '.join(cmd)}" + (f"   [{' '.join(tag)}]" if tag else ""))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"命令失败：{' '.join(cmd)}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    todo = [e for e in EXPERIMENTS if (not args or e[0].split("_")[0] in args or e[0] in args)]
    if not todo:
        print("没有匹配的实验，可选：", [e[0] for e in EXPERIMENTS])
        return

    print(f"本次要跑 {len(todo)} 组：{[e[0] for e in todo]}")
    print("=" * 64)

    for name, strategy, topk, rerank, pv in todo:
        out_json = f"result_{name}.json"
        if os.path.exists(out_json) and not force:
            print(f"[跳过] {name} 已有结果（加 --force 可重跑）")
            continue

        t0 = time.time()
        print(f"\n【{name}】策略={strategy} Top-K={topk} rerank={rerank} prompt={pv}")

        # 关键：策略不一致就重建索引
        if current_strategy() != strategy:
            print(f"  索引策略为 {current_strategy()}，需要重建为 {strategy}")
            run([PY, "step1_load.py", strategy])
            run([PY, "step2_embed.py"])
        else:
            print(f"  索引已是 {strategy}，无需重建")

        run([PY, "step4_eval.py", name, strategy, str(topk), "1" if rerank else "0"], pv, EXTRA_ENV.get(name))
        print(f"  完成，用时 {time.time()-t0:.0f}s")

    # ---------- 汇总对比 ----------
    rows = []
    for name, strategy, topk, rerank, pv in EXPERIMENTS:
        p = f"result_{name}.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        rows.append({
            "实验名": name,
            "切分策略": strategy,
            "Top-K": topk,
            "重排": "是" if rerank else "否",
            "Prompt": pv,
            "召回命中率": d.get("召回命中率"),
            "答案准确率": d.get("答案准确率"),
            "幻觉率": d.get("幻觉率"),
            "拒答正确率": d.get("拒答正确率"),
            "对抗诱导幻觉率": d.get("对抗诱导幻觉率"),
            "平均时延s": d.get("平均时延s"),
        })

    if not rows:
        print("\n还没有任何结果")
        return

    import csv
    with open("实验对比表.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 64)
    print("实验对比（已写入 实验对比表.csv）")
    hdr = f"{'实验名':<32}{'策略':<10}{'K':>3}{'重排':>5}{'P':>4}{'召回':>8}{'准确率':>8}{'幻觉':>8}"
    print(hdr)
    print("-" * 78)
    def pct(v):
        return f"{v*100:.1f}%" if isinstance(v, (int, float)) else "-"
    for r in rows:
        print(f"{r['实验名']:<32}{r['切分策略']:<10}{r['Top-K']:>3}{r['重排']:>5}{r['Prompt']:>4}"
              f"{pct(r['召回命中率']):>8}{pct(r['答案准确率']):>8}{pct(r['幻觉率']):>8}")

    best_acc = max(rows, key=lambda r: (r["答案准确率"] or 0))
    base = next((r for r in rows if r["实验名"].startswith("E2")), rows[0])
    print("-" * 64)
    print(f"基线（E2）准确率 {pct(base['答案准确率'])}  →  最佳（{best_acc['实验名']}）{pct(best_acc['答案准确率'])}")
    print("下一步：python step6_report.py 生成评测报告")


if __name__ == "__main__":
    main()
