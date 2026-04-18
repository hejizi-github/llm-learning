# 质量评估标准

状态：活跃  
建立：2026-04-19（session 20260419-021418）

## KPI 定义

| 指标 | 定义 | 如何衡量 |
|---|---|---|
| knowledge_nodes | nodes/ 下的知识节点数量 | `find nodes/ -name README.md | wc -l` |
| nodes_with_runnable_notebook | nodes/ 下能零错误执行的 .ipynb 数量 | `find nodes/ -name "*.ipynb" -exec jupyter nbconvert --execute {} \;` 或 `tools/notebook-run` |
| verified_citations_ratio | 通过 DOI/arxiv 验证的引用 / 总引用数 | `tools/cite-verify` |
| depth_score | 节点平均深度分 (0-10) | `tools/depth-score` |

## 护栏定义

| 护栏 | 阈值 | 触发动作 |
|---|---|---|
| broken_notebook_ratio | 必须为 0 | 本次改动回滚 |
| unverified_citation_ratio | ≤ 0.05 | 本次改动回滚 |
| readability_violation | ≤ 0.10 | 本次改动回滚 |

## readability_violation 判定规则

一个知识节点的某段落违反以下任一条，该段落计为 1 个 violation：
1. 用技术词没有立刻解释
2. 类比来自成年人/职场场景（不适合 14 岁）
3. 公式出现在类比/直觉之前
4. 第一段从定义开始（不是从读者已知的现象）
5. 出现「显然」「容易知道」「不难看出」「其实很简单」

`readability_violation = violations / total_paragraphs`

## depth_score 判定规则（待建工具，暂用人工审核）

| 分数 | 含义 |
|---|---|
| 1-3 | 只有定义和公式，没有直觉、动机、历史背景 |
| 4-6 | 有直觉，有代码，但读者还是要靠"努力理解"才能懂 |
| 7-8 | 读者按顺序读下去自然就懂，类比贴切，代码能跑 |
| 9-10 | 读完会有「啊原来如此」的感觉，历史背景真实，局限讲清楚了 |

目标：所有节点 depth_score ≥ 7
