# Quality Assessment Rubric

## 深度评分（5 维度，每维 0/1，合格线 3/5）

| 维度 | 标准 | 检查方式 |
|------|------|---------|
| verified_citation | 文中至少有一条有 DOI/ISBN/arxiv 的引用 | `tools/cite-verify` |
| linked_notebook | 文中链接了可运行的 notebook | `tools/md-link-check` |
| has_math | 至少有一个数学公式（LaTeX 内联或块）| 正则 `$...$` |
| has_intuition | 有类比/故事/直觉讲解 | 关键词检测 |
| has_history | 有历史背景（年份/前后节点链接）| 正则 |

## 可读性标准（面向 14 岁读者）

1. **每段前先类比**：新概念必须先有直觉/故事，再给公式
2. **术语自解释**：第一次出现的英文术语必须给中文解释
3. **数学步骤显式**：公式推导每一步都写出来，不跳步
4. **测试问题**：每节末可加"想一想"让读者自检

## 引用验证标准

- 所有引用必须在 `refs/references.bib` 中有对应条目
- 每条 bib 条目必须有 `doi=` 或 `eprint=` 或 `isbn=` 字段
- `refs/citations.jsonl` 记录每条引用的验证状态和验证日期
- 不可访问的论文：可引用（保留 DOI），但不得在文中写"论文说了 X"除非有摘要来源

## Notebook 质量标准

1. `jupyter nbconvert --execute` 零错误
2. 核心算法从零实现（不得直接用 sklearn 等黑盒）
3. 有可视化（图表展示核心概念）
4. 展示"成功"和"失败"两种情况（知道局限性）

## KPI 计算

```
knowledge_nodes = len(docs/*.md)
nodes_with_runnable_notebook = 跑通的 notebook 数
verified_citations_ratio = 有 doi/isbn/eprint 的引用数 / 总引用数
depth_score = PASS 节点数 / 总节点数 (PASS = ≥3/5)
broken_notebook_ratio = 跑不通的 notebook / 总 notebook (必须为 0)
unverified_citation_ratio = 无标识符引用 / 总引用 (必须 ≤ 0.05)
readability_violation = 违反可读性标准的节点 / 总节点 (必须 ≤ 0.1)
```
