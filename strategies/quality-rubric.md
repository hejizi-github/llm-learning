# 知识节点质量评分标准 (Quality Rubric)

> 每个知识节点 md 文档必须满足以下 6 个维度，才算"合格深度"。

## 6 维评分标准

| # | 维度 | 检查要点 | 工具 |
|---|------|----------|------|
| 1 | **背景故事** | 说清楚这个概念出现的时代背景、前人的困境（至少2处相关词/日期） | depth-score |
| 2 | **原理讲解** | 用初中生能懂的语言解释核心机制（至少1处"如何/步骤/工作"相关描述） | depth-score |
| 3 | **数学自包含** | 涉及的公式/数学概念必须有对初中生的解释（至少2处公式或数学词汇） | depth-score |
| 4 | **notebook链接** | 必须链接到可运行的 .ipynb 文件 | depth-score + notebook-run |
| 5 | **局限与衔接** | 分析当前方法的边界，指向下一个突破（至少2处"局限/然而/下一"相关词） | depth-score |
| 6 | **引用溯源** | 所有论文/事实必须有可验证的外部链接或 doi/arxiv | depth-score + cite-verify |

## 评分换算

- 6/6 → ⭐⭐⭐⭐⭐ (5/5)
- 5/6 → ⭐⭐⭐⭐ (4/5)
- 4/6 → ⭐⭐⭐ (3/5)
- 3/6 → ⭐⭐ (2/5)
- 1-2/6 → ⭐ (1/5)

## 门控规则

- `depth_score < 3` 的节点**不得 commit** 到 main
- `broken_notebook_ratio > 0` 阻断提交
- `unverified_citation_ratio > 0.05` 阻断提交

## 使用方法

```bash
# 对所有节点打分
tools/depth-score docs/

# 对单个节点
tools/depth-score docs/01-perceptron-1958.md

# 跑 notebook
tools/notebook-run notebooks/

# 验证引用
tools/cite-verify refs/references.bib

# 检查链接
tools/md-link-check docs/
```
