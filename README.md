# LLM Learning — 机器学习发展史知识库

> 沿着历史脉络，理解每一个 AI 突破的**为什么**。

## 这是什么

一个**自我迭代的教学知识库**：以机器学习的真实发展历史为骨架，每个知识节点讲清楚一个关键突破——它出现的动机、核心原理、数学细节、以及它又留下了什么未解的问题。

**目标读者**：具备初中数学（基础代数）+ 会读基础 Python 的好奇学习者。

## 组织形式：时间线 + 依赖图

本知识库采用**时间线为主轴、依赖关系为辅的**组织方式：

```
1943 → 1958 → 1969 → 1986 → 1989 → 1997 → 2006 → 2012 → 2014 → 2017 → ...
 MP    感知机  局限   BP   卷积   LSTM   深度  AlexNet  GAN  Transformer
```

**选择时间线的理由**：每个概念的出现都是被前一个概念的局限"逼出来"的。时间线保留了这条因果链，让读者体会到"为什么会有这个"，而不只是"它是什么"。

## 知识节点列表

| # | 年份 | 节点 | notebook | 深度 |
|---|------|------|---------|------|
| 01 | 1958 | [Rosenblatt 感知机](docs/01-perceptron-1958.md) | [ipynb](notebooks/01-perceptron-1958.ipynb) | ⭐⭐⭐⭐⭐ |

_更多节点持续添加中..._

## 目录结构

```
llm-learning/
├── docs/           # 知识节点 markdown 文档
├── notebooks/      # 可运行 Jupyter notebooks（算法手撕）
├── refs/
│   ├── references.bib      # 所有引用的 BibTeX
│   └── citations.jsonl     # 引用溯源记录
├── tools/          # 自检工具
│   ├── notebook-run        # 批量执行 notebook，检测报错
│   ├── cite-verify         # 验证引用 DOI/arxiv 真实性
│   ├── md-link-check       # 检查 md 中链接有效性
│   └── depth-score         # 给知识节点打深度分
├── strategies/
│   └── quality-rubric.md   # 深度评分标准
└── README.md
```

## 自检命令

```bash
# 跑所有 notebooks
tools/notebook-run

# 验证所有引用
tools/cite-verify

# 检查 md 链接
tools/md-link-check

# 打深度分
tools/depth-score
```

## 质量门控

- `broken_notebook_ratio = 0`（notebook 全部可运行）
- `unverified_citation_ratio ≤ 0.05`（引用几乎全部可查证）
- `depth_score ≥ 3/5`（每个节点达到合格深度）

---

_本知识库由 Agent 自主迭代生成，使用反幻觉机制保证引用可验证。_
