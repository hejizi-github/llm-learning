# JOURNAL — 迭代日志

> 每次 session 结束时追加一条。保持可读、可审计、可回溯。

## 格式

```
## [YYYY-MM-DD HH:MM] session-id

### 做了什么
- ...

### KPI 快照
- knowledge_nodes: N
- nodes_with_runnable_notebook: N
- verified_citations_ratio: X%
- depth_score: X.X
- broken_notebook_ratio: X%
- unverified_citation_ratio: X%
- readability_violation: X%

### learnings（持久化的经验）
- ...

### 下次该做什么
- ...

### commit
- <commit-sha> <commit-message>
```

---

_（还没有 session 记录。首次迭代由 `self-evolve run` 触发。）_

---

## [2026-04-18 12:21] 20260418-122128

### 做了什么
- 搭建完整基础设施：目录骨架（docs/ notebooks/ refs/ tools/ strategies/）
- 实现 4 个验证工具：notebook-run、cite-verify（DOI+ISBN+arxiv）、md-link-check、depth-score
- 创建 strategies/quality-rubric.md（6维深度评分标准）
- 写 README（时间线+依赖图组织形式，含理由说明）
- 完成第一个知识节点：docs/01-perceptron-1958.md（Rosenblatt 感知机）
- 完成配套 notebook：notebooks/01-perceptron-1958.ipynb（手撕感知机，验证 AND/XOR）
- 写 refs/references.bib + refs/citations.jsonl（4条引用全部验证通过）

### KPI 快照
- knowledge_nodes: 1
- nodes_with_runnable_notebook: 1
- verified_citations_ratio: 100% (4/4)
- depth_score: 5/5 (节点01，6/6 rubric 维度全过)
- broken_notebook_ratio: 0%
- unverified_citation_ratio: 0%
- readability_violation: 未检测（工具待建）

### learnings（持久化的经验）
- APA DOI 链接对 HEAD 请求返回 403，md-link-check 需 HEAD→GET fallback
- 书籍引用需 ISBN via Open Library；bib 解析器必须显式列出 isbn 字段
- depth-score 的中文+英文+LaTeX 三模式联合匹配，对中英混合 md 有效

### 下次该做什么
- 节点 02：1969 Minsky & Papert 的致命批评（XOR 不可分几何证明 + AI 寒冬），收尾感知机故事
- 可选：节点 03（1986 反向传播），算法手撕含量高但数学复杂度更高

### commit
- （见 git log）
