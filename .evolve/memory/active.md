# Active Memory

> 当前知识库的"状态快照"。每次 session 开始必读，session 结束必更新。

## 知识库当前状态

**基础设施**：完成（目录骨架 + 4工具 + README + 策略文件）

**知识节点**：
| # | 文件 | notebook | depth | citations |
|---|------|----------|-------|-----------|
| 01 | docs/01-perceptron-1958.md | notebooks/01-perceptron-1958.ipynb | 5/5 | 4/4 verified |
| 02 | docs/02-minsky-papert-1969.md | notebooks/02-minsky-papert-1969.ipynb | 5/5 | 3/3 verified（复用 refs 中已有条目）|

**引用库**：refs/references.bib（4条），refs/citations.jsonl（4条全部已验证）

**时间线覆盖**：1958（感知机）→ 1969（XOR证明 + AI寒冬）

## 上次 session 的 learnings

- APA DOI 链接对 HEAD 请求返回 403（非真正失效），md-link-check 需 HEAD→GET fallback
- 书籍引用需 ISBN via Open Library；bib 解析器必须显式列出 `isbn` 字段
- depth-score 的中文+英文+LaTeX 三模式联合匹配，对中英混合 md 有效
- notebook 中不能在 JSON 内容里用未转义的 ASCII 双引号——应用 Python json.dump 生成 ipynb 而不是手写
- Government reports（如 Lighthill 1973）没有 DOI，应在文本中引用但不加入 references.bib

## 下次 session 建议

**推荐下一步**：节点 03 — 1986 反向传播（Rumelhart, Hinton, Williams）

理由：
1. 延续时间线（1969寒冬 → 1986复苏），直接接着节点 02 的衔接段
2. rumelhart1986 已在 refs 中验证，可直接使用
3. 反向传播的算法手撕含量高（链式法则 + 梯度下降），notebook 会很有教学价值

节点 03 内容提纲：
- 背景：寒冬 17 年，为什么 1986 是转折点
- 核心想法：把"错误信号"从输出层往回传
- 数学自包含：链式法则（中学微积分入门），梯度下降的直觉
- notebook：从零手撕 2 层网络的前向传播 + 反向传播，训练 XOR（上一章的遗留问题！）
- 局限：局部最优、梯度消失（为下一章深度学习埋种子）
- 引用：Rumelhart1986（已验证），可能需要 Werbos1974（最初提出 BP 思想，在其论文中）

**PENDING 提案**：`.evolve/proposals/sub-agent-evaluation.md`
- 用 LLM 子 Agent 评估内容质量（响应用户 DIRECTIVE 20260418-123509）
- 等用户审批后实施
