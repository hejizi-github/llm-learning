# Active Memory

> 当前知识库的"状态快照"。每次 session 开始必读，session 结束必更新。

## 知识库当前状态

**基础设施**：完成（目录骨架 + 4工具 + README + 策略文件）

**知识节点**：
| # | 文件 | notebook | depth | citations |
|---|------|----------|-------|-----------|
| 01 | docs/01-perceptron-1958.md | notebooks/01-perceptron-1958.ipynb | 5/5 | 4/4 verified |

**引用库**：refs/references.bib（4条），refs/citations.jsonl（4条全部已验证）

**时间线覆盖**：1958（感知机）

## 上次 session 的 learnings

- APA DOI 链接对 HEAD 请求返回 403（非真正失效），md-link-check 需 HEAD→GET fallback
- 书籍引用需 ISBN via Open Library；bib 解析器必须显式列出 `isbn` 字段
- depth-score 的中文+英文+LaTeX 三模式联合匹配，对中英混合 md 有效

## 下次 session 建议

**推荐下一步**：节点 02 — 1969 Minsky & Papert 的致命批评

理由：
1. 收尾感知机的故事（感知机 → 被证明局限 → AI寒冬），保持时间线连续
2. 数学复杂度适中（主要是几何不可分直觉 + 线性代数初步证明）
3. 历史叙事张力强，适合面向初中生的讲述风格

节点 02 内容提纲：
- 背景：1960年代的乐观与泡沫
- Minsky & Papert 的贡献：严格证明单层感知机局限
- XOR 不可分的几何直觉 + 形式化证明
- AI 寒冬：经费削减，神经网络研究进入低谷
- 衔接：为什么"加一层"能解决 XOR？这埋下了多层网络的种子
- notebook：可视化线性可分 vs 不可分的边界，展示为什么多层网络能解 XOR
- 引用：Minsky1969（已在 refs 中），可能还需要 Block1962（感知机收敛定理）
