# Active Memory

> 当前知识库的"状态快照"。每次 session 开始必读，session 结束必更新。

## 知识库当前状态

**基础设施**：完成（目录骨架 + 5工具 + README + 策略文件）

**工具列表**：
- `tools/notebook-run` — 跑 notebook 验证
- `tools/cite-verify` — DOI/ISBN/arxiv 验证
- `tools/md-link-check` — md 链接检查
- `tools/depth-score` — 深度评分
- `tools/claude-advisor` — 外部 Claude 多角度分析（新增 20260418-125113）

**知识节点**：
| # | 文件 | notebook | depth | citations |
|---|------|----------|-------|-----------|
| 01 | docs/01-perceptron-1958.md | notebooks/01-perceptron-1958.ipynb | 5/5 | 4/4 verified |
| 02 | docs/02-minsky-papert-1969.md | notebooks/02-minsky-papert-1969.ipynb | 5/5 | 3/3 verified |

**引用库**：refs/references.bib（4条），refs/citations.jsonl（4条全部已验证）

**时间线覆盖**：1958（感知机）→ 1969（XOR证明 + AI寒冬）

**已修复**：docs/01 + docs/02 中 ISBN 格式错误（`978-0-262-63-070-2` → `978-0-262-63070-2`）

## 上次 session 的 learnings

- `claude -p --model haiku` 是最简调用，`--bare` 会跳过 OAuth keychain 不可用
- `--allowedTools ""` 空字符串会被 CLI 报错，应直接省略
- 外部 Claude 视角（claude-advisor）自发指出：没有测试框架时继续添加节点会引发技术债务滚雪球——这是 Agent 自身确认偏差中忽视的点
- claude-advisor 可进一步泛化：管道 md 内容进去做读者可读性评审

## 下次 session 建议

**第一优先（不可再推迟）**：创建 tests/ + conftest.py + pytest 用例（≥4个）
- test_delta=+0 已连续 3 次警告
- 外部顾问也独立得出相同结论：在测试框架缺失的情况下继续扩节点存在可靠性风险

**第二优先**：节点 03 — 1986 反向传播（Rumelhart, Hinton, Williams）
- 时间线接续（1969寒冬 → 1986复苏）
- rumelhart1986 已在 refs 中验证，可直接使用
- 提纲：寒冬17年背景 / 链式法则自包含讲解 / 手撕2层网络BP训练XOR / 局限（局部最优+梯度消失）

**PENDING 提案**：`.evolve/proposals/sub-agent-evaluation.md`
- 用 LLM 子 Agent 评估内容质量（响应用户 DIRECTIVE 20260418-123509）
- 等用户审批后实施
