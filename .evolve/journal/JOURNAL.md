# JOURNAL — 迭代日志

> 每次 session 结束时追加一条。保持可读、可审计、可回溯。

---

## 2026-04-18 | Session 20260418-233151

### 做了什么
从 CLEAN SLATE 引导整个知识库基础设施 + 第一个知识节点。

**工具链（tools/）**：
- `notebook-run` — 批量执行 notebooks，检测失败（修复 macOS bash 3.2 不支持 `mapfile`）
- `cite-verify` — 验证 references.bib 中每条引用有 doi/eprint/isbn 标识符
- `md-link-check` — 检查 docs/*.md 中内部相对链接（修复 macOS grep 不支持 `-P`）
- `depth-score` — 5 维度给知识节点打分（≥3/5 = PASS）

**策略（strategies/quality-assessment.md）**：深度评分标准、可读性标准、KPI 计算

**引用库（refs/）**：2 条已验证引用（Rosenblatt 1958 DOI + Minsky 1969 ISBN）

**节点 01**：
- `docs/01-perceptron-1958.md` — 面向 14 岁读者（故事→直觉→数学→局限→历史）
- `notebooks/01-perceptron-1958.ipynb` — 从零手撕感知机（AND 收敛 + XOR 失败）

### KPI

| 指标 | 变化 |
|------|------|
| knowledge_nodes | 0 → 1 |
| nodes_with_runnable_notebook | 0 → 1 |
| verified_citations_ratio | — → 1.000 ✓ |
| depth_score | — → 5/5 ✓ |
| broken_notebook_ratio | — → 0.000 ✓ |
| unverified_citation_ratio | — → 0.000 ✓ |

### 遇到的问题
1. Rosenblatt 1958 无 open access 版本（Unpaywall 确认）。DOI 验证成功，全文未能 fetch。文档只引用 DOI，不引用具体内容。
2. macOS bash 3.2 无 `mapfile` → 改 while+heredoc
3. macOS grep 无 `-P` → 改 python3 inline
4. Notebook JSON 中文引号未转义 → sed 修复

### 下次该做什么
- 节点 02：1969 Minsky-Papert XOR 局限 / AI 寒冬（bib 条目已就绪）
- 为 `cite-verify` 增加 curl DOI 存活性在线检查
- 在 `tests/` 加 Python 单元测试（测 Perceptron 类）
