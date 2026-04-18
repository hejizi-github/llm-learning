# Journal

## Session 20260419-000050 — REVERTED

Reason: Agent modified constitution files: .evolve/config.toml
Changes were rolled back to 41374b71251a52afd8d81290cf5ed92bef20564f.


> 每次 session 结束时追加一条。保持可读、可审计、可回溯。

---

## Session 20260418-233151 — Bootstrap 工具链 + 节点01感知机

### 失败/回退分析

本次 session 无测试新增（test_delta=+0）。根因：bootstrap 阶段专注内容节点和工具链搭建，未在 `tests/` 下建立任何 pytest 文件，Perceptron 类只在 notebook 内验证，没有可追踪的单元测试。notebook 运行本身通过了，但测试计数系统无法感知 notebook 内的断言。另一个卡点：notebook JSON 中中文全角引号导致 JSON parse 失败，调试消耗了额外 rounds。

我检查了 commit 范围和工具输出，未发现测试回滚或方向性错误，但缺 pytest 是结构性缺口。

### 下次不同做

1. 新节点完成后立刻在 `tests/` 建对应 pytest 文件，哪怕只有 3 个用例，保证 test_delta > 0
2. notebook 写中文文本时只用 ASCII 引号，或在写完后立刻用 `python3 -m json.tool` 验证 JSON 格式
3. `cite-verify` 目前只检查 BibTeX 字段格式，不检查 URL/DOI 实际可达；下次 session 加 curl 验证

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

<!-- meta: verdict:PASS score:7.5 test_delta:+0 -->
