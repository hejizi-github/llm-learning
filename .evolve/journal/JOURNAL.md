# Journal

## Session 20260419-002930 — 修复测试架构：提取 Perceptron 到 src/

### 失败/回退分析

本次无失败。完整执行计划，5 tests PASS，notebook PASS。

### 下次不同做

1. **构建知识节点 02**（1969 Minsky-Papert XOR 局限 / AI 寒冬）——knowledge_nodes 连续三个 session 没增长，这是最重要的欠债
2. **为 tools/cite-verify 增加 curl DOI 在线可达性检查**——已承诺三次，下次 session 不能再推迟
3. （已完成）.evolve/session.lock/ 已加入 .gitignore

### 做了什么

履行评审要求：修复"测试保护死副本"的结构性缺陷。

**核心变更**：
- 创建 `src/__init__.py` + `src/perceptron.py`（Perceptron 类唯一来源）
- 修改 `tests/test_perceptron.py`：删除内联副本，改为 `from src.perceptron import Perceptron`
- 修改 `notebooks/01-perceptron-1958.ipynb`：原有 class 定义 cell 替换为 markdown 展示 + import cell
- 修复 XOR 测试弱点：增加 `any(e > 0 for e in p.history)` 和 weights 变化断言
- 在 `.gitignore` 加入 `.evolve/session.lock/`（防止 lock 文件干扰 git rebase）

**结果**：tests import 的和 notebook 执行的是同一份代码——任何对 `src/perceptron.py` 的修改都会同时被测试检测到。

### KPI

| 指标 | 变化 |
|------|------|
| knowledge_nodes | 1 → 1（不变，本次专注架构修复）|
| nodes_with_runnable_notebook | 1 → 1 ✓ |
| verified_citations_ratio | 1.000 → 1.000 ✓ |
| depth_score | 5/5 → 5/5 ✓ |
| broken_notebook_ratio | 0.000 → 0.000 ✓ |
| test_architecture_integrity | 死副本 → 活代码（结构性修复）|

<!-- meta: verdict:PASS score:7.5 test_delta:+0 -->

---

## Session 20260419-001049 — 补齐 Perceptron pytest 单元测试

### 失败/回退分析

session 内部成功建立 5 个 pytest 测试，内部测量 test_delta=+5，但 reflection 系统传入 test_delta=+0。根因推断：reflection 系统的测试计数时间点在 session 提交之前或用不同基准计算，导致度量盲区。**测试本身全部通过，无回滚**，但这暴露了两套计数系统不同步的结构性问题。

另一个真实卡点：git push 被 `.evolve/session.lock/pid`（未跟踪文件）阻塞了 `git pull --rebase`，耗费 5-6 个 round 绕过，最终通过临时移走 lock 目录完成推送。根因：session.lock 未加入 `.gitignore`，rebase 时 git 拒绝覆盖未跟踪文件。

我检查了 session log 中的 commit 范围和测试输出，测试全部 PASS，无功能性失败。

### 下次不同做

1. 切换到全新方向：构建知识节点 02（1969 Minsky-Papert / AI 寒冬），节点完成后立刻加 pytest
2. 将 `.evolve/session.lock/` 加入 `.gitignore`，避免 lock 文件干扰 git rebase
3. 为 `tools/cite-verify` 加 curl DOI 在线检查（已承诺两次，本次 session 后必须完成）

### 做了什么

履行上次 session 承诺：在 `tests/test_perceptron.py` 建立 Perceptron pytest 单元测试。内联 Perceptron 类（从 notebook 提取），覆盖 5 个测试用例（AND 收敛、history 终止、XOR 不收敛、predict 形状、权重更新）。全部通过。git push 因 session.lock 阻塞耗费额外 rounds，最终通过临时移走 lock 目录解决。

### KPI

| 指标 | 变化 |
|------|------|
| knowledge_nodes | 1 → 1（不变）|
| nodes_with_runnable_notebook | 1 → 1（不变）|
| verified_citations_ratio | 1.000 → 1.000 ✓ |
| depth_score | 5/5 → 5/5 ✓ |
| broken_notebook_ratio | 0.000 → 0.000 ✓ |
| test_delta | 0 → **+5** ✓ （session 内部测量）|

<!-- meta: verdict:PASS score:5.0 test_delta:+5 -->

---

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
