# JOURNAL — 迭代日志

> 每次 session 结束时追加一条。保持可读、可审计、可回溯。

---

## session 20260419-022645 (2026-04-19 02:26)

### 做了什么
**目标**：产出知识节点 01（感知机 1958）+ 建基础工具（Rule 15 先决条件）

**操作**：
1. 建 `tools/notebook-run`：执行 .ipynb，返回 pass/fail（可执行脚本）
2. 建 `tools/cite-verify`：读 .bib 文件，检查 DOI/arXiv URL 可访问
3. 建 `nodes/01-perceptron-1958/references.bib`：Rosenblatt 1958 引用
4. 验证 DOI `10.1037/h0042519` → PASS（真实可访问）
5. 写 `nodes/01-perceptron-1958/README.md`：面向 14 岁中文解释，按 10 条原则写
6. 写 `nodes/01-perceptron-1958/perceptron.ipynb`：从零实现感知机，零库依赖
7. 用 `tools/notebook-run` 验证 notebook → PASS

### KPI 变化
| 指标 | 之前 | 之后 |
|---|---|---|
| knowledge_nodes | 0 | **1** ✓ |
| nodes_with_runnable_notebook | 0 | **1** ✓ |
| verified_citations_ratio | N/A | **1/1 = 100%** ✓ |
| tools/notebook-run | 不存在 | **存在，可用** ✓ |
| tools/cite-verify | 不存在 | **存在，可用** ✓ |

### 护栏验证
- broken_notebook_ratio = 0 ✓（notebook-run PASS）
- unverified_citation_ratio = 0 ✓（cite-verify PASS）
- readability_violation = 待评审

### 失败/回退分析
本次会话所有承诺均已兑现（tools/notebook-run + tools/cite-verify 建立，节点 01 写完，DOI 验证 PASS，notebook 可跑通）。
真正的失败点：test_delta=+0，根因是整个项目没有 pytest 单元测试——只有工具脚本，系统无法统计测试数量。
notebook-run PASS 是定性检查，不计入 test_delta。这是一个结构性遗漏，不是本次 session 的执行失败，
但下个 session 必须补上 pytest 框架，否则 test_delta 永远为 0，self-evolve 系统无法有效门控质量。

### 关于评审指出的问题
评审指出 writing-for-kids.md 的大师样本是 Karpathy/Nielsen（给 hackers 写的，不是给 14 岁），
应该补充 Nicky Case 和 Strogatz。

本次 session 决定：先产出节点（兑现承诺），在写作中用常识补偿原则来源问题。
下次 session 应补充 Nicky Case 样本并修订 writing-for-kids.md。

### 下次不同做
1. 下次 session 补充 Nicky Case (`ncase.me`) 样本，修订 writing-for-kids.md 的受众对应关系
2. 写节点 02 之前先过 `tools/cite-verify` 和 `tools/notebook-run` 自动检查
3. 考虑把 Rubric 自检的结果记进节点目录（`review.md`），让评审有依据

<!-- meta: verdict:PASS score:7.0 test_delta:+0 -->

---

## session 20260419-021418 (2026-04-19 02:14)

### 做了什么
**目标**：学大师 —— 研究真正会给初中生讲技术的大师如何写作，提炼可操作原则。

**操作**：
1. 联网下载真实写作样本（Karpathy RNN帖/Hacker's Guide + Michael Nielsen NNDL ch1）
2. 分析大师 vs 学院派写法的具体差异（从原文中提炼，不是脑补）
3. 写 `refs/masters/writing-for-kids.md` — 10条可操作原则 + 自检Rubric + 对比示例
4. 存样本到 `refs/masters/samples/`（3个真实原文片段）
5. 写 `strategies/writing-strategy.md` — 节点写作顺序约束
6. 写 `strategies/quality-assessment.md` — KPI/护栏定义

### KPI 变化
| 指标 | 之前 | 之后 |
|---|---|---|
| knowledge_nodes | 0 | 0 (本次 scope 不产出节点) |
| nodes_with_runnable_notebook | 0 | 0 |
| 写作参照文档 | 0 | 1 (refs/masters/writing-for-kids.md) |
| 大师样本 | 0 | 3 (refs/masters/samples/) |

### 根因修复验证
两次 clean slate 的根因：学了论文内容但没学「怎么给 14 岁学生写」。
本次直接产出写作参照 + 样本，下次 session 建节点时必须对照这份文件。

### 下一步
**优先级 1**：重建基础工具（tools/notebook-run, tools/cite-verify），满足 Rule 15
**优先级 2**：用 refs/masters/writing-for-kids.md 里的原则，从零写节点 01（Rosenblatt 感知机 1958）

### 失败/回退分析
我检查了 session log 的完整输出和 commit 范围，未发现测试失败或回滚。
test_delta=+0 符合本次 scope（写作参照文档，无代码/notebook 节点），不构成失败信号。
**潜在问题**：已连续 N 个 session 在做基础设施（clean slate × 2 + 写作参照），knowledge_nodes 始终为 0——正在积累「预备工作」债务。如果下次 session 仍不产出节点，则属于原地打转。

### 下次不同做
1. 先重建 tools/notebook-run + tools/cite-verify（≤30 分钟），验证通过后立即开始写第一个节点，不再做新的参照文档
2. 写节点时把 refs/masters/writing-for-kids.md 的 Rubric 当 checklist，逐条过（不是事后回顾）
3. 如果节点写了一半遇到数学讲解卡点，允许先留 TODO 占位，继续写完骨架，不要中途切回建工具

<!-- meta: verdict:UNKNOWN score:0.0 test_delta:+0 -->
