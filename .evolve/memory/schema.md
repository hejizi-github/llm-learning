# session_metrics.jsonl 字段定义

## commit_count

**定义**：该 session 中由 agent 生成的非 reflection commit 数量。

**计算方式**（`update-metrics.sh` 自动计算）：
```bash
git log --oneline | grep "evolve($SESSION_ID)" | grep -v "reflection" | wc -l
```

**排除**：harness 自动生成的 `reflection` commits（commit message 包含 "reflection" 字样）。  
**包含**：agent 主动产出的 "agent work"、"fix"、功能提交等。

**示例**：session 064159 有 3 个 commit：
- `evolve(064159): reflection` → 不计（排除）
- `evolve(064159): agent work (auto-committed)` → 计 1
- `evolve(064159): 修复metrics重复行+自评分规范+docstring白话化` → 计 1
- **commit_count = 2** ✓

---

## test_count

测试执行完成后的 `pytest --tb=no -q` 通过数量。

## review_score / review_verdict

由外部评审 agent 给出的评分（0–10）和判决（PASS / NEEDS_IMPROVEMENT）。
