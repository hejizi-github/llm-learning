#!/usr/bin/env bash
# 用法: update-metrics.sh <session_id> <verdict> <score>
# 例如: update-metrics.sh 20260419-032934 PASS 8.0
#
# 在 review 结束后调用，将 review 结果写回 session_metrics.jsonl。
# 只更新 review_verdict 和 review_score 字段，其他字段保留原值。
#
# 幂等性：对同一 session 多次调用会用新传入的 verdict/score 覆盖已有值（覆盖语义，不跳过）。
#
# 去重策略：如果同一 session 有多条记录，保留 test_count 最大的那条（而非信任顺序），
#           并打印审计日志。这避免了"错误数据恰好在前"导致正确数据被丢弃的问题。

set -euo pipefail

SESSION_ID="${1:-}"
VERDICT="${2:-}"
SCORE="${3:-}"

if [[ -z "$SESSION_ID" || -z "$VERDICT" || -z "$SCORE" ]]; then
  echo "用法: $0 <session_id> <verdict> <score>" >&2
  echo "例如: $0 20260419-032934 PASS 8.0" >&2
  exit 1
fi

METRICS_FILE="$(dirname "$0")/../.evolve/memory/session_metrics.jsonl"

if [[ ! -f "$METRICS_FILE" ]]; then
  echo "错误: 找不到 $METRICS_FILE" >&2
  exit 1
fi

# 检查 jq 是否可用
if ! command -v jq &>/dev/null; then
  echo "错误: 需要 jq，请先安装" >&2
  exit 1
fi

# 用 jq 匹配（不依赖 JSON 格式，兼容有空格/无空格两种写法）
# 统计当前该 session_id 的条目数
count=$(jq -rc --arg sid "$SESSION_ID" 'select(.session == $sid)' "$METRICS_FILE" \
  2>/dev/null | wc -l | tr -d ' ')

if [[ "$count" -gt 1 ]]; then
  # 存在重复条目：保留 test_count 最大的那条（避免因顺序假设保留了错误的0值行）
  best_line=$(jq -rc --arg sid "$SESSION_ID" 'select(.session == $sid)' "$METRICS_FILE" \
    | jq -s 'sort_by(.test_count) | last' -c 2>/dev/null || true)
  if [[ -z "$best_line" ]]; then
    # jq 失败时退化为第一条（保持向后兼容），并打印警告
    best_line=$(jq -rc --arg sid "$SESSION_ID" 'select(.session == $sid)' "$METRICS_FILE" \
      | head -1)
    echo "警告: jq sort 失败，退化为保留第一条" >&2
  fi
  # 审计日志：打印所有被评估的条目
  echo "去重: session $SESSION_ID 共 $count 条记录，保留 test_count=$(echo "$best_line" | jq -r '.test_count')"
  jq -rc --arg sid "$SESSION_ID" 'select(.session == $sid)' "$METRICS_FILE" \
    | while IFS= read -r entry; do
        tc=$(echo "$entry" | jq -r '.test_count' 2>/dev/null || echo "?")
        vc=$(echo "$entry" | jq -r '.review_verdict' 2>/dev/null || echo "?")
        echo "  候选: test_count=$tc verdict=$vc"
      done
  # 写入：用 jq 跳过所有同 session_id 行，在第一次遇到时插入 best_line
  tmpfile=$(mktemp)
  inserted=0
  while IFS= read -r line; do
    sid=$(echo "$line" | jq -r '.session // empty' 2>/dev/null || true)
    if [[ "$sid" == "$SESSION_ID" ]]; then
      if [[ "$inserted" -eq 0 ]]; then
        echo "$best_line" >> "$tmpfile"
        inserted=1
      fi
    else
      echo "$line" >> "$tmpfile"
    fi
  done < "$METRICS_FILE"
  mv "$tmpfile" "$METRICS_FILE"
fi

# 检查 session_id 是否存在（去重后）
existing=$(jq -rc --arg sid "$SESSION_ID" 'select(.session == $sid)' "$METRICS_FILE" \
  2>/dev/null || echo "")

if [[ -z "$existing" ]]; then
  # 不存在该 session：追加一条新记录
  today=$(date +%Y-%m-%d)
  # 用 git log 自动计算 commit_count（而非硬编码 0）
  # grep -c 在无匹配时 exit 1，用 set +e 防止 pipefail 中断
  set +e
  # "reflection" commits 不计入 commit_count（由 harness 自动生成，不代表 agent 工作量）
  auto_commit_count=$(git -C "$(dirname "$METRICS_FILE")/../.." log --oneline 2>/dev/null \
    | grep "evolve(${SESSION_ID})" | grep -v "reflection" | wc -l | tr -d '[:space:]')
  set -e
  # 确保是纯数字（去除换行等特殊字符）
  auto_commit_count=$(echo "${auto_commit_count:-0}" | tr -d '[:space:]')
  new_entry=$(jq -cn \
    --arg sid "$SESSION_ID" \
    --arg date "$today" \
    --arg verdict "$VERDICT" \
    --argjson score "$SCORE" \
    --argjson cc "$auto_commit_count" \
    '{session: $sid, date: $date, reverted: false, fix_rounds: 0,
      review_score: $score, review_verdict: $verdict,
      commit_count: $cc, test_count: 0,
      assertion_total: 0, assertion_passed: 0, assertion_compliance: null,
      prompt_sha: "unknown"}')
  echo "$new_entry" >> "$METRICS_FILE"
  echo "新建 session $SESSION_ID 记录: verdict=$VERDICT score=$SCORE commit_count=$auto_commit_count"
else
  # 存在该 session：更新 review_verdict、review_score 和 commit_count
  # 重新计算 commit_count（harness 预插入的是 0，这里覆写为真实值）
  set +e
  auto_commit_count=$(git -C "$(dirname "$METRICS_FILE")/../.." log --oneline 2>/dev/null \
    | grep "evolve(${SESSION_ID})" | grep -v "reflection" | wc -l | tr -d '[:space:]')
  set -e
  auto_commit_count=$(echo "${auto_commit_count:-0}" | tr -d '[:space:]')

  tmpfile=$(mktemp)
  while IFS= read -r line; do
    echo "$line" | jq -c --arg sid "$SESSION_ID" \
                         --arg verdict "$VERDICT" \
                         --argjson score "$SCORE" \
                         --argjson cc "$auto_commit_count" \
      'if .session == $sid then
         .review_verdict = $verdict | .review_score = $score | .commit_count = $cc
       else . end'
  done < "$METRICS_FILE" > "$tmpfile"
  mv "$tmpfile" "$METRICS_FILE"
  echo "已更新 session $SESSION_ID: verdict=$VERDICT score=$SCORE commit_count=$auto_commit_count"
fi

# 立即验证写入结果（用 jq 读回，不依赖 grep 格式）
result=$(jq -rc --arg sid "$SESSION_ID" 'select(.session == $sid) | .review_verdict' \
  "$METRICS_FILE" 2>/dev/null | tail -1 || echo "")
if [[ "$result" != "$VERDICT" ]]; then
  echo "错误: 写入验证失败！期望 $VERDICT，实际读回 $result" >&2
  exit 1
fi
echo "验证通过: session $SESSION_ID verdict=$result"
