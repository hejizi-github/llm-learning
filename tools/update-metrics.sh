#!/usr/bin/env bash
# 用法: update-metrics.sh <session_id> <verdict> <score>
# 例如: update-metrics.sh 20260419-032934 PASS 8.0
#
# 在 review 结束后调用，将 review 结果写回 session_metrics.jsonl。
# 只更新 review_verdict 和 review_score 字段，其他字段保留原值。
#
# 幂等性：对同一 session 多次调用会用新传入的 verdict/score 覆盖已有值（覆盖语义，不跳过）。

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

# 统计当前该 session_id 的条目数（防止重复污染）
set +o pipefail
count=$(grep "\"session\":\"${SESSION_ID}\"" "$METRICS_FILE" 2>/dev/null | wc -l | tr -d ' ')
set -o pipefail

if [[ "$count" -gt 1 ]]; then
  # 存在重复条目：去重——保留第一条（通常是较完整的那条），删除后续同 session_id 的行
  tmpfile=$(mktemp)
  seen=0
  while IFS= read -r line; do
    sid=$(echo "$line" | jq -r '.session // empty' 2>/dev/null || true)
    if [[ "$sid" == "$SESSION_ID" ]]; then
      if [[ "$seen" -eq 0 ]]; then
        echo "$line" >> "$tmpfile"
        seen=1
      fi
      # 跳过后续重复行
    else
      echo "$line" >> "$tmpfile"
    fi
  done < "$METRICS_FILE"
  mv "$tmpfile" "$METRICS_FILE"
  echo "警告: session $SESSION_ID 有 $count 条重复记录，已去重保留第一条"
fi

# 检查 session_id 是否存在
existing=$(grep "\"session\":\"${SESSION_ID}\"" "$METRICS_FILE" 2>/dev/null || echo "")

if [[ -z "$existing" ]]; then
  # 不存在该 session：追加一条新记录
  today=$(date +%Y-%m-%d)
  new_entry=$(jq -cn \
    --arg sid "$SESSION_ID" \
    --arg date "$today" \
    --arg verdict "$VERDICT" \
    --argjson score "$SCORE" \
    '{session: $sid, date: $date, reverted: false, fix_rounds: 0,
      review_score: $score, review_verdict: $verdict,
      commit_count: 0, test_count: 0,
      assertion_total: 0, assertion_passed: 0, assertion_compliance: null,
      prompt_sha: "unknown"}')
  echo "$new_entry" >> "$METRICS_FILE"
  echo "新建 session $SESSION_ID 记录: verdict=$VERDICT score=$SCORE"
else
  # 存在该 session：更新 review_verdict 和 review_score
  tmpfile=$(mktemp)
  while IFS= read -r line; do
    echo "$line" | jq -c --arg sid "$SESSION_ID" \
                         --arg verdict "$VERDICT" \
                         --argjson score "$SCORE" \
      'if .session == $sid then
         .review_verdict = $verdict | .review_score = $score
       else . end'
  done < "$METRICS_FILE" > "$tmpfile"
  mv "$tmpfile" "$METRICS_FILE"
  echo "已更新 session $SESSION_ID: verdict=$VERDICT score=$SCORE"
fi

# 立即验证写入结果
result=$(grep "\"session\":\"${SESSION_ID}\"" "$METRICS_FILE" 2>/dev/null | jq -r '.review_verdict' 2>/dev/null || echo "")
if [[ "$result" != "$VERDICT" ]]; then
  echo "错误: 写入验证失败！期望 $VERDICT，实际读回 $result" >&2
  exit 1
fi
echo "验证通过: session $SESSION_ID verdict=$result"
