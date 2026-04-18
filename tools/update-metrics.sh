#!/usr/bin/env bash
# 用法: update-metrics.sh <session_id> <verdict> <score>
# 例如: update-metrics.sh 20260419-032934 PASS 8.0
#
# 在 review 结束后调用，将 review 结果写回 session_metrics.jsonl。
# 只更新 review_verdict 和 review_score 字段，其他字段保留原值。

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

# 用 jq 更新匹配 session 的记录，其他行原样输出
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
