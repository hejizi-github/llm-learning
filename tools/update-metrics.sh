#!/usr/bin/env bash
# 用法: update-metrics.sh [--external] <session_id> <verdict> <score>
#
# 默认（无 --external）：把 agent 自评写入 self_score / self_verdict 字段。
# 带 --external：把外部评审结果写入 review_score / review_verdict 字段。
#
# 这样区分确保 review_score/review_verdict 只由外部评审写入，
# 不被 Agent 自评污染。
#
# test_count：如果 /tmp/pytest_result_<session>.txt 存在，从中解析测试数量。
#
# 全局去重：每次调用都对整个文件做一次去重（同 session_id 保留 test_count 最大的条）。

set -euo pipefail

# 解析 --external 标志
EXTERNAL=0
if [[ "${1:-}" == "--external" ]]; then
  EXTERNAL=1
  shift
fi

SESSION_ID="${1:-}"
VERDICT="${2:-}"
SCORE="${3:-}"

if [[ -z "$SESSION_ID" || -z "$VERDICT" || -z "$SCORE" ]]; then
  echo "用法: $0 [--external] <session_id> <verdict> <score>" >&2
  echo "例如: $0 20260419-032934 PASS 8.0" >&2
  echo "      $0 --external 20260419-032934 PASS 9.0" >&2
  exit 1
fi

METRICS_FILE="$(dirname "$0")/../.evolve/memory/session_metrics.jsonl"

if [[ ! -f "$METRICS_FILE" ]]; then
  echo "错误: 找不到 $METRICS_FILE" >&2
  exit 1
fi

if ! command -v jq &>/dev/null; then
  echo "错误: 需要 jq，请先安装" >&2
  exit 1
fi

# ── 全局去重：对整个文件，按 session_id 分组，保留 test_count 最大的条 ──
# 这会清理历史遗留的重复记录，不只是当前 session 的。
global_dedup() {
  local tmpfile
  tmpfile=$(mktemp)
  # jq: 以 session 为 key 分组，每组取 test_count 最大的条，再按原始顺序输出
  # 用 unique 保留每个 session 的唯一记录
  jq -rcs '
    group_by(.session) |
    map(sort_by(.test_count) | last) |
    sort_by(.session) |
    .[]
  ' "$METRICS_FILE" > "$tmpfile" 2>/dev/null || true

  # jq 失败时保留原文件
  if [[ -s "$tmpfile" ]]; then
    mv "$tmpfile" "$METRICS_FILE"
    echo "全局去重完成"
  else
    rm -f "$tmpfile"
    echo "警告: 全局去重 jq 失败，跳过" >&2
  fi
}

global_dedup

# ── 读取 pytest 结果文件（如果存在） ──
PYTEST_RESULT_FILE="/tmp/pytest_result_${SESSION_ID}.txt"
test_count=0
if [[ -f "$PYTEST_RESULT_FILE" ]]; then
  # 文件格式："61 passed in 16.49s" 或 "61 passed, 2 warnings in 16.49s"
  raw=$(cat "$PYTEST_RESULT_FILE" | tr -d '\n')
  if [[ "$raw" =~ ([0-9]+)\ passed ]]; then
    test_count="${BASH_REMATCH[1]}"
    echo "从 $PYTEST_RESULT_FILE 读取 test_count=$test_count"
  fi
fi

# ── 计算 commit_count ──
set +e
auto_commit_count=$(git -C "$(dirname "$METRICS_FILE")/../.." log --oneline 2>/dev/null \
  | grep "evolve(${SESSION_ID})" | grep -v "reflection" | wc -l | tr -d '[:space:]')
set -e
auto_commit_count=$(echo "${auto_commit_count:-0}" | tr -d '[:space:]')

# ── 检查 session 是否已存在 ──
existing=$(jq -rc --arg sid "$SESSION_ID" 'select(.session == $sid)' "$METRICS_FILE" \
  2>/dev/null || echo "")

if [[ -z "$existing" ]]; then
  # 不存在：新建记录
  today=$(date +%Y-%m-%d)
  if [[ "$EXTERNAL" -eq 1 ]]; then
    # 外部评审新建：review 字段填值，self 字段留 null
    new_entry=$(jq -cn \
      --arg sid "$SESSION_ID" \
      --arg date "$today" \
      --arg verdict "$VERDICT" \
      --argjson score "$SCORE" \
      --argjson cc "$auto_commit_count" \
      --argjson tc "$test_count" \
      '{session: $sid, date: $date, reverted: false, fix_rounds: 0,
        self_score: null, self_verdict: null,
        review_score: $score, review_verdict: $verdict,
        commit_count: $cc, test_count: $tc,
        assertion_total: 0, assertion_passed: 0, assertion_compliance: null,
        prompt_sha: "unknown"}')
  else
    # Agent 自评新建：self 字段填值，review 字段留 null
    new_entry=$(jq -cn \
      --arg sid "$SESSION_ID" \
      --arg date "$today" \
      --arg verdict "$VERDICT" \
      --argjson score "$SCORE" \
      --argjson cc "$auto_commit_count" \
      --argjson tc "$test_count" \
      '{session: $sid, date: $date, reverted: false, fix_rounds: 0,
        self_score: $score, self_verdict: $verdict,
        review_score: null, review_verdict: "PENDING",
        commit_count: $cc, test_count: $tc,
        assertion_total: 0, assertion_passed: 0, assertion_compliance: null,
        prompt_sha: "unknown"}')
  fi
  echo "$new_entry" >> "$METRICS_FILE"
  echo "新建 session $SESSION_ID 记录: external=$EXTERNAL verdict=$VERDICT score=$SCORE commit_count=$auto_commit_count test_count=$test_count"
else
  # 存在：更新对应字段
  tmpfile=$(mktemp)
  if [[ "$EXTERNAL" -eq 1 ]]; then
    # 外部评审：更新 review_score、review_verdict 和 commit_count/test_count
    while IFS= read -r line; do
      echo "$line" | jq -c \
        --arg sid "$SESSION_ID" \
        --arg verdict "$VERDICT" \
        --argjson score "$SCORE" \
        --argjson cc "$auto_commit_count" \
        --argjson tc "$test_count" \
        'if .session == $sid then
           .review_verdict = $verdict | .review_score = $score |
           .commit_count = $cc |
           (if $tc > 0 then .test_count = $tc else . end)
         else . end'
    done < "$METRICS_FILE" > "$tmpfile"
    echo "已更新 session $SESSION_ID 外部评审: review_verdict=$VERDICT review_score=$SCORE"
  else
    # Agent 自评：更新 self_score、self_verdict 和 commit_count/test_count
    while IFS= read -r line; do
      echo "$line" | jq -c \
        --arg sid "$SESSION_ID" \
        --arg verdict "$VERDICT" \
        --argjson score "$SCORE" \
        --argjson cc "$auto_commit_count" \
        --argjson tc "$test_count" \
        'if .session == $sid then
           .self_verdict = $verdict | .self_score = $score |
           .commit_count = $cc |
           (if $tc > 0 then .test_count = $tc else . end)
         else . end'
    done < "$METRICS_FILE" > "$tmpfile"
    echo "已更新 session $SESSION_ID 自评: self_verdict=$VERDICT self_score=$SCORE"
  fi
  mv "$tmpfile" "$METRICS_FILE"
fi

# ── 验证写入结果 ──
if [[ "$EXTERNAL" -eq 1 ]]; then
  result=$(jq -rc --arg sid "$SESSION_ID" 'select(.session == $sid) | .review_verdict' \
    "$METRICS_FILE" 2>/dev/null | tail -1 || echo "")
  if [[ "$result" != "$VERDICT" ]]; then
    echo "错误: 写入验证失败！期望 review_verdict=$VERDICT，实际读回 $result" >&2
    exit 1
  fi
  echo "验证通过: session $SESSION_ID review_verdict=$result"
else
  result=$(jq -rc --arg sid "$SESSION_ID" 'select(.session == $sid) | .self_verdict' \
    "$METRICS_FILE" 2>/dev/null | tail -1 || echo "")
  if [[ "$result" != "$VERDICT" ]]; then
    echo "错误: 写入验证失败！期望 self_verdict=$VERDICT，实际读回 $result" >&2
    exit 1
  fi
  echo "验证通过: session $SESSION_ID self_verdict=$result"
fi
