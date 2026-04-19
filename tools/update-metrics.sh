#!/usr/bin/env bash
# 用法: update-metrics.sh [--external] [--test-count N] <session_id> <verdict> <score>
#
# 默认（无 --external）：把 agent 自评写入 self_score / self_verdict 字段。
# 带 --external：把外部评审结果写入 review_score / review_verdict 字段。
#
# 这样区分确保 review_score/review_verdict 只由外部评审写入，
# 不被 Agent 自评污染。
#
# test_count：
#   - 优先使用 --test-count N 显式参数（调用方运行 pytest 后传入）
#   - fallback：如果 /tmp/pytest_result_<session>.txt 存在，从中解析
#   - 两者都没有：打印 warning，记录 0（不静默）
#
# 全局去重：每次调用都对整个文件做一次去重，同 session_id 的多条记录按数据完整性优先级合并。
# 合并规则：review_score 非 null 的记录优先级最高，null 字段从低优先级记录填入；test_count 取 max。
# commit_count：refresh_commit_counts() 总是用 git log 重算，取 max(git_count, existing_count)。

set -euo pipefail

# 解析标志（顺序无关）
EXTERNAL=0
EXPLICIT_TEST_COUNT=""

while [[ $# -gt 0 ]]; do
  case "${1}" in
    --external)
      EXTERNAL=1
      shift
      ;;
    --test-count)
      EXPLICIT_TEST_COUNT="${2:-}"
      if [[ -n "$EXPLICIT_TEST_COUNT" && ! "$EXPLICIT_TEST_COUNT" =~ ^[0-9]+$ ]]; then
        echo "错误: --test-count 必须是非负整数，收到: '$EXPLICIT_TEST_COUNT'" >&2
        exit 1
      fi
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

SESSION_ID="${1:-}"
VERDICT="${2:-}"
SCORE="${3:-}"

if [[ -z "$SESSION_ID" || -z "$VERDICT" || -z "$SCORE" ]]; then
  echo "用法: $0 [--external] [--test-count N] <session_id> <verdict> <score>" >&2
  echo "例如: $0 --test-count 79 20260419-032934 PASS 8.0" >&2
  echo "      $0 --external --test-count 79 20260419-032934 PASS 9.0" >&2
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

# ── 全局去重：对整个文件，按 session_id 分组，合并同 session 的多条记录 ──
# 合并规则（P0 修复）：
#   1. 按"数据优先级"排序：review_score 非 null +10，self_score 非 null +5，test_count 加值
#   2. 以最高优先级记录为基础，从低优先级记录中只填入 null 字段（非 null 不覆盖）
#   3. test_count 额外取 max；commit_count 由 update 路径的 auto_commit_count 权威覆写
# 这样确保真实外部评审数据（review_score）不会被空占位记录覆盖。
global_dedup() {
  local tmpfile
  tmpfile=$(mktemp)
  jq -rcs '
    group_by(.session) |
    map(
      map(
        . + {_priority: (
          (if .review_score != null then 10 else 0 end) +
          (if .self_score != null then 5 else 0 end) +
          (.test_count // 0)
        )}
      ) |
      sort_by(._priority) | reverse |
      reduce .[1:][] as $other (
        .[0];
        . as $acc |
        reduce ($other | to_entries[]) as $entry (
          $acc;
          if $entry.key == "_priority" then .
          elif $entry.key == "test_count" and ($entry.value // 0) > ($acc.test_count // 0) then
            .test_count = $entry.value
          elif $entry.value != null and $acc[$entry.key] == null then
            .[$entry.key] = $entry.value
          else . end
        )
      ) |
      del(._priority)
    ) |
    sort_by(.session) |
    .[]
  ' "$METRICS_FILE" > "$tmpfile" 2>/dev/null || true

  if [[ -s "$tmpfile" ]]; then
    mv "$tmpfile" "$METRICS_FILE"
    echo "全局去重完成"
  else
    rm -f "$tmpfile"
    echo "警告: 全局去重 jq 失败，跳过" >&2
  fi
}

global_dedup

# ── 刷新 commit_count：总是用 git log 重算，取 max(git_count, existing_count) ──
# max 策略：不降低历史已知正确的高值，也允许修正"非零但偏低"的错误值。
refresh_commit_counts() {
  local repo_dir tmpfile
  repo_dir="$(cd "$(dirname "$METRICS_FILE")/../.." && pwd)"
  tmpfile=$(mktemp)
  while IFS= read -r line; do
    local sid existing_count
    sid=$(printf '%s' "$line" | jq -r '.session // ""')
    existing_count=$(printf '%s' "$line" | jq -r '.commit_count // 0')
    if [[ -n "$sid" && "$sid" != "null" ]]; then
      local count max_count
      # grep 无匹配时 exit code=1，需要 set +e 防止 pipefail 崩溃脚本
      set +e
      count=$(git -C "$repo_dir" log --oneline 2>/dev/null \
        | grep "evolve(${sid})" | wc -l | tr -d '[:space:]')
      set -e
      # wc -l 输出数字（最小 "0"），count 永远非空，所以 :-0 不会触发
      # 真正的防御：若命令替换返回非数字（极端情况），兜底为 0
      [[ "$count" =~ ^[0-9]+$ ]] || count=0
      # max(git_count, existing_count)：不降低历史值，但也无法修正偏高的错误值
      if [[ "$count" -gt "$existing_count" ]]; then
        max_count="$count"
      else
        max_count="$existing_count"
      fi
      printf '%s' "$line" | jq -c --argjson cc "${max_count}" '.commit_count = $cc'
    else
      printf '%s\n' "$line"
    fi
  done < "$METRICS_FILE" > "$tmpfile"
  if [[ -s "$tmpfile" ]]; then
    mv "$tmpfile" "$METRICS_FILE"
  else
    rm -f "$tmpfile"
  fi
}

refresh_commit_counts

# ── 确定 test_count ──
test_count=0
if [[ -n "$EXPLICIT_TEST_COUNT" ]]; then
  # 优先：调用方显式传入（最可靠，避免临时文件消失问题）
  test_count="$EXPLICIT_TEST_COUNT"
  echo "使用显式 --test-count=$test_count"
else
  # fallback：读取临时文件
  PYTEST_RESULT_FILE="/tmp/pytest_result_${SESSION_ID}.txt"
  if [[ -f "$PYTEST_RESULT_FILE" ]]; then
    raw=$(cat "$PYTEST_RESULT_FILE" | tr -d '\n')
    if [[ "$raw" =~ ([0-9]+)\ passed ]]; then
      test_count="${BASH_REMATCH[1]}"
      echo "从 $PYTEST_RESULT_FILE 读取 test_count=$test_count"
    fi
  else
    echo "warning: --test-count 未传入且 $PYTEST_RESULT_FILE 不存在，test_count 将记录为 0。" >&2
    echo "建议：先运行 pytest，再调用 update-metrics.sh --test-count <N>" >&2
  fi
fi

# ── 计算 commit_count ──
set +e
auto_commit_count=$(git -C "$(dirname "$METRICS_FILE")/../.." log --oneline 2>/dev/null \
  | grep "evolve(${SESSION_ID})" | wc -l | tr -d '[:space:]')
set -e
auto_commit_count=$(echo "${auto_commit_count:-0}" | tr -d '[:space:]')

# 外部评审时：commit_count=0 可能意味着传入了错误的 session_id
if [[ "$EXTERNAL" -eq 1 && "$auto_commit_count" -eq 0 ]]; then
  echo "警告: --external 模式下 session '$SESSION_ID' 的 commit_count=0，请确认 session_id 正确。" >&2
fi

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

# ── 验证写入结果（同时校验 verdict 字符串和 score 数值） ──
if [[ "$EXTERNAL" -eq 1 ]]; then
  result_verdict=$(jq -rc --arg sid "$SESSION_ID" 'select(.session == $sid) | .review_verdict' \
    "$METRICS_FILE" 2>/dev/null | tail -1 || echo "")
  result_score=$(jq -rc --arg sid "$SESSION_ID" 'select(.session == $sid) | .review_score' \
    "$METRICS_FILE" 2>/dev/null | tail -1 || echo "")
  if [[ "$result_verdict" != "$VERDICT" ]]; then
    echo "错误: 写入验证失败！期望 review_verdict=$VERDICT，实际读回 $result_verdict" >&2
    exit 1
  fi
  if ! jq -en --argjson a "$result_score" --argjson b "$SCORE" '$a == $b' >/dev/null; then
    echo "错误: 写入验证失败！期望 review_score=$SCORE，实际读回 $result_score" >&2
    exit 1
  fi
  echo "验证通过: session $SESSION_ID review_verdict=$result_verdict review_score=$result_score"
else
  result_verdict=$(jq -rc --arg sid "$SESSION_ID" 'select(.session == $sid) | .self_verdict' \
    "$METRICS_FILE" 2>/dev/null | tail -1 || echo "")
  result_score=$(jq -rc --arg sid "$SESSION_ID" 'select(.session == $sid) | .self_score' \
    "$METRICS_FILE" 2>/dev/null | tail -1 || echo "")
  if [[ "$result_verdict" != "$VERDICT" ]]; then
    echo "错误: 写入验证失败！期望 self_verdict=$VERDICT，实际读回 $result_verdict" >&2
    exit 1
  fi
  if ! jq -en --argjson a "$result_score" --argjson b "$SCORE" '$a == $b' >/dev/null; then
    echo "错误: 写入验证失败！期望 self_score=$SCORE，实际读回 $result_score" >&2
    exit 1
  fi
  echo "验证通过: session $SESSION_ID self_verdict=$result_verdict self_score=$result_score"
fi
