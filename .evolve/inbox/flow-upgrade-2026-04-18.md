[DIRECTIVE] FLOW.md 已升级：增加"能力轴"

## 本次 session 的唯一任务（不做新节点）

FLOW.md 有重大修改。在做任何新节点/新内容之前，**本次 session 优先做这四件事**：

1. **通读新 FLOW.md**
   - 理解"内容轴 + 能力轴"双轴约束
   - 理解 craft/ 的用途（已初始化空壳）
   - 理解"产出类型不是清单，是可组合元素"——不再强制六件套
   - 理解"reader 门控"（tools/reader-gate 待建）

2. **写 tools/reader-gate**
   - 用 bash，调用 `tools/claude-advisor --mode reader` 对每个新/改的 docs/*.md 做读者测试
   - 解析输出里"看不懂"、"失去兴趣"的数量
   - 超过阈值（建议 3 处）→ 退出码非 0（阻断 commit）
   - 把它加进 .evolve/config.toml 的 [verification].commands
   - 注意：reader-gate 不应在每个 commit 都跑全部 docs（太慢），只跑**本次 session 修改过**的 .md

3. **用 reader persona 审计已有节点**
   - 对 docs/01..docs/16 逐个跑 `tools/claude-advisor --mode reader`
   - 把"看不懂"的段落**原文**归档到 `craft/failed-attempts.md`
   - 不需要修复，只归档（修复留给后续"风格打磨"类 session）
   - 目标：failed-attempts.md 里至少有 5 条真实样本

4. **从已有节点抽取 craft 样本**
   - 审 01..16 的开场段，选出写得好的 3 条加到 `craft/great-openers.md`
   - 从已有文档里找已经奏效的类比，至少 3 条加到 `craft/great-analogies.md`
   - 找一段数学铺垫做得好的，加到 `craft/math-scaffolding.md`

## 本 session 不要做的事

- **不要**开新节点（节点17 DDIM / 节点18 之类）
- **不要**修复 DDPM 节点16 的可读性（留给后续 session）
- **不要**在没读 FLOW 的情况下乱动 craft/

## 验收

session 结束时：
- `tools/reader-gate` 存在且可执行
- `.evolve/config.toml` 的 [verification].commands 包含 reader-gate
- `craft/failed-attempts.md` 至少 5 条真实样本（来自现有节点）
- `craft/great-openers.md` / `great-analogies.md` 各至少 3 条样本
- journal 的"写作反思"段落首次出现（按新 FLOW 的层 2 格式）

## 背景

这个 FLOW 升级是因为观察到：
- 节点数在增长（内容轴 OK）
- 但每个节点的写作水平没明显进步（能力轴停滞）
- 节点16 DDPM 拿到 depth-score 5/5，但对初中生而言大量数学裸奔
- 节点17 DDIM 刚刚的尝试又重复了这个问题

你自己在 active learnings 里已经识别了"技术正确性与受众适配是两个独立质量维度"，但因为没有度量机制，这个识别没有变成行为改变。新 FLOW + reader-gate 是给这个识别配上执行牙齿。
