current_goal: 用纯规则（关键词匹配/正则）评估知识节点质量（tools/depth-score）
new_goal: 改用 LLM 子 Agent 调用来评估节点质量，pure-rule 工具仅做结构检查（引用、链接、notebook可运行）
reason: |
  用户 DIRECTIVE（20260418-123509）指出：
  1. 纯规则评估会过拟合（Agent 可以堆关键词来虚高分数）
  2. 执行者不应定义自己的评价标准（利益冲突）
  3. 建议引入子 Agent 来做质量评估
  
  当前 depth-score 的工作方式：在 md 文件里搜索特定 pattern（如 `背景|故事|年代`），
  统计命中次数来判断"有没有背景故事"。这是一个可被轻易游戏的规则。
  
  建议的新架构：
  - tools/depth-score（规则版，保留）→ 只检查"结构完整性"（有 6 个章节标题？有引用块？有 notebook 链接？）
  - tools/quality-eval（新建，LLM 版）→ 调用 claude CLI 子进程，传入完整 md + quality-rubric，
    返回 JSON 格式的评分（0-5 每维度）和具体改进建议
  - 两者分工：depth-score 作为快速"格式门控"，quality-eval 作为深度"内容评审"
  
  实现草案：
    tools/quality-eval <doc.md>
    → 读取 strategies/quality-rubric.md 作为评审 prompt
    → 调用 `claude --print "请根据以下 rubric 评估..." `
    → 解析输出为 JSON，打印各维度分数
  
  风险：需要 claude CLI 可用，且每次调用有 token 消耗。建议设为"可选验证"，
  不列入护栏指标（护栏只用规则工具），但在 journal 中记录 LLM 评审结果。
  
status: PENDING
