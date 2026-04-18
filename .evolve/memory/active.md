# Active Learnings

Accumulated wisdom from optimization iterations.

---

## Recent (last 2 weeks) — Full Detail

### [CONSOLIDATED] RLVR test_delta 误报：test_count_cache=0 是假警报，以 session log 为准
**Date:** 2026-04-18 | **Sessions:** 20260418-130019, 140058, 143829, 150833, 160105, 163949, 165640, 182502, 183531, 191429, 193217

**Context:** 连续多个 session 出现 RLVR 报 test_delta=+0 或 test_delta=-N 的情况，但实际增量均为正值。根因汇总：① `.test_count_cache_*` 写入 0（pytest 运行时缺依赖）；② cache 文件未 git add（untracked）；③ RLVR 在 agent work 空提交后、实际交付前快照了测试数量。

**Takeaway:**
- `.test_count_cache` 文件不可信，cache=0 大概率是环境问题，不是真实回退。
- 每次 session 开始和结束时手动运行 `pytest tests/ --co -q | tail -1` 记录真实数字，写入 session log。
- 收到 RLVR 零增量或负增量警告时，先交叉验证实际数量，再判断是否调整方向。不要被误报驱动改变正确策略。
- reflection commit 固定第一步：`git add .evolve/memory/.test_count_cache_*`，防止 untracked 触发 -N 误报。

---

### 手动补录 test_count_cache 会触发虚假负增量警报
**Date:** 2026-04-18 | **Session:** 20260418-184918

**Context:** 评审修复 session 中手动补录 test_count=388，系统将「0→388」解读为 test_delta=-388，触发回退警告。

**Takeaway:** 不在修复 session 里手动补录 cache；若必须补录，先在 session log 说明以便反思时区分虚报。让系统自动 diff，不要人工干预 cache 文件的值。

---

### fix-only session 会形成徘徊陷阱
**Date:** 2026-04-18 | **Sessions:** 20260418-162109, 163158

**Context:** 连续两个 session 做同一节点的 bug-fix，test_delta=+0，RLVR 反复惩罚，节点14 一直未启动。

**Takeaway:** bug-fix 和 test-add 必须捆绑：每修一个错误，同步新增 pytest 覆盖该修复点。孤立的 bug-fix session 永远 test_delta=+0，会陷入「fix→评审→fix→评审」死循环，无法推进知识库节点数。

---

### 可读性改造 session 与 test_delta 的结构性冲突
**Date:** 2026-04-18 | **Session:** 20260418-155035

**Context:** 节点06 Attention 做了可读性深度重写，RLVR test_delta=+0 是真实零增量（非误报）——纯文档重写本质不产生测试增量。

**Takeaway:** 「可读性改造」和「测试增量」是互斥 session 目标：纯文档重写永远 test_delta=+0，必须绑定 pytest 新增才可量化；接受这个约束比绕过它更诚实。

---

### 技术正确性与受众适配是两个独立质量维度
**Date:** 2026-04-18 | **Session:** 20260418-152409

**Context:** 完成 12 个节点后收到用户反馈：文章技术正确但「初中生看不懂」，可读性是独立于技术正确性的质量瓶颈。

**Takeaway:** 写作策略需周期性切换：交付新内容时侧重「技术正确+引用验证」，之后需专门做「可读性改造 session」来对齐受众——不能期待两者同时自动达标。

---

### 三件套同 session 交付流程已稳定
**Date:** 2026-04-18 | **Session:** 20260418-141432

**Context:** 节点07 Transformer 文档+notebook+pytest 一次性交付，第三次连续成功，bib 15/15 验证通过。

**Takeaway:** doc/notebook/test 三件套模板已固化：纯 NumPy 手撕 + nbconvert 验证 + bib cite-verify，可直接复用到后续节点，不需要再试错。

---

### 测试债务清零模式：兑现承诺比新增内容更重要
**Date:** 2026-04-18 | **Sessions:** 20260418-123514, 125113, 133816

**Context:** 连续三次 session 承诺写 pytest 测试均被更紧急任务覆盖，test_delta=+0 三连红灯；专门安排清债 session 后 test_delta 从 22→37 全绿。

**Takeaway:**
- Session 开始前先跑测试基线、读 commitments.md，未完成承诺优先执行，不允许 DIRECTIVE 或新节点直接覆盖。
- 当 RLVR 连续触发零增量警告时，下一个 session 应完全聚焦清债（补测试/修评审），而不是继续新节点。清债 session 的评价标准是 test_delta 和 verdict，不是节点数量。
- 知识节点的文档/notebook 和对应的 pytest 必须在同一个 session 内一起交付，否则测试债务会以「下次补」的名义永远延迟。

---

### notebook 梯度近似函数：dict key 混用类型会导致迭代解包失败
**Date:** 2026-04-18 | **Session:** 20260418-134625

**Context:** `numerical_gradient` 函数对某些参数用字符串 key，对另一些用元组 key，迭代时 `dict.items()` 解包失败导致 Cell 5 报错。

**Takeaway:** 梯度近似函数的返回值统一用 `list of (param_dict, name, grad)` 元组，禁止同一 dict 混用不同类型 key；这类 bug 只在 nbconvert 实际执行时暴露，事先 review gen_nb 脚本可提前发现。

---

### nbconvert 工作目录是 notebooks/，savefig 路径须相对于此
**Date:** 2026-04-18 | **Session:** 20260418-130735

**Context:** 生成 notebook 时 `savefig` 用了 `docs/assets/` 路径，nbconvert 执行时工作目录是 `notebooks/`，导致路径错误。

**Takeaway:** 生成 notebook 的脚本中所有路径要以 `notebooks/` 为基准，文件写到 `docs/` 时用 `../docs/assets/`。

---

### 知识库引用验证：DOI 链接需要 HEAD→GET fallback
**Date:** 2026-04-18 | **Session:** 20260418-122128

**Context:** 用 HEAD 请求验证 APA DOI 链接时，服务器返回 403，导致引用被误判为失效。

**Takeaway:** md-link-check 类工具验证 DOI 链接时必须先 HEAD，403/405 后再 GET fallback，否则所有 APA/Crossref DOI 都会误报失效。

---

## Medium (2–8 weeks old) — Condensed

*(No entries in this range as of 2026-04-18)*

---

## Old (8+ weeks) — Thematic Summaries

*(No entries in this range as of 2026-04-18)*
