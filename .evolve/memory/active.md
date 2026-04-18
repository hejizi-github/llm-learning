# Active Learnings

Accumulated wisdom from optimization iterations.

---

## Recent (last 2 weeks) — Full Detail

### RLVR 零增量警告需与 pytest 计数交叉验证
**Date:** 2026-04-18 | **Session:** 20260418-163949

**Context:** Session log 显示 test_delta=+23，但 RLVR 报告 +0；根因是评审器可能在中间提交时快照测试数量。

**Takeaway:** 每次交付完成后立即运行 `pytest --co -q | wc -l`，记录实际数量；若 RLVR 信号与之矛盾则在 journal 标注根因，不被零增量警告牵着走改变方向。

---

### fix-only session 会形成徘徊陷阱
**Date:** 2026-04-18 | **Session:** 20260418-163158

**Context:** 连续两个 session 做同一节点的 bug-fix，test_delta=+0，RLVR 反复惩罚，节点14 一直未启动。

**Takeaway:** bug-fix session 必须 fix+test 绑定：每修一个错误，同步新增 pytest 覆盖该修复点。否则会陷入「fix→评审→fix→评审」死循环，无法推进知识库节点数。

---

### RLVR -N 误报根因：test_count_cache 未 git add
**Date:** 2026-04-18 | **Session:** 20260418-160105

**Context:** RLVR 报 test_delta=-279，实际是 +22。.test_count_cache 为 untracked，RLVR 读前缓存=279、新缓存=0，差值=-279。同一 bug 在 session 130019 已记录，但 reflection 仍未 git add。

**Takeaway:** reflection commit 的固定第一步必须是 `git add .evolve/memory/.test_count_cache_*`；当 RLVR 报大幅负增量（-N ≈ 上次测试总数）时，直接检查 test_count_cache 是否已 committed。

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

**Takeaway:** doc/notebook/test 三件套模板已固化：纯 NumPy 手撕 + nbconvert 验证 + bib cite-verify，可直接复用到后续节点，不需要再试错流程。

---

### notebook 梯度近似函数：dict key 混用类型会导致迭代解包失败
**Date:** 2026-04-18 | **Session:** 20260418-134625

**Context:** numerical_gradient 函数对某些参数用字符串 key，对另一些用元组 key，迭代时 dict.items() 解包失败导致 Cell 5 报错。

**Takeaway:** 梯度近似函数的返回值统一用 `list of (param_dict, name, grad)` 元组，禁止同一 dict 混用不同类型 key；这类 bug 只在 nbconvert 实际执行时暴露，事先 review gen_nb 脚本可以提前发现。

---

### 测试债务清零模式：兑现承诺比新增内容更重要
**Date:** 2026-04-18 | **Session:** 20260418-133816

**Context:** 连续两个 session 出现 test_delta=+0（内容和测试拆分交付），专门安排清债 session 后 test_delta 从 22→37 全绿。

**Takeaway:** 当 RLVR 连续触发零增量警告时，下一个 session 应完全聚焦清债（补测试/修评审），而不是继续新节点。清债 session 的评价标准是 test_delta 和 verdict，不是知识节点数量。

---

### 内容交付与测试覆盖必须同 session 绑定
**Date:** 2026-04-18 | **Session:** 20260418-132534

**Context:** 节点04 LeNet 交付了完整文档+notebook，但 test_delta=+0，连续两次节点都是"先内容后测试"模式触发 RLVR 零增量警告。

**Takeaway:** 知识节点的文档/notebook 和对应的 pytest 测试用例必须在同一个 session 内一起交付，否则测试债务会以"下次补"的名义永远延迟。

---

### nbconvert 工作目录是 notebooks/，savefig 路径须相对于此
**Date:** 2026-04-18 | **Session:** 20260418-130735

**Context:** 生成 notebook 时 savefig 用了 `docs/assets/` 路径，nbconvert 执行时工作目录是 `notebooks/`，导致路径错误。

**Takeaway:** 生成 notebook 的脚本中所有路径要以 `notebooks/` 为基准，文件写到 `docs/` 时用 `../docs/assets/`。

---

### 承诺需要硬性前置检查机制
**Date:** 2026-04-18 | **Session:** 20260418-125113

**Context:** 连续三次 session 承诺写 pytest 测试，每次都被更紧急任务（基础设施/内容节点/DIRECTIVE）推迟，导致 test_delta=+0 三连红灯。

**Takeaway:** Session 开始时先跑测试基线、读 commitments.md，若有未完成承诺则优先执行，不允许任何新任务（包括 DIRECTIVE）直接覆盖承诺。

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
