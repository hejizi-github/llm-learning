# Active Learnings

Accumulated wisdom from optimization iterations.

---

## Recent (last 2 weeks) — Full Detail

### 知识库引用验证：DOI 链接需要 HEAD→GET fallback
**Date:** 2026-04-18 | **Session:** 20260418-122128

**Context:** 用 HEAD 请求验证 APA DOI 链接时，服务器返回 403，导致引用被误判为失效。

**Takeaway:** md-link-check 类工具验证 DOI 链接时必须先 HEAD，403/405 后再 GET fallback，否则所有 APA/Crossref DOI 都会误报失效。

---

### pytest 测试债务的滚雪球效应：连续推迟的代价
**Date:** 2026-04-18 | **Session:** 20260418-123514

**Context:** test_delta=+0 连续两个 session 出现；tests/ 目录至今不存在；每次都选择优先交付内容节点导致承诺被推迟。

**Takeaway:** 内容交付和测试覆盖不是非此即彼——每次 session 开始前用 5-10 分钟先确认测试基线，哪怕只加 1 个 smoke test，也比持续推迟强；RLVR 连续红灯是系统在说"方向不对，不是执行问题"。

---

### 承诺被 DIRECTIVE 覆盖的结构性漏洞
**Date:** 2026-04-18 | **Session:** 20260418-125113

**Context:** 连续三次 session 承诺写 pytest 测试，但每次都被更紧急的任务（基础设施/内容节点/DIRECTIVE）推迟，导致 test_delta=+0 三连红灯。

**Takeaway:** 承诺需要硬性前置检查机制：session 开始时先跑测试基线、读 commitments.md，若有未完成承诺则优先执行，而不是允许任何新任务直接覆盖承诺。

---

### RLVR 误报 test_delta=+0 的根因：.test_count_cache_* 未被 git 追踪
**Date:** 2026-04-18 | **Session:** 20260418-130019

**Context:** 本次 session 实际 test_delta=+10，但 reflection 收到 RLVR 红灯（test_delta=+0）。根因是 `.evolve/memory/.test_count_cache_20260418-130019` 为 untracked 文件，RLVR 系统找不到新缓存，退化为 +0。

**Takeaway:** 每次 reflection 提交时必须显式 `git add .evolve/memory/.test_count_cache_*`，确保 RLVR 能读到正确的测试基线；否则即使实际有增量，RLVR 也会误报红灯。

---

### nbconvert 工作目录是 notebooks/，savefig 路径须相对于此
**Date:** 2026-04-18 | **Session:** 20260418-130735

**Context:** 生成 notebook 时 savefig 用了 `docs/assets/` 路径，nbconvert 执行时工作目录是 `notebooks/`，导致路径错误。

**Takeaway:** 生成 notebook 的脚本中所有路径要以 `notebooks/` 为基准，文件写到 `docs/` 时用 `../docs/assets/`。

---

### 内容交付与测试覆盖必须绑定，不能拆成两个 session
**Date:** 2026-04-18 | **Session:** 20260418-132534

**Context:** 节点04 LeNet 交付了完整文档+notebook，但 test_delta=+0，连续两次节点都是"先内容后测试"模式导致 RLVR 零增量警告。

**Takeaway:** 知识节点的文档/notebook 和对应的 pytest 测试用例必须在同一个 session 内一起交付，否则测试债务会以"下次补"的名义永远延迟。

---

## Medium (2–8 weeks old) — Condensed

*(No entries in this range as of 2026-04-18)*

---

## Old (8+ weeks) — Thematic Summaries

*(No entries in this range as of 2026-04-18)*
