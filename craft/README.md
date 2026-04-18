# Craft — 写作资产库

FLOW.md 的"能力轴"依赖这个目录。Agent 每次 session 前后都要读写这里。

## 文件用途

- `great-openers.md` — 写得好的开场段（附点评：为什么好）
- `great-analogies.md` — 有效的类比（附使用场景 + 为什么奏效）
- `math-scaffolding.md` — 给初中生铺垫高级数学概念的样本
- `failed-attempts.md` — 被 reader persona / 读者判为"看不懂"的段落（反面教材）
- `exemplars/` — 外部优秀教学范例的节选（Feynman Lectures / 3Blue1Brown 脚本 / 科学美国人 ...）

## 用法

**写作前**：读 `great-analogies.md` 和 `math-scaffolding.md` 至少 2 段
**写作后**：抽取本次值得保留的段落加入对应文件；失败段落加入 `failed-attempts.md`

## 原则

- 每条样本必须注明来源（哪个节点哪段）和点评（为什么好/为什么失败）
- 不是越多越好——质量优于数量
- 失败比成功更重要：failed-attempts.md 每增加一条都是防止未来重蹈覆辙的投资
