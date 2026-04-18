#!/usr/bin/env python3
# gen_nb_10.py -- generate notebooks/10-gpt3-2020.ipynb
import json, pathlib

cells = []

def code(src): return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': src}
def md(src): return {'cell_type': 'markdown', 'metadata': {}, 'source': src}

# -- Cell 0: Setup -------------------------------------------------------
cells.append(md(
    '# 节点10：GPT-3（2020）——1750 亿参数，少样本学习的涌现\n\n'
    '本 notebook 用纯 NumPy 模拟 GPT-3 的核心机制：\n'
    '- 自回归采样（温度 + top-k）\n'
    '- In-context learning 格式化\n'
    '- Scaling Law 可视化\n'
    '- Mini GPT-3 Block（Pre-LN + Causal Attention）'
))

cells.append(code(
    'import numpy as np\n'
    'import matplotlib\n'
    'matplotlib.rcParams["font.family"] = "DejaVu Sans"\n'
    'import matplotlib.pyplot as plt\n'
    'import warnings\n'
    'warnings.filterwarnings("ignore")\n'
    'np.random.seed(42)\n'
    'print("Setup complete")'
))

# -- Cell 1: Autoregressive Sampling with Temperature --------------------
cells.append(md(
    '## Step 1: 自回归采样——温度控制\n\n'
    'GPT-3 生成文本时，每次从词汇表的概率分布中**采样**下一个词。\n\n'
    '**温度（Temperature）** 控制分布的"尖锐程度"：\n'
    '- 温度 → 0：总是选概率最高的词（贪心，保守）\n'
    '- 温度 = 1：按原始概率采样（标准）\n'
    '- 温度 > 1：分布更平，更随机，更有创意\n\n'
    '数学：先把 logits（原始分数）除以温度，再取 softmax：\n\n'
    '$$P_T(x_i) = \\frac{\\exp(z_i / T)}{\\sum_j \\exp(z_j / T)}$$'
))

cells.append(code(
    'def softmax(x):\n'
    '    """Numerically stable softmax."""\n'
    '    x = x - np.max(x)\n'
    '    e = np.exp(x)\n'
    '    return e / e.sum()\n'
    '\n'
    'def sample_with_temperature(logits, temperature=1.0):\n'
    '    """Sample a token index from logits with temperature scaling."""\n'
    '    if temperature <= 0:\n'
    '        return int(np.argmax(logits))\n'
    '    scaled = logits / temperature\n'
    '    probs = softmax(scaled)\n'
    '    return int(np.random.choice(len(probs), p=probs))\n'
    '\n'
    '# Demonstrate: same logits, different temperatures\n'
    'vocab = ["apple", "banana", "cherry", "date", "elderberry"]\n'
    'logits = np.array([3.0, 2.0, 1.0, 0.5, 0.2])  # apple is most likely\n'
    '\n'
    'fig, axes = plt.subplots(1, 4, figsize=(14, 3))\n'
    'temps = [0.1, 0.5, 1.0, 2.0]\n'
    'for ax, T in zip(axes, temps):\n'
    '    probs = softmax(logits / T) if T > 0 else (np.arange(5) == np.argmax(logits)).astype(float)\n'
    '    ax.bar(vocab, probs, color="steelblue", alpha=0.7)\n'
    '    ax.set_title(f"T = {T}")\n'
    '    ax.set_ylim(0, 1)\n'
    '    ax.tick_params(axis="x", rotation=30)\n'
    'plt.suptitle("Temperature Effect on Sampling Distribution", y=1.02)\n'
    'plt.tight_layout()\n'
    'plt.savefig("../docs/assets/10_temperature.png", dpi=80, bbox_inches="tight")\n'
    'plt.show()\n'
    'print("T=0.1 → near-greedy, T=2.0 → diverse sampling")'
))

# -- Cell 2: Top-k Sampling ----------------------------------------------
cells.append(md(
    '## Step 2: Top-k 采样\n\n'
    'Temperature 采样的问题：即使概率极低的词也可能被采到，生成胡言乱语。\n\n'
    '**Top-k 采样**：只保留概率最高的 k 个词，其余置零再重新归一化。\n\n'
    '这样既有随机性（有趣），又限制了"踩雷"（不会生成完全不合理的词）。'
))

cells.append(code(
    'def top_k_sample(logits, k=5, temperature=1.0):\n'
    '    """Top-k sampling: keep only top-k logits, then sample."""\n'
    '    scaled = logits / max(temperature, 1e-8)\n'
    '    # Get top-k indices\n'
    '    top_k_idx = np.argsort(scaled)[-k:]\n'
    '    # Mask out non-top-k\n'
    '    masked = np.full_like(scaled, -np.inf)\n'
    '    masked[top_k_idx] = scaled[top_k_idx]\n'
    '    probs = softmax(masked)\n'
    '    return int(np.random.choice(len(probs), p=probs))\n'
    '\n'
    '# Simulate a short autoregressive generation loop\n'
    'VOCAB_SIZE = 20\n'
    'SEQ_LEN = 10\n'
    '\n'
    'def autoregressive_generate(seed_tokens, n_new=5, k=5, temperature=0.8):\n'
    '    """Simple autoregressive generation using random "model" logits."""\n'
    '    tokens = list(seed_tokens)\n'
    '    for step in range(n_new):\n'
    '        # In a real GPT, the model would compute logits from all past tokens\n'
    '        # Here we simulate with deterministic pseudo-logits based on position\n'
    '        np.random.seed(step + sum(tokens))  # reproducible but position-dependent\n'
    '        logits = np.random.randn(VOCAB_SIZE)\n'
    '        next_tok = top_k_sample(logits, k=k, temperature=temperature)\n'
    '        tokens.append(next_tok)\n'
    '    return tokens\n'
    '\n'
    'seed = [1, 5, 3]\n'
    'generated = autoregressive_generate(seed, n_new=7, k=5, temperature=0.8)\n'
    'print(f"Seed tokens:      {seed}")\n'
    'print(f"Generated tokens: {generated}")\n'
    'print(f"New tokens added: {generated[len(seed):]}")'
))

# -- Cell 3: In-Context Learning Format ----------------------------------
cells.append(md(
    '## Step 3: In-Context Learning——格式的魔力\n\n'
    'GPT-3 最核心的发现：**只要把例子放进提示词，模型就能"学会"新任务**，无需更新任何参数。\n\n'
    '下面我们模拟三种 ICL 格式，观察语言模型在生成时如何利用上下文。\n\n'
    '关键洞察：模型在预训练时看过无数"例子→答案"格式的文本，所以它"知道"这种格式意味着什么。'
))

cells.append(code(
    'def format_zero_shot(task_description, query):\n'
    '    """Zero-shot: just the task description + query."""\n'
    '    return f"{task_description}\\nInput: {query}\\nOutput:"\n'
    '\n'
    'def format_one_shot(task_description, example_in, example_out, query):\n'
    '    """One-shot: one example + query."""\n'
    '    return (\n'
    '        f"{task_description}\\n"\n'
    '        f"Input: {example_in}\\nOutput: {example_out}\\n\\n"\n'
    '        f"Input: {query}\\nOutput:"\n'
    '    )\n'
    '\n'
    'def format_few_shot(task_description, examples, query):\n'
    '    """Few-shot: multiple examples + query."""\n'
    '    prompt = task_description + "\\n"\n'
    '    for inp, out in examples:\n'
    '        prompt += f"Input: {inp}\\nOutput: {out}\\n\\n"\n'
    '    prompt += f"Input: {query}\\nOutput:"\n'
    '    return prompt\n'
    '\n'
    '# Sentiment analysis examples\n'
    'task = "Classify sentiment as Positive or Negative."\n'
    'examples = [\n'
    '    ("This movie is amazing!", "Positive"),\n'
    '    ("I hate waiting in long lines.", "Negative"),\n'
    '    ("The food was absolutely delicious.", "Positive"),\n'
    ']\n'
    'query = "The service was terrible and the room was dirty."\n'
    '\n'
    'print("=== Zero-shot prompt ===")\n'
    'print(format_zero_shot(task, query))\n'
    'print()\n'
    'print("=== One-shot prompt ===")\n'
    'print(format_one_shot(task, examples[0][0], examples[0][1], query))\n'
    'print()\n'
    'print("=== Few-shot prompt (3 examples) ===")\n'
    'print(format_few_shot(task, examples, query))'
))

# -- Cell 4: Scaling Law Visualization -----------------------------------
cells.append(md(
    '## Step 4: 规模律（Scaling Laws）可视化\n\n'
    'Kaplan et al. (2020) 发现，语言模型的损失和参数量之间满足**幂律关系**：\n\n'
    '$$L(N) \\approx \\left(\\frac{N_c}{N}\\right)^{\\alpha_N}$$\n\n'
    '在 log-log 坐标下，这是一条直线。GPT-3 的设计就是沿着这条线外推的。\n\n'
    '下面我们可视化实际的规模律数据（来自 Kaplan et al. 论文的近似值）。'
))

cells.append(code(
    '# Approximate data from Kaplan et al. 2020 (Figure 1)\n'
    '# (N, L) pairs: parameter count vs validation loss\n'
    'scaling_data = [\n'
    '    (1e6,   4.60),   # 1M params\n'
    '    (3e6,   4.00),\n'
    '    (1e7,   3.50),\n'
    '    (3e7,   3.10),\n'
    '    (1e8,   2.80),\n'
    '    (3e8,   2.55),\n'
    '    (1e9,   2.35),\n'
    '    (3e9,   2.18),\n'
    '    (1e10,  2.05),\n'
    '    (3e10,  1.92),\n'
    '    (1e11,  1.82),   # ~100B\n'
    '    (1.75e11, 1.77), # GPT-3 (175B)\n'
    ']\n'
    '\n'
    'params = np.array([d[0] for d in scaling_data])\n'
    'losses = np.array([d[1] for d in scaling_data])\n'
    '\n'
    '# Fit power law: log(L) = alpha * log(N) + C\n'
    'log_params = np.log10(params)\n'
    'log_losses = np.log10(losses)\n'
    'coeffs = np.polyfit(log_params, log_losses, 1)\n'
    'alpha = coeffs[0]\n'
    '\n'
    'fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n'
    '\n'
    '# Linear scale\n'
    'ax1.scatter(params, losses, color="steelblue", s=50, zorder=3)\n'
    'ax1.scatter([1.75e11], [1.77], color="red", s=120, zorder=4, label="GPT-3 (175B)")\n'
    'ax1.set_xscale("log")\n'
    'ax1.set_xlabel("Parameters (log scale)")\n'
    'ax1.set_ylabel("Validation Loss")\n'
    'ax1.set_title("Scaling Law: Parameters vs Loss")\n'
    'ax1.legend()\n'
    'ax1.grid(True, alpha=0.3)\n'
    '\n'
    '# Log-log scale (should be linear)\n'
    'fit_params = np.logspace(6, 11.5, 100)\n'
    'fit_losses = 10 ** np.polyval(coeffs, np.log10(fit_params))\n'
    'ax2.scatter(log_params, log_losses, color="steelblue", s=50, zorder=3)\n'
    'ax2.scatter([np.log10(1.75e11)], [np.log10(1.77)], color="red", s=120, zorder=4, label="GPT-3 (175B)")\n'
    'ax2.plot(np.log10(fit_params), np.log10(fit_losses), "k--", alpha=0.7, label=f"Power law fit (α={alpha:.3f})")\n'
    'ax2.set_xlabel("log₁₀(Parameters)")\n'
    'ax2.set_ylabel("log₁₀(Loss)")\n'
    'ax2.set_title("Log-Log: Linear Relationship")\n'
    'ax2.legend()\n'
    'ax2.grid(True, alpha=0.3)\n'
    '\n'
    'plt.tight_layout()\n'
    'plt.savefig("../docs/assets/10_scaling_law.png", dpi=80, bbox_inches="tight")\n'
    'plt.show()\n'
    'print(f"Power law exponent α ≈ {alpha:.3f}")\n'
    'print(f"Interpretation: 10x more params → loss multiplied by 10^{alpha:.3f} = {10**alpha:.3f}")'
))

# -- Cell 5: Mini GPT Block (same as GPT-2 but shown for completeness) ---
cells.append(md(
    '## Step 5: Mini GPT Block（Pre-LN Transformer Decoder）\n\n'
    'GPT-3 的网络架构和 GPT-2 几乎相同（Pre-LN + Causal Attention + FFN）。\n'
    '区别只在于规模：GPT-3 有 96 层，每层隐向量维度 12288，96 个注意力头。\n\n'
    '下面实现一个完整的 Mini GPT Block，验证前向传播的形状和行为。'
))

cells.append(code(
    'def layer_norm(x, eps=1e-5):\n'
    '    """Layer normalization."""\n'
    '    mean = x.mean(axis=-1, keepdims=True)\n'
    '    std = x.std(axis=-1, keepdims=True)\n'
    '    return (x - mean) / (std + eps)\n'
    '\n'
    'def gelu(x):\n'
    '    """GELU activation (approximate)."""\n'
    '    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))\n'
    '\n'
    'def causal_attention(Q, K, V, mask):\n'
    '    """Scaled dot-product attention with causal mask."""\n'
    '    d_k = Q.shape[-1]\n'
    '    scores = Q @ K.T / np.sqrt(d_k)\n'
    '    scores = np.where(mask, scores, -1e9)\n'
    '    attn = softmax(scores.reshape(-1)).reshape(scores.shape)\n'
    '    # Apply softmax row-wise\n'
    '    attn_rows = np.stack([softmax(row) for row in scores])\n'
    '    return attn_rows @ V\n'
    '\n'
    'def mini_gpt_block(x, d_model=32, d_ff=64):\n'
    '    """One Pre-LN GPT block: LN → Attention → residual → LN → FFN → residual."""\n'
    '    seq_len = x.shape[0]\n'
    '    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))\n'
    '\n'
    '    # Pre-LN + Self-Attention\n'
    '    x_norm = layer_norm(x)\n'
    '    np.random.seed(0)\n'
    '    Wq = np.random.randn(d_model, d_model) * 0.02\n'
    '    Wk = np.random.randn(d_model, d_model) * 0.02\n'
    '    Wv = np.random.randn(d_model, d_model) * 0.02\n'
    '    Q = x_norm @ Wq\n'
    '    K = x_norm @ Wk\n'
    '    V = x_norm @ Wv\n'
    '    attn_out = causal_attention(Q, K, V, mask)\n'
    '    x = x + attn_out  # residual\n'
    '\n'
    '    # Pre-LN + FFN\n'
    '    x_norm2 = layer_norm(x)\n'
    '    W1 = np.random.randn(d_model, d_ff) * 0.02\n'
    '    W2 = np.random.randn(d_ff, d_model) * 0.02\n'
    '    ffn_out = gelu(x_norm2 @ W1) @ W2\n'
    '    x = x + ffn_out  # residual\n'
    '\n'
    '    return x\n'
    '\n'
    '# Test with a sequence of 8 tokens, d_model=32\n'
    'seq_len, d_model = 8, 32\n'
    'x = np.random.randn(seq_len, d_model).astype(np.float32)\n'
    'out = mini_gpt_block(x, d_model=d_model, d_ff=64)\n'
    'print(f"Input shape:  {x.shape}")\n'
    'print(f"Output shape: {out.shape}")\n'
    'print(f"Shape preserved: {x.shape == out.shape}")\n'
    'print(f"Output norm (mean): {np.linalg.norm(out, axis=-1).mean():.3f}")'
))

# -- Cell 6: Emergent Arithmetic Capability Simulation -------------------
cells.append(md(
    '## Step 6: 涌现能力模拟——算术任务\n\n'
    'GPT-3 论文中一个令人惊讶的发现：模型能做两位数加法（~98% 准确率）。\n'
    '这能力在小模型中几乎不存在，在大模型中"突然出现"——这就是**涌现（emergence）**。\n\n'
    '下面我们模拟 few-shot 算术的 prompt 格式，以及简单的词元级加法。'
))

cells.append(code(
    'def format_arithmetic_few_shot(examples, query):\n'
    '    """Format arithmetic few-shot prompt."""\n'
    '    prompt = "Solve the arithmetic problem.\\n\\n"\n'
    '    for a, b, ans in examples:\n'
    '        prompt += f"Q: {a} + {b} = ?\\nA: {ans}\\n\\n"\n'
    '    prompt += f"Q: {query[0]} + {query[1]} = ?\\nA:"\n'
    '    return prompt\n'
    '\n'
    'examples = [(12, 34, 46), (55, 23, 78), (71, 18, 89)]\n'
    'query = (47, 35)\n'
    'prompt = format_arithmetic_few_shot(examples, query)\n'
    'print(prompt)\n'
    'print(f"\\nExpected answer: {query[0] + query[1]}")\n'
    '\n'
    '# Simulate "emergent" accuracy vs model size (from Kaplan 2020 / GPT-3 paper)\n'
    'model_sizes = [125e6, 350e6, 1.3e9, 6.7e9, 13e9, 175e9]\n'
    'two_digit_acc = [0.02, 0.05, 0.15, 0.45, 0.75, 0.98]  # approximate\n'
    '\n'
    'fig, ax = plt.subplots(figsize=(8, 4))\n'
    'ax.plot([np.log10(s) for s in model_sizes], two_digit_acc, "o-", color="steelblue", linewidth=2)\n'
    'ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% threshold")\n'
    'ax.set_xlabel("log₁₀(Parameters)")\n'
    'ax.set_ylabel("Accuracy on 2-digit addition")\n'
    'ax.set_title("Emergent Arithmetic: Accuracy vs Model Size")\n'
    'ax.set_xticks([np.log10(s) for s in model_sizes])\n'
    'ax.set_xticklabels(["125M", "350M", "1.3B", "6.7B", "13B", "175B"])\n'
    'ax.legend()\n'
    'ax.grid(True, alpha=0.3)\n'
    'plt.tight_layout()\n'
    'plt.savefig("../docs/assets/10_emergence.png", dpi=80, bbox_inches="tight")\n'
    'plt.show()\n'
    'print("Emergent capability: near-zero at small scale, ~98% at GPT-3 scale")'
))

# -- Cell 7: Summary -------------------------------------------------------
cells.append(md(
    '## 总结\n\n'
    '| 概念 | 核心公式/机制 | 一句话 |\n'
    '|------|-------------|--------|\n'
    '| 自回归采样 | $P_T(x_i) = \\text{softmax}(z_i/T)$ | 温度越低越保守，越高越随机 |\n'
    '| Top-k 采样 | 只保留前 k 个 logit | 随机但不胡言乱语 |\n'
    '| In-context learning | prompt 里放例子，参数不变 | 不训练，靠格式"教"模型 |\n'
    '| 规模律 | $L(N) \\approx (N_c/N)^{\\alpha}$ | log-log 下是直线，可外推 |\n'
    '| 涌现能力 | 小模型几乎 0，大模型突然出现 | 量变引发质变 |\n\n'
    '**GPT-3 的历史意义**：证明了"大就是强"——规模本身就是一种算法。\n'
    '它留下的问题（不听指令、幻觉、成本高）直接催生了 InstructGPT 和 ChatGPT。'
))

# Build notebook
nb = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.8.0'},
    },
    'cells': cells,
}

out = pathlib.Path(__file__).parent.parent / 'notebooks' / '10-gpt3-2020.ipynb'
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f'Written {out}  ({len(cells)} cells)')
