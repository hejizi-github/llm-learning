"""
Tests for GPT-3 (2020) core mechanisms:
- Temperature sampling
- Top-k sampling
- In-context learning prompt formatting
- Scaling law behavior
- Mini GPT Block
"""
import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────
# Shared helpers (duplicated from notebook for test isolation)
# ──────────────────────────────────────────────────────────────────────────

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def sample_with_temperature(logits, temperature=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    if temperature <= 0:
        return int(np.argmax(logits))
    scaled = logits / temperature
    probs = softmax(scaled)
    return int(rng.choice(len(probs), p=probs))


def top_k_sample(logits, k=5, temperature=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    scaled = logits / max(temperature, 1e-8)
    top_k_idx = np.argsort(scaled)[-k:]
    masked = np.full_like(scaled, -np.inf)
    masked[top_k_idx] = scaled[top_k_idx]
    probs = softmax(masked)
    return int(rng.choice(len(probs), p=probs))


def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def causal_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    masked_scores = np.where(mask, scores, -1e9)
    attn_rows = np.stack([softmax(row) for row in masked_scores])
    return attn_rows @ V


def mini_gpt_block(x, d_model=32, d_ff=64, seed=0):
    seq_len = x.shape[0]
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    rng = np.random.default_rng(seed)
    x_norm = layer_norm(x)
    Wq = rng.standard_normal((d_model, d_model)) * 0.02
    Wk = rng.standard_normal((d_model, d_model)) * 0.02
    Wv = rng.standard_normal((d_model, d_model)) * 0.02
    attn_out = causal_attention(x_norm @ Wq, x_norm @ Wk, x_norm @ Wv, mask)
    x = x + attn_out
    x_norm2 = layer_norm(x)
    W1 = rng.standard_normal((d_model, d_ff)) * 0.02
    W2 = rng.standard_normal((d_ff, d_model)) * 0.02
    ffn_out = gelu(x_norm2 @ W1) @ W2
    return x + ffn_out


def format_few_shot(task_description, examples, query):
    prompt = task_description + "\n"
    for inp, out in examples:
        prompt += f"Input: {inp}\nOutput: {out}\n\n"
    prompt += f"Input: {query}\nOutput:"
    return prompt


# ──────────────────────────────────────────────────────────────────────────
# Temperature Sampling Tests
# ──────────────────────────────────────────────────────────────────────────

class TestTemperatureSampling:

    def test_zero_temperature_returns_argmax(self):
        logits = np.array([1.0, 5.0, 2.0, 0.5])
        result = sample_with_temperature(logits, temperature=0.0)
        assert result == 1  # index of max

    def test_greedy_always_picks_max(self):
        logits = np.array([0.1, 0.9, 0.3, 0.7])
        results = {sample_with_temperature(logits, temperature=0.0) for _ in range(10)}
        assert results == {1}

    def test_high_temperature_produces_varied_output(self):
        np.random.seed(7)
        logits = np.array([3.0, 2.9, 2.8, 2.7])  # nearly uniform
        rng = np.random.default_rng(42)
        results = {sample_with_temperature(logits, temperature=10.0, rng=rng) for _ in range(30)}
        assert len(results) > 1  # should visit multiple tokens

    def test_output_is_valid_token_index(self):
        logits = np.random.randn(50)
        rng = np.random.default_rng(0)
        result = sample_with_temperature(logits, temperature=1.0, rng=rng)
        assert 0 <= result < 50

    def test_softmax_probabilities_sum_to_one(self):
        logits = np.random.randn(100)
        probs = softmax(logits / 0.7)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_low_temperature_concentrates_probability(self):
        logits = np.array([3.0, 1.0, 0.5, 0.1])
        probs_low = softmax(logits / 0.1)
        probs_high = softmax(logits / 2.0)
        assert probs_low[0] > probs_high[0]  # low T → more concentrated

    def test_temperature_scaling_preserves_ordering(self):
        logits = np.array([3.0, 2.0, 1.0, 0.0])
        for T in [0.5, 1.0, 2.0]:
            probs = softmax(logits / T)
            # ordering should be preserved
            assert probs[0] > probs[1] > probs[2] > probs[3]

    def test_negative_logits_handled_correctly(self):
        logits = np.array([-1.0, -2.0, -3.0])
        result = sample_with_temperature(logits, temperature=0.0)
        assert result == 0  # least negative = highest


# ──────────────────────────────────────────────────────────────────────────
# Top-k Sampling Tests
# ──────────────────────────────────────────────────────────────────────────

class TestTopKSampling:

    def test_top1_equals_greedy(self):
        logits = np.array([5.0, 3.0, 1.0, 0.5])
        result = top_k_sample(logits, k=1, temperature=0.01)
        assert result == 0

    def test_top_k_restricts_to_top_tokens(self):
        logits = np.array([5.0, 4.0, 0.1, -10.0, -20.0])
        # With k=2, only tokens 0 and 1 should be sampled
        rng = np.random.default_rng(99)
        results = {top_k_sample(logits, k=2, temperature=1.0, rng=rng) for _ in range(100)}
        assert results.issubset({0, 1})

    def test_top_k_output_in_valid_range(self):
        vocab_size = 30
        logits = np.random.randn(vocab_size)
        rng = np.random.default_rng(5)
        result = top_k_sample(logits, k=5, rng=rng)
        assert 0 <= result < vocab_size

    def test_top_k_equals_vocab_size_allows_all_tokens(self):
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        rng = np.random.default_rng(123)
        results = {top_k_sample(logits, k=4, temperature=2.0, rng=rng) for _ in range(200)}
        assert len(results) > 1

    def test_top_k_never_samples_excluded_tokens(self):
        logits = np.array([0.1, 0.2, 100.0, -100.0])  # token 2 dominates
        rng = np.random.default_rng(7)
        results = {top_k_sample(logits, k=2, temperature=1.0, rng=rng) for _ in range(50)}
        assert 3 not in results  # token 3 is the lowest, should never appear

    def test_probabilities_after_top_k_sum_to_one(self):
        logits = np.array([3.0, 2.0, 1.0, 0.0, -1.0])
        k = 3
        scaled = logits / 1.0
        top_k_idx = np.argsort(scaled)[-k:]
        masked = np.full_like(scaled, -np.inf)
        masked[top_k_idx] = scaled[top_k_idx]
        probs = softmax(masked)
        assert abs(probs.sum() - 1.0) < 1e-6


# ──────────────────────────────────────────────────────────────────────────
# In-Context Learning Format Tests
# ──────────────────────────────────────────────────────────────────────────

class TestInContextLearning:

    def test_few_shot_contains_all_examples(self):
        task = "Translate English to French."
        examples = [("Hello", "Bonjour"), ("Thank you", "Merci")]
        query = "Good morning"
        prompt = format_few_shot(task, examples, query)
        assert "Hello" in prompt
        assert "Bonjour" in prompt
        assert "Merci" in prompt

    def test_few_shot_ends_with_output_colon(self):
        task = "Classify."
        examples = [("good", "positive")]
        query = "bad"
        prompt = format_few_shot(task, examples, query)
        assert prompt.rstrip().endswith("Output:")

    def test_few_shot_query_appears_last(self):
        task = "Task."
        examples = [("a", "1"), ("b", "2")]
        query = "c"
        prompt = format_few_shot(task, examples, query)
        # Query should appear after all examples
        last_example_pos = prompt.rfind("Output: 2")
        query_pos = prompt.rfind(f"Input: {query}")
        assert query_pos > last_example_pos

    def test_zero_examples_still_valid_prompt(self):
        task = "Summarize."
        prompt = format_few_shot(task, [], "Hello world")
        assert "Hello world" in prompt
        assert "Output:" in prompt

    def test_many_examples_all_included(self):
        task = "Math."
        examples = [(str(i), str(i * 2)) for i in range(10)]
        query = "11"
        prompt = format_few_shot(task, examples, query)
        for i in range(10):
            assert str(i * 2) in prompt


# ──────────────────────────────────────────────────────────────────────────
# Scaling Law Tests
# ──────────────────────────────────────────────────────────────────────────

class TestScalingLaw:

    def test_power_law_fit_has_negative_slope(self):
        params = np.array([1e6, 1e7, 1e8, 1e9, 1e10])
        losses = np.array([4.5, 3.8, 3.2, 2.8, 2.5])
        coeffs = np.polyfit(np.log10(params), np.log10(losses), 1)
        alpha = coeffs[0]
        assert alpha < 0, "Scaling law slope must be negative (more params → lower loss)"

    def test_larger_model_has_lower_loss(self):
        # Power law: L(N) = (N_c / N)^alpha with alpha > 0
        N_c = 1e13
        alpha = 0.076
        params = np.array([1e8, 1e9, 1e10, 1e11])
        losses = (N_c / params) ** alpha
        assert np.all(np.diff(losses) < 0), "Loss should decrease as N increases"

    def test_log_log_is_linear(self):
        # Generate exact power law and verify linear fit quality
        params = np.logspace(6, 11, 20)
        N_c, alpha = 1e13, 0.076
        losses = (N_c / params) ** alpha
        coeffs = np.polyfit(np.log10(params), np.log10(losses), 1)
        predicted = np.polyval(coeffs, np.log10(params))
        residuals = np.log10(losses) - predicted
        assert np.max(np.abs(residuals)) < 1e-8, "Power law should be perfectly linear in log-log"

    def test_scaling_exponent_reasonable_range(self):
        # From Kaplan 2020: alpha_N ≈ 0.076 for GPT-family
        N_c, alpha = 1e13, 0.076
        params = np.logspace(6, 11, 50)
        losses = (N_c / params) ** alpha
        fitted = np.polyfit(np.log10(params), np.log10(losses), 1)
        fitted_alpha = -fitted[0]  # slope is -alpha in log space
        assert 0.05 < fitted_alpha < 0.15, f"Fitted exponent {fitted_alpha:.3f} out of expected range"


# ──────────────────────────────────────────────────────────────────────────
# Mini GPT Block Tests
# ──────────────────────────────────────────────────────────────────────────

class TestMiniGPTBlock:

    def test_output_shape_preserved(self):
        x = np.random.randn(8, 32)
        out = mini_gpt_block(x, d_model=32, d_ff=64)
        assert out.shape == x.shape

    def test_layer_norm_zero_mean(self):
        x = np.random.randn(10, 16)
        normed = layer_norm(x)
        assert np.allclose(normed.mean(axis=-1), 0, atol=1e-5)

    def test_layer_norm_unit_std(self):
        x = np.random.randn(10, 16)
        normed = layer_norm(x)
        assert np.allclose(normed.std(axis=-1), 1, atol=1e-4)

    def test_gelu_positive_region(self):
        x = np.array([1.0, 2.0, 3.0])
        result = gelu(x)
        assert np.all(result > 0)

    def test_gelu_negative_region_near_zero(self):
        x = np.array([-3.0, -2.0, -1.0])
        result = gelu(x)
        assert np.all(result < 0.2)

    def test_causal_mask_shape(self):
        seq_len = 6
        mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
        assert mask.shape == (seq_len, seq_len)
        assert np.all(mask[np.triu_indices(seq_len, k=1)] == False)

    def test_gpt_block_different_inputs(self):
        x1 = np.random.randn(5, 16)
        x2 = x1 + 1.0
        out1 = mini_gpt_block(x1, d_model=16, d_ff=32)
        out2 = mini_gpt_block(x2, d_model=16, d_ff=32)
        assert not np.allclose(out1, out2)
