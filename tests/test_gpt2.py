"""Tests for GPT-2 (2019) node — Causal Mask, Causal Attention, BPE, Layer Norm, GELU,
Temperature Sampling, and Scaling Laws."""
import numpy as np
import pytest


# ── Helper implementations (no imports from notebook files) ───────────────────

def make_causal_mask(seq_len):
    """Lower-triangular mask: position i can attend to positions 0..i."""
    return np.tril(np.ones((seq_len, seq_len)))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_attention(Q, K, V, mask=None):
    """Scaled dot-product attention with optional causal mask."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = scores + (1 - mask) * (-1e9)
    weights = softmax(scores)
    return weights @ V, weights


def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def temperature_sample(logits, temperature=1.0):
    if temperature == 0:
        return int(np.argmax(logits))
    scaled = logits / temperature
    probs = softmax(scaled)
    return int(np.random.choice(len(probs), p=probs))


# ── BPE helpers ────────────────────────────────────────────────────────────────

def get_pairs(vocab):
    """Return set of all adjacent symbol pairs across all words in vocab dict.

    vocab: dict mapping tuple-of-symbols -> frequency
    """
    pairs = {}
    for word, freq in vocab.items():
        symbols = list(word)
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs


def merge_vocab(pair, vocab):
    """Merge the most frequent pair in vocab, returning new vocab dict."""
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in vocab.items():
        # word is a tuple; rebuild as string, merge, split back
        word_str = " ".join(word)
        word_str = word_str.replace(bigram, replacement)
        new_vocab[tuple(word_str.split())] = freq
    return new_vocab


def count_tokens(vocab):
    """Total number of symbol tokens across all words."""
    return sum(len(word) * freq for word, freq in vocab.items())


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_vocab():
    """Small BPE vocabulary: words as tuples of characters."""
    return {
        ("l", "o", "w"): 5,
        ("l", "o", "w", "e", "r"): 2,
        ("n", "e", "w", "e", "s", "t"): 6,
        ("w", "i", "d", "e", "s", "t"): 3,
    }


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ── 1. Causal Mask ─────────────────────────────────────────────────────────────

class TestCausalMask:

    def test_causal_mask_shape(self):
        mask = make_causal_mask(4)
        assert mask.shape == (4, 4)

    def test_causal_mask_lower_triangle(self):
        """Diagonal and below should all be 1."""
        mask = make_causal_mask(5)
        rows, cols = np.tril_indices(5)
        assert np.all(mask[rows, cols] == 1.0)

    def test_causal_mask_upper_triangle(self):
        """Above diagonal should all be 0."""
        mask = make_causal_mask(5)
        rows, cols = np.triu_indices(5, k=1)
        assert np.all(mask[rows, cols] == 0.0)

    def test_causal_mask_1x1(self):
        mask = make_causal_mask(1)
        assert mask.shape == (1, 1)
        assert mask[0, 0] == 1.0

    def test_causal_mask_dtype(self):
        mask = make_causal_mask(3)
        assert mask.dtype in (np.float32, np.float64, np.int32, np.int64,
                               np.int8, np.uint8)

    @pytest.mark.parametrize("n", [2, 4, 8, 16])
    def test_causal_mask_diagonal_all_ones(self, n):
        mask = make_causal_mask(n)
        np.testing.assert_array_equal(np.diag(mask), np.ones(n))


# ── 2. Causal Attention ────────────────────────────────────────────────────────

class TestCausalAttention:

    def test_causal_attention_shape(self):
        np.random.seed(0)
        seq_len, d_k = 6, 8
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_k)
        mask = make_causal_mask(seq_len)
        out, _ = causal_attention(Q, K, V, mask=mask)
        assert out.shape == (seq_len, d_k)

    def test_causal_attention_no_future_leak(self):
        """Output at position 0 must not depend on V at position 1."""
        np.random.seed(1)
        seq_len, d_k = 4, 8
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V_base = np.random.randn(seq_len, d_k)
        mask = make_causal_mask(seq_len)

        out_base, _ = causal_attention(Q, K, V_base, mask=mask)

        V_perturbed = V_base.copy()
        V_perturbed[1] = 1e6 * np.ones(d_k)   # huge value at position 1
        out_perturbed, _ = causal_attention(Q, K, V_perturbed, mask=mask)

        # Position 0 should be identical: it cannot see position 1
        np.testing.assert_allclose(out_base[0], out_perturbed[0], atol=1e-4)

    def test_causal_attention_softmax_sum_to_one(self):
        """Each row of masked attention weights should sum to 1."""
        np.random.seed(2)
        seq_len, d_k = 5, 16
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_k)
        mask = make_causal_mask(seq_len)
        _, weights = causal_attention(Q, K, V, mask=mask)
        np.testing.assert_allclose(weights.sum(axis=-1), np.ones(seq_len), atol=1e-6)

    def test_causal_attention_single_token(self):
        """A single-token sequence should work without errors."""
        Q = np.array([[1.0, 0.0]])
        K = np.array([[1.0, 0.0]])
        V = np.array([[3.0, 4.0]])
        mask = make_causal_mask(1)
        out, weights = causal_attention(Q, K, V, mask=mask)
        assert out.shape == (1, 2)
        np.testing.assert_allclose(weights, [[1.0]], atol=1e-6)

    def test_causal_attention_values_bounded(self):
        """Output must be finite (no NaN or Inf)."""
        np.random.seed(3)
        seq_len, d_k = 8, 16
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_k)
        mask = make_causal_mask(seq_len)
        out, weights = causal_attention(Q, K, V, mask=mask)
        assert np.all(np.isfinite(out))
        assert np.all(np.isfinite(weights))

    def test_causal_attention_upper_weights_near_zero(self):
        """Weights above the diagonal should be effectively zero (masked out)."""
        np.random.seed(4)
        n, d_k = 6, 8
        Q = np.random.randn(n, d_k)
        K = np.random.randn(n, d_k)
        V = np.random.randn(n, d_k)
        mask = make_causal_mask(n)
        _, weights = causal_attention(Q, K, V, mask=mask)
        upper = weights[np.triu_indices(n, k=1)]
        assert np.all(upper < 1e-6)


# ── 3. BPE ────────────────────────────────────────────────────────────────────

class TestBPE:

    def test_bpe_get_pairs(self, simple_vocab):
        """get_pairs should return a dict of adjacent symbol pairs."""
        pairs = get_pairs(simple_vocab)
        assert isinstance(pairs, dict)
        # 'l','o' co-occur in 'low' (5) and 'lower' (2) → count 7
        assert ("l", "o") in pairs
        assert pairs[("l", "o")] == 7

    def test_bpe_merge(self, simple_vocab):
        """After merging ('e','s'), that pair should no longer appear as adjacent."""
        pair = ("e", "s")
        new_vocab = merge_vocab(pair, simple_vocab)
        new_pairs = get_pairs(new_vocab)
        assert pair not in new_pairs

    def test_bpe_reduces_vocab_size(self, simple_vocab):
        """Repeated merges should decrease total token count."""
        tokens_before = count_tokens(simple_vocab)
        vocab = simple_vocab
        for _ in range(3):
            pairs = get_pairs(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = merge_vocab(best, vocab)
        tokens_after = count_tokens(vocab)
        assert tokens_after < tokens_before

    def test_bpe_empty_input(self):
        """An empty vocab should not crash get_pairs."""
        pairs = get_pairs({})
        assert pairs == {}

    def test_bpe_single_char_words_no_pairs(self):
        """Words of length 1 have no adjacent pairs."""
        vocab = {("a",): 10, ("b",): 5}
        pairs = get_pairs(vocab)
        assert len(pairs) == 0


# ── 4. Layer Norm & GELU ──────────────────────────────────────────────────────

class TestLayerNormAndGELU:

    def test_layer_norm_output_mean_near_zero(self):
        np.random.seed(0)
        x = np.random.randn(6, 32) * 10 + 7
        normed = layer_norm(x)
        np.testing.assert_allclose(normed.mean(axis=-1), np.zeros(6), atol=1e-5)

    def test_layer_norm_output_std_near_one(self):
        np.random.seed(1)
        x = np.random.randn(6, 32) * 5 - 3
        normed = layer_norm(x)
        np.testing.assert_allclose(normed.std(axis=-1), np.ones(6), atol=1e-4)

    def test_gelu_positive_region(self):
        """For large positive x, GELU(x) ≈ x."""
        x = np.array([5.0, 10.0, 20.0])
        np.testing.assert_allclose(gelu(x), x, rtol=1e-3)

    def test_gelu_negative_suppressed(self):
        """GELU(-10) should be very close to 0."""
        result = gelu(np.array([-10.0]))
        assert abs(result[0]) < 1e-4

    def test_gelu_zero(self):
        """GELU(0) = 0."""
        assert abs(gelu(np.array([0.0]))[0]) < 1e-7

    @pytest.mark.parametrize("dim", [8, 16, 64])
    def test_layer_norm_shape_preserved(self, dim):
        x = np.random.randn(4, dim)
        assert layer_norm(x).shape == (4, dim)

    def test_gelu_monotone_positive(self):
        """GELU should be monotonically increasing for large positive inputs."""
        xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ys = gelu(xs)
        assert np.all(np.diff(ys) > 0)


# ── 5. Temperature Sampling ───────────────────────────────────────────────────

class TestTemperatureSampling:

    def test_greedy_decode_argmax(self):
        """Temperature=0 should always return the argmax."""
        logits = np.array([0.1, 0.5, 3.0, 1.2, 0.3])
        for _ in range(20):
            idx = temperature_sample(logits, temperature=0)
            assert idx == 2

    def test_temperature_high_makes_uniform(self):
        """Very high temperature → near-uniform distribution over vocab."""
        np.random.seed(5)
        logits = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
        counts = np.zeros(5)
        n_samples = 5000
        for _ in range(n_samples):
            idx = temperature_sample(logits, temperature=1000.0)
            counts[idx] += 1
        probs = counts / n_samples
        # Each should be close to 0.2; allow ±0.05 margin
        assert np.all(np.abs(probs - 0.2) < 0.05)

    def test_temperature_low_concentrates(self):
        """Very low temperature → concentrated on argmax."""
        np.random.seed(6)
        logits = np.array([0.1, 0.2, 5.0, 0.3])
        counts = np.zeros(4, dtype=int)
        n_samples = 1000
        for _ in range(n_samples):
            idx = temperature_sample(logits, temperature=0.01)
            counts[idx] += 1
        # argmax token should dominate
        assert counts[2] > 990

    def test_sampling_output_is_valid_index(self):
        """Output index must be within [0, vocab_size)."""
        np.random.seed(7)
        vocab_size = 10
        logits = np.random.randn(vocab_size)
        for _ in range(100):
            idx = temperature_sample(logits, temperature=1.0)
            assert 0 <= idx < vocab_size

    @pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
    def test_sampling_probs_sum_to_one(self, temperature):
        """Internal probability distribution must sum to 1."""
        logits = np.array([1.0, 2.0, 3.0])
        scaled = logits / temperature
        probs = softmax(scaled)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_temperature_one_matches_raw_softmax(self):
        """At temperature=1, sampling probs equal raw softmax."""
        logits = np.array([0.5, 1.5, 2.5, 0.0])
        expected = softmax(logits)
        actual = softmax(logits / 1.0)
        np.testing.assert_allclose(actual, expected, atol=1e-9)


# ── 6. Scaling Laws ───────────────────────────────────────────────────────────

class TestScalingLaws:

    def _power_law_loss(self, compute, alpha=-0.05, beta=2.5):
        """Synthetic Chinchilla-style loss: L = beta * C^alpha."""
        return beta * np.power(compute, alpha)

    def test_power_law_log_linear(self):
        """log(loss) vs log(compute) should have a negative slope."""
        compute = np.logspace(6, 12, 20)           # 1e6 … 1e12
        loss = self._power_law_loss(compute)
        log_c = np.log(compute)
        log_l = np.log(loss)
        # Fit a line to log-log space
        slope, _ = np.polyfit(log_c, log_l, 1)
        assert slope < 0, f"Expected negative slope, got {slope:.4f}"

    def test_power_law_fit(self):
        """Fitted exponent in log-log space should be negative."""
        compute = np.logspace(6, 12, 50)
        alpha_true = -0.07
        loss = 3.0 * np.power(compute, alpha_true)
        slope, _ = np.polyfit(np.log(compute), np.log(loss), 1)
        assert slope < 0

    def test_scaling_larger_compute_lower_loss(self):
        """Bigger compute → lower predicted loss."""
        c_small = 1e8
        c_large = 1e12
        loss_small = self._power_law_loss(c_small)
        loss_large = self._power_law_loss(c_large)
        assert loss_large < loss_small

    @pytest.mark.parametrize("alpha", [-0.03, -0.05, -0.10])
    def test_power_law_negative_exponent(self, alpha):
        """For any negative alpha, more compute means less loss."""
        compute = np.array([1e9, 1e10, 1e11])
        loss = 2.0 * np.power(compute, alpha)
        assert np.all(np.diff(loss) < 0), \
            f"Loss should decrease with compute for alpha={alpha}"
