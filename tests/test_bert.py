"""Tests for BERT (2018) node — Masked LM, NSP input, BERT embeddings, Encoder, Classification."""
import numpy as np
import pytest


# ── Shared vocab / constants ───────────────────────────────────────────────────

VOCAB = {
    "[CLS]": 0, "[SEP]": 1, "[MASK]": 2, "[PAD]": 3,
    "the": 4, "cat": 5, "sat": 6, "on": 7, "mat": 8,
    "bank": 9, "river": 10, "money": 11, "loan": 12,
    "near": 13, "deposited": 14, "fish": 15,
}
VOCAB_SIZE = len(VOCAB)
MASK_ID = VOCAB["[MASK]"]
HIDDEN = 8


# ── Helpers (duplicated from notebook for test isolation) ──────────────────────

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def mask_tokens(token_ids, vocab_size, mask_id=2,
                mask_rate=0.15, mask_frac=0.8, random_frac=0.1, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    ids = np.array(token_ids, dtype=np.int32)
    labels = np.full_like(ids, -100)
    n = len(ids)
    num_mask = max(1, int(n * mask_rate))
    selected = rng.choice(n, size=num_mask, replace=False)
    masked_ids = ids.copy()
    for pos in selected:
        labels[pos] = ids[pos]
        r = rng.random()
        if r < mask_frac:
            masked_ids[pos] = mask_id
        elif r < mask_frac + random_frac:
            masked_ids[pos] = rng.integers(0, vocab_size)
    mask_flags = (labels != -100)
    return masked_ids, labels, mask_flags


def make_bert_input(tokens_a, tokens_b, vocab):
    CLS, SEP = vocab["[CLS]"], vocab["[SEP]"]
    token_ids = [CLS] + tokens_a + [SEP] + tokens_b + [SEP]
    segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
    position_ids = list(range(len(token_ids)))
    return (
        np.array(token_ids, dtype=np.int32),
        np.array(segment_ids, dtype=np.int32),
        np.array(position_ids, dtype=np.int32),
    )


class BERTEmbedding:
    def __init__(self, vocab_size, hidden, max_len=512, seed=0):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(1.0 / hidden)
        self.token_emb = rng.standard_normal((vocab_size, hidden)) * scale
        self.position_emb = rng.standard_normal((max_len, hidden)) * scale
        self.segment_emb = rng.standard_normal((2, hidden)) * scale
        self.hidden = hidden

    def forward(self, token_ids, segment_ids, position_ids):
        return (
            self.token_emb[token_ids]
            + self.position_emb[position_ids]
            + self.segment_emb[segment_ids]
        )


def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    w = softmax(scores)
    return w @ V, w


class MHA:
    def __init__(self, d_model, num_heads, seed=0):
        assert d_model % num_heads == 0
        rng = np.random.default_rng(seed)
        self.h = num_heads
        dk = d_model // num_heads
        self.WQ = [rng.standard_normal((d_model, dk)) * 0.1 for _ in range(num_heads)]
        self.WK = [rng.standard_normal((d_model, dk)) * 0.1 for _ in range(num_heads)]
        self.WV = [rng.standard_normal((d_model, dk)) * 0.1 for _ in range(num_heads)]
        self.WO = rng.standard_normal((d_model, d_model)) * 0.1

    def forward(self, X):
        heads = []
        for i in range(self.h):
            out, _ = attention(X @ self.WQ[i], X @ self.WK[i], X @ self.WV[i])
            heads.append(out)
        return np.concatenate(heads, axis=-1) @ self.WO


class FFN:
    def __init__(self, d_model, d_ff, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((d_model, d_ff)) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.standard_normal((d_ff, d_model)) * 0.1
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        return np.maximum(0, x @ self.W1 + self.b1) @ self.W2 + self.b2


class EncoderBlock:
    def __init__(self, d_model, num_heads, d_ff, seed=0):
        self.mha = MHA(d_model, num_heads, seed=seed)
        self.ffn = FFN(d_model, d_ff, seed=seed + 1)

    def forward(self, X):
        X = layer_norm(X + self.mha.forward(X))
        X = layer_norm(X + self.ffn.forward(X))
        return X


class BERTEncoder:
    def __init__(self, d_model, num_heads, d_ff, num_layers, seed=0):
        self.layers = [
            EncoderBlock(d_model, num_heads, d_ff, seed=seed + i)
            for i in range(num_layers)
        ]

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X


class ClassificationHead:
    def __init__(self, hidden, num_classes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((hidden, num_classes)) * 0.1
        self.b = np.zeros(num_classes)

    def forward(self, cls_vec):
        return cls_vec @ self.W + self.b


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_sentence():
    return [VOCAB["the"], VOCAB["cat"], VOCAB["sat"], VOCAB["on"], VOCAB["mat"],
            VOCAB["bank"]]


@pytest.fixture
def bert_input():
    sent_a = [VOCAB["the"], VOCAB["cat"], VOCAB["sat"]]
    sent_b = [VOCAB["bank"], VOCAB["near"], VOCAB["river"]]
    return make_bert_input(sent_a, sent_b, VOCAB)


@pytest.fixture
def emb_layer():
    return BERTEmbedding(vocab_size=VOCAB_SIZE, hidden=HIDDEN, seed=1)


# ── MLM Tests ─────────────────────────────────────────────────────────────────

class TestMaskedLM:

    def test_mask_rate_approximately_15_percent(self, sample_sentence):
        rng = np.random.default_rng(0)
        n = 100  # longer sequence for meaningful rate estimation
        long_ids = (list(range(VOCAB_SIZE)) * (n // VOCAB_SIZE + 1))[:n]
        _, _, mask_flags = mask_tokens(long_ids, VOCAB_SIZE, rng=rng)
        rate = mask_flags.sum() / n
        assert 0.10 <= rate <= 0.20, f"Expected ~15% masking, got {rate:.2%}"

    def test_shape_preserved(self, sample_sentence):
        _, labels, _ = mask_tokens(sample_sentence, VOCAB_SIZE)
        assert len(labels) == len(sample_sentence)

    def test_labels_are_minus100_at_unmasked(self, sample_sentence):
        rng = np.random.default_rng(42)
        _, labels, mask_flags = mask_tokens(sample_sentence, VOCAB_SIZE, rng=rng)
        assert np.all(labels[~mask_flags] == -100)

    def test_labels_store_original_at_masked(self, sample_sentence):
        rng = np.random.default_rng(42)
        masked_ids, labels, mask_flags = mask_tokens(sample_sentence, VOCAB_SIZE, rng=rng)
        for i in np.where(mask_flags)[0]:
            assert labels[i] == sample_sentence[i], \
                f"Position {i}: label={labels[i]}, original={sample_sentence[i]}"

    def test_at_least_one_position_masked(self, sample_sentence):
        _, _, mask_flags = mask_tokens(sample_sentence, VOCAB_SIZE)
        assert mask_flags.sum() >= 1

    def test_80_percent_of_selected_use_mask_token(self):
        rng = np.random.default_rng(7)
        n = 200
        ids = list(range(4, min(4 + n, VOCAB_SIZE))) * (n // (VOCAB_SIZE - 4) + 1)
        ids = ids[:n]
        masked_ids, _, mask_flags = mask_tokens(ids, VOCAB_SIZE, rng=rng)
        selected_indices = np.where(mask_flags)[0]
        num_selected = len(selected_indices)
        num_mask_token = sum(1 for i in selected_indices if masked_ids[i] == MASK_ID)
        frac = num_mask_token / num_selected
        # Expect ~80%, allow wide margin since sample is small
        assert 0.50 <= frac <= 1.00, f"Expected ~80% [MASK], got {frac:.2%}"

    @pytest.mark.parametrize("mask_rate", [0.10, 0.15, 0.20])
    def test_parametrized_mask_rate(self, mask_rate):
        rng = np.random.default_rng(0)
        n = 100
        ids = [4] * n
        _, _, mask_flags = mask_tokens(ids, VOCAB_SIZE, mask_rate=mask_rate, rng=rng)
        rate = mask_flags.sum() / n
        assert abs(rate - mask_rate) <= 0.05


# ── BERT Input Construction Tests ─────────────────────────────────────────────

class TestBERTInput:

    def test_starts_with_cls(self, bert_input):
        token_ids, _, _ = bert_input
        assert token_ids[0] == VOCAB["[CLS]"]

    def test_ends_with_sep(self, bert_input):
        token_ids, _, _ = bert_input
        assert token_ids[-1] == VOCAB["[SEP]"]

    def test_length_formula(self):
        sent_a = [VOCAB["the"], VOCAB["cat"]]
        sent_b = [VOCAB["bank"], VOCAB["river"]]
        token_ids, segment_ids, position_ids = make_bert_input(sent_a, sent_b, VOCAB)
        expected = 1 + len(sent_a) + 1 + len(sent_b) + 1
        assert len(token_ids) == expected

    def test_segment_ids_sent_a_are_zero(self, bert_input):
        token_ids, segment_ids, _ = bert_input
        # [CLS] + 3 tokens + [SEP] -> positions 0..4 are segment 0
        assert np.all(segment_ids[:5] == 0)

    def test_segment_ids_sent_b_are_one(self, bert_input):
        token_ids, segment_ids, _ = bert_input
        # positions 5..8 are segment 1
        assert np.all(segment_ids[5:] == 1)

    def test_position_ids_are_sequential(self, bert_input):
        _, _, position_ids = bert_input
        expected = np.arange(len(position_ids), dtype=np.int32)
        np.testing.assert_array_equal(position_ids, expected)

    def test_sep_separates_sentences(self, bert_input):
        token_ids, segment_ids, _ = bert_input
        # Find [SEP] positions
        sep_positions = np.where(token_ids == VOCAB["[SEP]"])[0]
        assert len(sep_positions) == 2


# ── BERT Embedding Tests ───────────────────────────────────────────────────────

class TestBERTEmbedding:

    def test_token_emb_shape(self, emb_layer):
        assert emb_layer.token_emb.shape == (VOCAB_SIZE, HIDDEN)

    def test_position_emb_shape(self, emb_layer):
        assert emb_layer.position_emb.shape == (512, HIDDEN)

    def test_segment_emb_shape(self, emb_layer):
        assert emb_layer.segment_emb.shape == (2, HIDDEN)

    def test_output_shape(self, bert_input, emb_layer):
        token_ids, segment_ids, position_ids = bert_input
        X = emb_layer.forward(token_ids, segment_ids, position_ids)
        assert X.shape == (len(token_ids), HIDDEN)

    def test_different_segments_give_different_embeddings(self, emb_layer):
        ids = np.array([VOCAB["the"]], dtype=np.int32)
        pos = np.array([0], dtype=np.int32)
        seg_a = np.array([0], dtype=np.int32)
        seg_b = np.array([1], dtype=np.int32)
        xa = emb_layer.forward(ids, seg_a, pos)
        xb = emb_layer.forward(ids, seg_b, pos)
        assert not np.allclose(xa, xb)


# ── BERT Encoder Tests ────────────────────────────────────────────────────────

class TestBERTEncoder:

    def test_output_shape_preserved(self, bert_input, emb_layer):
        token_ids, segment_ids, position_ids = bert_input
        X = emb_layer.forward(token_ids, segment_ids, position_ids)
        encoder = BERTEncoder(d_model=HIDDEN, num_heads=2, d_ff=16, num_layers=2, seed=0)
        out = encoder.forward(X)
        assert out.shape == X.shape

    def test_layer_norm_applied(self, bert_input, emb_layer):
        token_ids, segment_ids, position_ids = bert_input
        X = emb_layer.forward(token_ids, segment_ids, position_ids)
        encoder = BERTEncoder(d_model=HIDDEN, num_heads=2, d_ff=16, num_layers=1, seed=0)
        out = encoder.forward(X)
        np.testing.assert_allclose(out.mean(axis=-1), np.zeros(len(token_ids)), atol=1e-5)
        np.testing.assert_allclose(out.std(axis=-1), np.ones(len(token_ids)), atol=1e-5)


# ── Classification Head Tests ─────────────────────────────────────────────────

class TestClassificationHead:

    def test_output_shape_binary(self):
        head = ClassificationHead(hidden=HIDDEN, num_classes=2, seed=0)
        cls_vec = np.random.randn(HIDDEN)
        logits = head.forward(cls_vec)
        assert logits.shape == (2,)

    def test_output_shape_multiclass(self):
        head = ClassificationHead(hidden=HIDDEN, num_classes=5, seed=0)
        cls_vec = np.random.randn(HIDDEN)
        logits = head.forward(cls_vec)
        assert logits.shape == (5,)

    def test_softmax_probs_sum_to_one(self):
        head = ClassificationHead(hidden=HIDDEN, num_classes=3, seed=0)
        cls_vec = np.random.randn(HIDDEN)
        logits = head.forward(cls_vec)
        probs = softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-6
