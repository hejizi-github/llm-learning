"""
tests/test_llama.py
节点12：LLaMA 与开源爆炸（2023）
测试 LoRA、量化、Self-Instruct 格式的核心实现。
"""

import numpy as np
import pytest


# ── 复用 notebook 中的实现 ────────────────────────────────────────────────────

class LoRALayer:
    def __init__(self, d_in: int, d_out: int, rank: int, alpha: float = None):
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.alpha = alpha if alpha is not None else 2 * rank
        self.scaling = self.alpha / self.rank
        self.W0 = np.random.randn(d_out, d_in) * 0.02
        self.A = np.random.randn(rank, d_in) * 0.02
        self.B = np.zeros((d_out, rank))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.W0 @ x + self.scaling * (self.B @ (self.A @ x))

    def delta_W(self) -> np.ndarray:
        return self.scaling * (self.B @ self.A)

    def num_trainable_params(self) -> int:
        return self.A.size + self.B.size

    def num_full_params(self) -> int:
        return self.W0.size


def quantize_to_nbit(weights: np.ndarray, bits: int):
    n_levels = 2 ** bits
    w_min, w_max = weights.min(), weights.max()
    scale = (w_max - w_min) / (n_levels - 1)
    quantized = np.round((weights - w_min) / scale).astype(int)
    quantized = np.clip(quantized, 0, n_levels - 1)
    dequantized = quantized * scale + w_min
    return quantized, dequantized, scale, w_min


def format_alpaca_prompt(sample: dict) -> str:
    if sample.get('input', ''):
        return (
            f"### 指令:\n{sample['instruction']}\n\n"
            f"### 输入:\n{sample['input']}\n\n"
            f"### 输出:\n{sample['output']}"
        )
    else:
        return (
            f"### 指令:\n{sample['instruction']}\n\n"
            f"### 输出:\n{sample['output']}"
        )


def low_rank_approx(W: np.ndarray, rank: int) -> np.ndarray:
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    s_trunc = np.zeros_like(s)
    s_trunc[:rank] = s[:rank]
    return U @ np.diag(s_trunc) @ Vt


# ── LoRA 层形状测试 ───────────────────────────────────────────────────────────

class TestLoRALayerShape:
    def test_output_shape_1d(self):
        layer = LoRALayer(d_in=64, d_out=128, rank=4)
        x = np.random.randn(64)
        out = layer.forward(x)
        assert out.shape == (128,)

    def test_w0_shape(self):
        layer = LoRALayer(d_in=32, d_out=64, rank=8)
        assert layer.W0.shape == (64, 32)

    def test_A_shape(self):
        layer = LoRALayer(d_in=32, d_out=64, rank=8)
        assert layer.A.shape == (8, 32)

    def test_B_shape(self):
        layer = LoRALayer(d_in=32, d_out=64, rank=8)
        assert layer.B.shape == (64, 8)

    def test_delta_W_shape(self):
        layer = LoRALayer(d_in=32, d_out=64, rank=4)
        dw = layer.delta_W()
        assert dw.shape == (64, 32)

    def test_square_output_shape(self):
        layer = LoRALayer(d_in=256, d_out=256, rank=16)
        x = np.ones(256)
        out = layer.forward(x)
        assert out.shape == (256,)


# ── LoRA 参数量测试 ───────────────────────────────────────────────────────────

class TestLoRAParamCount:
    def test_trainable_params_formula(self):
        d_in, d_out, rank = 64, 64, 8
        layer = LoRALayer(d_in=d_in, d_out=d_out, rank=rank)
        expected = rank * d_in + d_out * rank  # A + B
        assert layer.num_trainable_params() == expected

    def test_full_params(self):
        d_in, d_out = 64, 128
        layer = LoRALayer(d_in=d_in, d_out=d_out, rank=4)
        assert layer.num_full_params() == d_in * d_out

    def test_compression_ratio_large(self):
        """rank << d 时，压缩比应很大。"""
        d, rank = 1024, 8
        layer = LoRALayer(d_in=d, d_out=d, rank=rank)
        ratio = layer.num_full_params() / layer.num_trainable_params()
        assert ratio > 50  # 实际 1024*1024 / (2*1024*8) = 64

    def test_trainable_less_than_full(self):
        layer = LoRALayer(d_in=256, d_out=256, rank=16)
        assert layer.num_trainable_params() < layer.num_full_params()


# ── LoRA 初始化测试 ───────────────────────────────────────────────────────────

class TestLoRAInitialization:
    def test_B_initialized_to_zero(self):
        layer = LoRALayer(d_in=64, d_out=64, rank=8)
        assert np.allclose(layer.B, 0.0)

    def test_delta_W_zero_at_init(self):
        """B=0 时 ΔW 应为 0。"""
        layer = LoRALayer(d_in=64, d_out=64, rank=8)
        dw = layer.delta_W()
        assert np.allclose(dw, 0.0)

    def test_forward_equals_w0_at_init(self):
        """B=0 时 forward 输出等于 W0@x。"""
        np.random.seed(0)
        layer = LoRALayer(d_in=64, d_out=64, rank=8)
        x = np.random.randn(64)
        out = layer.forward(x)
        expected = layer.W0 @ x
        assert np.allclose(out, expected)

    def test_A_not_zero_at_init(self):
        """A 应随机初始化（非全零）。"""
        layer = LoRALayer(d_in=64, d_out=64, rank=8)
        assert not np.allclose(layer.A, 0.0)


# ── LoRA 缩放因子测试 ─────────────────────────────────────────────────────────

class TestLoRAScaling:
    def test_default_alpha_is_2r(self):
        layer = LoRALayer(d_in=32, d_out=32, rank=8)
        assert layer.alpha == 16  # 2 * rank

    def test_custom_alpha(self):
        layer = LoRALayer(d_in=32, d_out=32, rank=8, alpha=32)
        assert layer.alpha == 32

    def test_scaling_formula(self):
        layer = LoRALayer(d_in=32, d_out=32, rank=8, alpha=16)
        assert layer.scaling == pytest.approx(2.0)

    def test_scaling_applied_in_delta_W(self):
        np.random.seed(1)
        layer = LoRALayer(d_in=16, d_out=16, rank=2, alpha=4)  # scaling=2
        layer.B = np.ones((16, 2))
        layer.A = np.ones((2, 16))
        dw = layer.delta_W()
        # ΔW = scaling * B @ A = 2 * (all-ones 16×2 @ all-ones 2×16) = 2 * 2 * all-ones 16×16
        assert np.allclose(dw, 4.0)


# ── LoRA 合并权重测试 ─────────────────────────────────────────────────────────

class TestLoRAMerge:
    def test_merged_weight_forward_equals_lora_forward(self):
        np.random.seed(42)
        layer = LoRALayer(d_in=32, d_out=32, rank=4)
        layer.B = np.random.randn(32, 4) * 0.1
        x = np.random.randn(32)
        out_lora = layer.forward(x)
        W_merged = layer.W0 + layer.delta_W()
        out_merged = W_merged @ x
        assert np.allclose(out_lora, out_merged, atol=1e-10)

    def test_different_B_gives_different_output(self):
        np.random.seed(0)
        layer = LoRALayer(d_in=16, d_out=16, rank=4)
        x = np.random.randn(16)
        out1 = layer.forward(x)
        layer.B = np.random.randn(16, 4) * 0.5
        out2 = layer.forward(x)
        assert not np.allclose(out1, out2)


# ── 低秩近似测试 ─────────────────────────────────────────────────────────────

class TestLowRankApprox:
    def test_exact_recovery_at_true_rank(self):
        """rank=4 构造的矩阵用 rank=4 近似应误差为 0。"""
        np.random.seed(42)
        d = 64
        B = np.random.randn(d, 4)
        A = np.random.randn(4, d)
        W = B @ A  # 真实 rank = 4
        approx = low_rank_approx(W, 4)
        assert np.allclose(W, approx, atol=1e-8)

    def test_error_decreases_with_rank(self):
        """rank 越大，近似误差越小。"""
        np.random.seed(0)
        d = 64
        W = np.random.randn(d, d)
        errors = []
        for r in [1, 4, 16, 32]:
            approx = low_rank_approx(W, r)
            err = np.linalg.norm(W - approx, 'fro')
            errors.append(err)
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1]

    def test_rank1_approx_shape(self):
        W = np.random.randn(8, 8)
        approx = low_rank_approx(W, 1)
        assert approx.shape == W.shape

    def test_full_rank_approx_equals_original(self):
        np.random.seed(5)
        d = 16
        W = np.random.randn(d, d)
        approx = low_rank_approx(W, d)
        assert np.allclose(W, approx, atol=1e-8)


# ── 量化测试 ─────────────────────────────────────────────────────────────────

class TestQuantization:
    def test_output_shape_preserved(self):
        weights = np.random.randn(100)
        q, dq, _, _ = quantize_to_nbit(weights, 8)
        assert q.shape == weights.shape
        assert dq.shape == weights.shape

    def test_8bit_quantized_range(self):
        weights = np.random.randn(500)
        q, _, _, _ = quantize_to_nbit(weights, 8)
        assert q.min() >= 0
        assert q.max() <= 255

    def test_4bit_quantized_range(self):
        weights = np.random.randn(500)
        q, _, _, _ = quantize_to_nbit(weights, 4)
        assert q.min() >= 0
        assert q.max() <= 15

    def test_4bit_more_error_than_8bit(self):
        np.random.seed(0)
        weights = np.random.randn(1000)
        _, dq8, _, _ = quantize_to_nbit(weights, 8)
        _, dq4, _, _ = quantize_to_nbit(weights, 4)
        err8 = np.abs(weights - dq8).mean()
        err4 = np.abs(weights - dq4).mean()
        assert err4 > err8

    def test_dequantized_within_original_range(self):
        np.random.seed(1)
        weights = np.random.randn(200)
        _, dq, _, _ = quantize_to_nbit(weights, 8)
        assert dq.min() >= weights.min() - 1e-9
        assert dq.max() <= weights.max() + 1e-9

    def test_constant_weights_no_error(self):
        """常数权重量化后应完美恢复。"""
        weights = np.ones(100) * 3.14
        _, dq, _, _ = quantize_to_nbit(weights, 4)
        assert np.allclose(dq, weights)


# ── Self-Instruct / Alpaca 格式测试 ──────────────────────────────────────────

class TestAlpacaFormat:
    def test_no_input_format(self):
        sample = {'instruction': '写一首诗', 'input': '', 'output': '春眠不觉晓'}
        prompt = format_alpaca_prompt(sample)
        assert '### 指令:' in prompt
        assert '写一首诗' in prompt
        assert '### 输出:' in prompt
        assert '春眠不觉晓' in prompt
        assert '### 输入:' not in prompt

    def test_with_input_format(self):
        sample = {
            'instruction': '翻译以下句子',
            'input': 'Hello World',
            'output': '你好，世界'
        }
        prompt = format_alpaca_prompt(sample)
        assert '### 指令:' in prompt
        assert '### 输入:' in prompt
        assert 'Hello World' in prompt
        assert '### 输出:' in prompt
        assert '你好，世界' in prompt

    def test_format_order(self):
        """格式顺序：指令 → 输入 → 输出。"""
        sample = {'instruction': 'A', 'input': 'B', 'output': 'C'}
        prompt = format_alpaca_prompt(sample)
        idx_inst = prompt.index('### 指令:')
        idx_inp = prompt.index('### 输入:')
        idx_out = prompt.index('### 输出:')
        assert idx_inst < idx_inp < idx_out

    def test_instruction_in_prompt(self):
        instruction = '用三句话解释量子力学'
        sample = {'instruction': instruction, 'input': '', 'output': '...'}
        prompt = format_alpaca_prompt(sample)
        assert instruction in prompt

    def test_output_at_end(self):
        sample = {'instruction': 'Q', 'input': '', 'output': 'final_answer'}
        prompt = format_alpaca_prompt(sample)
        assert prompt.endswith('final_answer')


# ── 参数压缩率计算测试 ────────────────────────────────────────────────────────

class TestCompressionRatio:
    @pytest.mark.parametrize("d,r,expected_ratio", [
        (1000, 8, 62.5),
        (512, 16, 16.0),
        (4096, 8, 256.0),
    ])
    def test_compression_formula(self, d, r, expected_ratio):
        full = d * d
        lora = 2 * d * r
        ratio = full / lora
        assert ratio == pytest.approx(expected_ratio)

    def test_rank_1_max_compression(self):
        d = 100
        full = d * d
        lora_r1 = 2 * d * 1
        lora_r8 = 2 * d * 8
        assert full / lora_r1 > full / lora_r8
