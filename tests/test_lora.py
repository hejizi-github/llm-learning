"""
tests/test_lora.py — 节点22 LoRA 测试

覆盖：参数计数 | 初始ΔW为零 | 前向传播形状 | 梯度更新 | 压缩比
"""
import numpy as np
import pytest
import os


# ─── Helpers (mirror notebook implementations) ────────────────────────────

def lora_param_count(d, k, r):
    """LoRA 可训练参数量：d×r + r×k"""
    return d * r + r * k


def full_param_count(d, k):
    """全量微调参数量：d×k"""
    return d * k


def make_lora(d, k, r, seed=42):
    """初始化 LoRA 参数：B=zeros, A=random"""
    rng = np.random.RandomState(seed)
    B = np.zeros((d, r))
    A = rng.randn(r, k) * 0.02
    return B, A


def lora_forward(W_pretrained, B, A, x):
    """LoRA 前向传播：(W + B@A) @ x"""
    W_eff = W_pretrained + B @ A
    return W_eff @ x


def lora_grad_step(W_pretrained, B, A, X, Y, lr=0.01):
    """对 A 和 B 各做一步梯度下降，返回更新后的 B, A"""
    W_eff = W_pretrained + B @ A
    Y_pred = W_eff @ X
    error = Y_pred - Y
    n = X.shape[1]
    grad_W = (2 / n) * error @ X.T
    grad_B = grad_W @ A.T
    grad_A = B.T @ grad_W
    B_new = B - lr * grad_B
    A_new = A - lr * grad_A
    return B_new, A_new


# ─── 1. 参数计数 ──────────────────────────────────────────────────────────

class TestLoRAParameterCount:
    def test_lora_fewer_params_than_full(self):
        """LoRA 参数量应小于全量微调"""
        d, k, r = 100, 100, 4
        assert lora_param_count(d, k, r) < full_param_count(d, k)

    def test_lora_parameter_count_formula(self):
        """验证参数量公式 d*r + r*k"""
        d, k, r = 50, 80, 3
        assert lora_param_count(d, k, r) == d * r + r * k

    def test_lora_parameter_count_various_ranks(self):
        """对不同 rank，LoRA 参数量均小于全量"""
        d, k = 200, 200
        for r in [1, 2, 4, 8, 16]:
            assert lora_param_count(d, k, r) < full_param_count(d, k), \
                f"rank={r} 时 LoRA 参数量应小于全量微调"

    def test_lora_param_count_for_small_r(self):
        """r < min(d,k)//2 时，LoRA 参数量严格小于全量"""
        d, k = 128, 256
        r = min(d, k) // 2 - 1
        assert lora_param_count(d, k, r) < full_param_count(d, k)

    def test_lora_param_count_specific_values(self):
        """手算验证：d=10, k=10, r=2 → 10*2+2*10=40 < 100"""
        assert lora_param_count(10, 10, 2) == 40
        assert full_param_count(10, 10) == 100
        assert lora_param_count(10, 10, 2) < full_param_count(10, 10)


# ─── 2. 初始 ΔW 为零 ─────────────────────────────────────────────────────

class TestLoRAInitialDeltaIsZero:
    def test_initial_delta_is_zero_matrix(self):
        """B=zeros, A=random → B@A = zero matrix"""
        B, A = make_lora(d=50, k=50, r=4)
        delta_W = B @ A
        assert np.allclose(delta_W, 0.0), "初始 ΔW 应为全零矩阵"

    def test_initial_delta_frobenius_norm_zero(self):
        """初始 ΔW 的 Frobenius 范数应为 0"""
        B, A = make_lora(d=100, k=100, r=8)
        assert np.linalg.norm(B @ A) == 0.0

    def test_initial_effective_weight_equals_pretrained(self):
        """初始时 W_eff = W_pretrained + B@A = W_pretrained"""
        np.random.seed(7)
        d, k, r = 30, 40, 4
        W_pretrained = np.random.randn(d, k)
        B, A = make_lora(d, k, r)
        W_eff = W_pretrained + B @ A
        assert np.allclose(W_eff, W_pretrained), \
            "初始时有效权重应等于预训练权重"

    def test_B_initialized_to_zeros(self):
        """B 矩阵应初始化为全零"""
        B, A = make_lora(d=20, k=20, r=3)
        assert np.all(B == 0.0), "B 应初始化为零矩阵"

    def test_A_initialized_nonzero(self):
        """A 矩阵应随机初始化（非零）"""
        B, A = make_lora(d=20, k=20, r=3)
        assert not np.all(A == 0.0), "A 应随机初始化（非全零）"

    def test_delta_zero_regardless_of_A(self):
        """无论 A 如何初始化，只要 B=0，ΔW 就是零矩阵"""
        np.random.seed(99)
        d, k, r = 15, 25, 5
        B = np.zeros((d, r))
        A = np.random.randn(r, k) * 10.0  # 大值 A
        assert np.allclose(B @ A, 0.0)


# ─── 3. 前向传播形状和输出 ────────────────────────────────────────────────

class TestLoRAForwardPass:
    @pytest.fixture
    def setup(self):
        np.random.seed(42)
        d, k, r = 20, 30, 4
        W_pretrained = np.random.randn(d, k)
        B, A = make_lora(d, k, r)
        # 训练 B 一步（让 B 不为零）
        B = np.random.randn(d, r) * 0.1
        x = np.random.randn(k, 5)  # batch of 5
        return W_pretrained, B, A, x, d, k, r

    def test_output_shape(self, setup):
        """W_eff @ x 的输出形状应为 (d, batch)"""
        W_pretrained, B, A, x, d, k, r = setup
        out = lora_forward(W_pretrained, B, A, x)
        assert out.shape == (d, x.shape[1]), \
            f"期望输出形状 ({d}, {x.shape[1]})，得到 {out.shape}"

    def test_effective_weight_shape(self, setup):
        """W_eff = W + B@A 形状应为 (d, k)"""
        W_pretrained, B, A, x, d, k, r = setup
        W_eff = W_pretrained + B @ A
        assert W_eff.shape == (d, k)

    def test_output_is_finite(self, setup):
        """前向传播输出不应含 NaN 或 Inf"""
        W_pretrained, B, A, x, d, k, r = setup
        out = lora_forward(W_pretrained, B, A, x)
        assert np.all(np.isfinite(out)), "前向传播输出不应含 NaN 或 Inf"

    def test_initial_lora_output_matches_pretrained(self):
        """B=zeros 时，LoRA 输出应与直接用 W_pretrained 相同"""
        np.random.seed(0)
        d, k, r = 10, 10, 4
        W_pretrained = np.random.randn(d, k)
        B, A = make_lora(d, k, r)
        x = np.random.randn(k, 3)
        out_lora = lora_forward(W_pretrained, B, A, x)
        out_pretrained = W_pretrained @ x
        assert np.allclose(out_lora, out_pretrained), \
            "初始 LoRA 输出应与预训练模型输出相同"

    def test_output_changes_after_B_update(self):
        """B 更新后，LoRA 输出应与预训练模型不同"""
        np.random.seed(1)
        d, k, r = 10, 10, 4
        W_pretrained = np.random.randn(d, k)
        B, A = make_lora(d, k, r)
        x = np.random.randn(k, 3)
        out_before = lora_forward(W_pretrained, B, A, x)
        # 手动给 B 一个非零值
        B[0, 0] = 1.0
        out_after = lora_forward(W_pretrained, B, A, x)
        assert not np.allclose(out_before, out_after), \
            "B 更新后输出应发生变化"


# ─── 4. 梯度更新 ─────────────────────────────────────────────────────────

class TestLoRAGradientUpdate:
    @pytest.fixture
    def regression_setup(self):
        np.random.seed(42)
        d, k, r = 15, 15, 3
        W_pretrained = np.random.randn(d, k)
        B, A = make_lora(d, k, r)
        X = np.random.randn(k, 50)
        # 目标：W_pretrained + 小扰动
        Y = (W_pretrained + np.random.randn(d, k) * 0.1) @ X
        return W_pretrained, B, A, X, Y, d, k, r

    def test_A_changes_after_gradient_step(self, regression_setup):
        """两步更新后，A 应发生变化（第一步 B 从零更新，第二步 grad_A 非零）"""
        W_pretrained, B, A, X, Y, d, k, r = regression_setup
        # Step 1: B moves away from zero (grad_B = grad_W @ A.T is non-zero)
        B_1, A_1 = lora_grad_step(W_pretrained, B, A, X, Y, lr=0.01)
        # Step 2: now B_1 != 0, so grad_A = B_1.T @ grad_W != 0 → A changes
        A_before_step2 = A_1.copy()
        B_2, A_2 = lora_grad_step(W_pretrained, B_1, A_1, X, Y, lr=0.01)
        assert not np.allclose(A_2, A_before_step2), "第二步更新后 A 应发生变化"

    def test_B_shape_unchanged_after_update(self, regression_setup):
        """梯度更新后，B 的形状不变"""
        W_pretrained, B, A, X, Y, d, k, r = regression_setup
        B_new, A_new = lora_grad_step(W_pretrained, B, A, X, Y, lr=0.01)
        assert B_new.shape == B.shape, f"B 形状应保持 {B.shape}"

    def test_A_shape_unchanged_after_update(self, regression_setup):
        """梯度更新后，A 的形状不变"""
        W_pretrained, B, A, X, Y, d, k, r = regression_setup
        B_new, A_new = lora_grad_step(W_pretrained, B, A, X, Y, lr=0.01)
        assert A_new.shape == A.shape, f"A 形状应保持 {A.shape}"

    def test_loss_decreases_over_steps(self, regression_setup):
        """多步梯度更新后，loss 应下降"""
        W_pretrained, B, A, X, Y, d, k, r = regression_setup
        B_curr, A_curr = B.copy(), A.copy()

        def compute_loss(B_, A_):
            W_eff = W_pretrained + B_ @ A_
            return float(np.mean((W_eff @ X - Y) ** 2))

        loss_init = compute_loss(B_curr, A_curr)

        for _ in range(100):
            B_curr, A_curr = lora_grad_step(W_pretrained, B_curr, A_curr, X, Y, lr=0.005)

        loss_final = compute_loss(B_curr, A_curr)
        assert loss_final < loss_init, \
            f"100步后 loss 应下降：初始 {loss_init:.4f} → 最终 {loss_final:.4f}"

    def test_W_pretrained_unchanged_after_update(self, regression_setup):
        """梯度更新中，预训练权重 W 不应改变"""
        W_pretrained, B, A, X, Y, d, k, r = regression_setup
        W_before = W_pretrained.copy()
        lora_grad_step(W_pretrained, B, A, X, Y, lr=0.01)
        assert np.allclose(W_pretrained, W_before), \
            "预训练权重 W 在 LoRA 更新中不应改变"


# ─── 5. 压缩比 ────────────────────────────────────────────────────────────

class TestLoRACompressionRatio:
    def test_compression_ratio_768_768_r4(self):
        """d=768, k=768, r=4 时压缩比应 > 48x"""
        d, k, r = 768, 768, 4
        ratio = full_param_count(d, k) / lora_param_count(d, k, r)
        assert ratio > 48, f"压缩比应 > 48x，实际: {ratio:.1f}x"

    def test_compression_ratio_increases_as_rank_decreases(self):
        """rank 越小，压缩比越大"""
        d, k = 100, 100
        ratios = [full_param_count(d, k) / lora_param_count(d, k, r)
                  for r in [1, 2, 4, 8, 16]]
        assert all(a > b for a, b in zip(ratios, ratios[1:])), \
            "rank 越小，压缩比应越大"

    def test_compression_ratio_gpt3_scale(self):
        """GPT-3 规模（d=k=12288），r=4 时压缩比应 > 1000x"""
        d, k, r = 12288, 12288, 4
        ratio = full_param_count(d, k) / lora_param_count(d, k, r)
        assert ratio > 1000, f"GPT-3 规模压缩比应 > 1000x，实际: {ratio:.1f}x"

    def test_lora_params_correct_formula_768(self):
        """d=768, k=768, r=4 时 LoRA 参数量 = 768*4 + 4*768 = 6144"""
        d, k, r = 768, 768, 4
        expected = 768 * 4 + 4 * 768  # = 6144
        assert lora_param_count(d, k, r) == expected

    def test_compression_ratio_positive(self):
        """压缩比应始终大于 1（LoRA 参数量 < 全量）"""
        for d, k, r in [(64, 64, 4), (128, 256, 8), (1024, 1024, 16)]:
            ratio = full_param_count(d, k) / lora_param_count(d, k, r)
            assert ratio > 1.0, f"d={d},k={k},r={r} 时压缩比应 > 1"


# ─── 6. Document Structure ────────────────────────────────────────────────

class TestDocumentStructure:
    """验证节点22文档和notebook文件存在且内容正确"""

    DOC_PATH = os.path.join(os.path.dirname(__file__), "../docs/22-lora-2021.md")
    NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "../notebooks/22-lora-2021.ipynb")

    def test_document_exists(self):
        assert os.path.exists(self.DOC_PATH), "docs/22-lora-2021.md 应存在"

    def test_notebook_exists(self):
        assert os.path.exists(self.NOTEBOOK_PATH), "notebooks/22-lora-2021.ipynb 应存在"

    def test_document_contains_lora(self):
        with open(self.DOC_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        assert "LoRA" in content

    def test_document_contains_low_rank(self):
        with open(self.DOC_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        assert "低秩" in content or "low-rank" in content.lower()

    def test_document_contains_delta_w(self):
        with open(self.DOC_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        assert "ΔW" in content or "delta_W" in content or "B @ A" in content

    def test_document_contains_citation(self):
        with open(self.DOC_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        assert "2106.09685" in content or "hu2022lora" in content

    def test_document_contains_gpt3_param_count(self):
        with open(self.DOC_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        assert "1750" in content, "文档应提及 GPT-3 的 1750 亿参数"

    def test_notebook_trailing_newline(self):
        with open(self.NOTEBOOK_PATH, "rb") as f:
            content = f.read()
        assert content.endswith(b"\n"), "notebook JSON 应以换行结尾"
