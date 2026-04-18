"""
tests/test_stable_diffusion.py — 节点18 Stable Diffusion / LDM 单元测试

覆盖：LinearAutoencoder, TextConditionedDenoiser, 隐空间 DDIM, 数学性质
"""
import numpy as np
import pytest

# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

PIXEL_DIM = 64
LATENT_DIM = 8
TEXT_DIM = 16
T = 50
BETA_START = 1e-4
BETA_END = 0.02


@pytest.fixture
def alpha_bar():
    betas = np.linspace(BETA_START, BETA_END, T)
    return np.cumprod(1.0 - betas)


@pytest.fixture
def autoencoder():
    np.random.seed(7)

    class LinearAutoencoder:
        def __init__(self, pixel_dim, latent_dim, lr=0.005):
            self.pixel_dim = pixel_dim
            self.latent_dim = latent_dim
            self.lr = lr
            self.W_enc = np.random.randn(latent_dim, pixel_dim) * 0.001
            self.W_dec = np.random.randn(pixel_dim, latent_dim) * 0.001

        def encode(self, x):
            return np.tanh(np.clip(self.W_enc @ x, -30, 30))

        def decode(self, z):
            return np.tanh(np.clip(self.W_dec @ z, -30, 30))

        def reconstruct(self, x):
            z = self.encode(x)
            return self.decode(z), z

        def train_step(self, x):
            z = self.encode(x)
            x_hat = self.decode(z)
            loss = np.mean((x_hat - x) ** 2)
            grad_xhat = 2 * (x_hat - x) / self.pixel_dim
            pre_dec = np.clip(self.W_dec @ z, -30, 30)
            grad_z_dec = (1 - np.tanh(pre_dec) ** 2) * grad_xhat
            grad_Wdec = np.clip(np.outer(grad_z_dec, z), -1, 1)
            pre_enc = np.clip(self.W_enc @ x, -30, 30)
            grad_z_enc = self.W_dec.T @ grad_z_dec
            grad_z_pre = (1 - np.tanh(pre_enc) ** 2) * grad_z_enc
            grad_Wenc = np.clip(np.outer(grad_z_pre, x), -1, 1)
            self.W_dec -= self.lr * grad_Wdec
            self.W_enc -= self.lr * grad_Wenc
            return loss

    ae = LinearAutoencoder(PIXEL_DIM, LATENT_DIM)
    dataset = [np.sin(np.linspace(0, 2 * np.pi, PIXEL_DIM) * (i * 0.5 + 1)) for i in range(100)]
    for _ in range(200):
        for x in dataset:
            ae.train_step(x)
    return ae


@pytest.fixture
def denoiser():
    np.random.seed(13)

    class TextConditionedDenoiser:
        def __init__(self, latent_dim, text_dim):
            self.latent_dim = latent_dim
            self.text_dim = text_dim
            self.W_noise = np.random.randn(latent_dim, latent_dim) * 0.1
            self.W_text = np.random.randn(latent_dim, text_dim) * 0.1

        def predict_noise(self, z_t, text_vec, t_step, T):
            t_enc = np.sin(np.pi * t_step / T * np.linspace(0, 1, self.latent_dim))
            text_condition = self.W_text @ text_vec
            h = np.tanh(self.W_noise @ z_t + t_enc)
            return h + 0.1 * text_condition

    return TextConditionedDenoiser(LATENT_DIM, TEXT_DIM)


# ────────────────────────────────────────────────────────────────────────────
# TestAutoencoder
# ────────────────────────────────────────────────────────────────────────────

class TestAutoencoder:
    def test_encode_output_shape(self, autoencoder):
        x = np.random.randn(PIXEL_DIM)
        z = autoencoder.encode(x)
        assert z.shape == (LATENT_DIM,)

    def test_decode_output_shape(self, autoencoder):
        z = np.random.randn(LATENT_DIM)
        x_hat = autoencoder.decode(z)
        assert x_hat.shape == (PIXEL_DIM,)

    def test_compression_ratio(self):
        ratio = PIXEL_DIM / LATENT_DIM
        assert ratio == 8.0, f"期望压缩比8x，实际{ratio}"

    def test_encode_bounded(self, autoencoder):
        """tanh 输出必须在 [-1, 1]"""
        x = np.random.randn(PIXEL_DIM) * 10
        z = autoencoder.encode(x)
        assert np.all(np.abs(z) <= 1.0 + 1e-9)

    def test_decode_bounded(self, autoencoder):
        """tanh 输出必须在 [-1, 1]"""
        z = np.random.randn(LATENT_DIM) * 10
        x_hat = autoencoder.decode(z)
        assert np.all(np.abs(x_hat) <= 1.0 + 1e-9)

    def test_reconstruction_quality(self, autoencoder):
        """训练后重建误差应在合理范围内"""
        dataset = [np.sin(np.linspace(0, 2 * np.pi, PIXEL_DIM) * (i * 0.5 + 1)) for i in range(20)]
        mses = [np.mean((autoencoder.decode(autoencoder.encode(x)) - x) ** 2) for x in dataset]
        assert np.mean(mses) < 0.5, f"重建MSE过大: {np.mean(mses):.4f}"

    def test_training_loss_decreases(self):
        """训练损失应该下降"""
        np.random.seed(42)

        class SimpleAE:
            def __init__(self):
                self.W_enc = np.random.randn(4, 16) * 0.001
                self.W_dec = np.random.randn(16, 4) * 0.001
                self.lr = 0.005

            def encode(self, x): return np.tanh(np.clip(self.W_enc @ x, -30, 30))
            def decode(self, z): return np.tanh(np.clip(self.W_dec @ z, -30, 30))
            def train_step(self, x):
                z = self.encode(x)
                x_hat = self.decode(z)
                loss = np.mean((x_hat - x) ** 2)
                g = 2 * (x_hat - x) / 16
                pre_dec = np.clip(self.W_dec @ z, -30, 30)
                gz = (1 - np.tanh(pre_dec) ** 2) * g
                self.W_dec -= self.lr * np.clip(np.outer(gz, z), -1, 1)
                pre_enc = np.clip(self.W_enc @ x, -30, 30)
                ge = self.W_dec.T @ gz
                gp = (1 - np.tanh(pre_enc) ** 2) * ge
                self.W_enc -= self.lr * np.clip(np.outer(gp, x), -1, 1)
                return loss

        ae = SimpleAE()
        data = [np.sin(np.linspace(0, np.pi, 16) * (i + 1)) for i in range(30)]
        early_losses = [ae.train_step(x) for x in data]
        for _ in range(200):
            for x in data:
                ae.train_step(x)
        late_losses = [np.mean((ae.decode(ae.encode(x)) - x) ** 2) for x in data]
        assert np.mean(late_losses) < np.mean(early_losses), "训练后损失应该更低"


# ────────────────────────────────────────────────────────────────────────────
# TestTextConditioning
# ────────────────────────────────────────────────────────────────────────────

class TestTextConditioning:
    def test_predict_noise_shape(self, denoiser):
        z = np.random.randn(LATENT_DIM)
        text = np.random.randn(TEXT_DIM)
        eps = denoiser.predict_noise(z, text, T // 2, T)
        assert eps.shape == (LATENT_DIM,)

    def test_different_texts_produce_different_predictions(self, denoiser):
        z = np.random.randn(LATENT_DIM)
        text_a = np.array([1.0] + [0.0] * (TEXT_DIM - 1))
        text_b = np.array([0.0, 1.0] + [0.0] * (TEXT_DIM - 2))
        pred_a = denoiser.predict_noise(z, text_a, T // 2, T)
        pred_b = denoiser.predict_noise(z, text_b, T // 2, T)
        assert np.linalg.norm(pred_a - pred_b) > 1e-6, "不同文字应产生不同预测"

    def test_same_input_same_output(self, denoiser):
        """确定性：同样输入必须产生同样输出"""
        z = np.random.randn(LATENT_DIM)
        text = np.random.randn(TEXT_DIM)
        pred1 = denoiser.predict_noise(z, text, 25, T)
        pred2 = denoiser.predict_noise(z, text, 25, T)
        np.testing.assert_array_equal(pred1, pred2)

    def test_zero_text_unconditioned(self, denoiser):
        """零向量文字描述和非零文字描述应产生不同预测"""
        z = np.random.randn(LATENT_DIM)
        text_zero = np.zeros(TEXT_DIM)
        text_nonzero = np.ones(TEXT_DIM)
        pred_zero = denoiser.predict_noise(z, text_zero, T // 2, T)
        pred_nonzero = denoiser.predict_noise(z, text_nonzero, T // 2, T)
        assert np.linalg.norm(pred_zero - pred_nonzero) > 0


# ────────────────────────────────────────────────────────────────────────────
# TestNoiseSchedule
# ────────────────────────────────────────────────────────────────────────────

class TestNoiseSchedule:
    def test_alpha_bar_monotone_decreasing(self, alpha_bar):
        assert np.all(np.diff(alpha_bar) < 0), "alpha_bar 必须单调递减"

    def test_alpha_bar_starts_near_one(self, alpha_bar):
        assert alpha_bar[0] > 0.99

    def test_alpha_bar_ends_low(self, alpha_bar):
        """t=T 时信号占比显著低于 t=0（T=50 时约降至 60%，用 80% 阈值）"""
        assert alpha_bar[-1] < alpha_bar[0] * 0.8

    def test_forward_diffusion_increases_noise(self, alpha_bar):
        """加噪后方差更大（或范数更接近纯高斯）"""
        np.random.seed(5)
        z0 = np.random.randn(LATENT_DIM)
        eps = np.random.randn(LATENT_DIM)
        ab = alpha_bar[T - 1]
        z_T = np.sqrt(ab) * z0 + np.sqrt(1 - ab) * eps
        # 加噪后对原信号的投影应小于原信号范数
        dot = np.dot(z_T, z0 / (np.linalg.norm(z0) + 1e-8))
        assert dot < np.linalg.norm(z0), "加噪应降低原信号占比"


# ────────────────────────────────────────────────────────────────────────────
# TestLatentDDIM
# ────────────────────────────────────────────────────────────────────────────

class TestLatentDDIM:

    def _ddim_step(self, z_t, t, t_prev, alpha_bar, denoiser, text_vec):
        ab_t = alpha_bar[t]
        ab_prev = alpha_bar[t_prev] if t_prev >= 0 else 1.0
        eps_pred = denoiser.predict_noise(z_t, text_vec, t, T)
        z0_pred = (z_t - np.sqrt(1 - ab_t) * eps_pred) / np.sqrt(ab_t)
        z0_pred = np.clip(z0_pred, -3, 3)
        return np.sqrt(ab_prev) * z0_pred + np.sqrt(1 - ab_prev) * eps_pred

    def _sample(self, denoiser, text_vec, alpha_bar, n_steps=10):
        z = np.random.randn(LATENT_DIM)
        timesteps = np.linspace(T - 1, 0, n_steps + 1, dtype=int)
        for i in range(len(timesteps) - 1):
            z = self._ddim_step(z, timesteps[i], timesteps[i + 1], alpha_bar, denoiser, text_vec)
        return z

    def test_ddim_step_shape(self, denoiser, alpha_bar):
        z_t = np.random.randn(LATENT_DIM)
        text = np.zeros(TEXT_DIM)
        z_prev = self._ddim_step(z_t, T - 1, T // 2, alpha_bar, denoiser, text)
        assert z_prev.shape == (LATENT_DIM,)

    def test_ddim_deterministic(self, denoiser, alpha_bar):
        """相同种子应产生完全相同的采样结果"""
        text = np.array([1.0] + [0.0] * (TEXT_DIM - 1))
        np.random.seed(77)
        z1 = self._sample(denoiser, text, alpha_bar, n_steps=10)
        np.random.seed(77)
        z2 = self._sample(denoiser, text, alpha_bar, n_steps=10)
        np.testing.assert_array_almost_equal(z1, z2)

    def test_different_texts_different_samples(self, denoiser, alpha_bar):
        """不同文字描述应生成不同的隐向量"""
        text_a = np.array([1.0] + [0.0] * (TEXT_DIM - 1))
        text_b = np.array([0.0, 1.0] + [0.0] * (TEXT_DIM - 2))
        np.random.seed(42)
        z_a = self._sample(denoiser, text_a, alpha_bar)
        np.random.seed(42)
        z_b = self._sample(denoiser, text_b, alpha_bar)
        diff = np.linalg.norm(z_a - z_b)
        assert diff > 1e-6, f"不同文字条件应生成不同隐向量，diff={diff:.6f}"

    def test_sample_output_finite(self, denoiser, alpha_bar):
        """采样结果应为有限值（无 NaN/Inf）"""
        text = np.zeros(TEXT_DIM)
        np.random.seed(1)
        z = self._sample(denoiser, text, alpha_bar)
        assert np.all(np.isfinite(z)), "采样结果包含 NaN 或 Inf"

    def test_sample_in_reasonable_range(self, denoiser, alpha_bar):
        """采样隐向量范数应在合理范围内"""
        text = np.random.randn(TEXT_DIM)
        np.random.seed(2)
        z = self._sample(denoiser, text, alpha_bar)
        norm = np.linalg.norm(z)
        assert norm < 50, f"采样隐向量范数过大: {norm:.2f}"
