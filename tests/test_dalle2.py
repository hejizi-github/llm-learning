"""
tests/test_dalle2.py — Tests for Node 20: DALL-E 2 (2022)
Covers: L2 normalize, cosine similarity, Prior simulation, decoder simulation, slerp interpolation, document structure.
"""
import os
import numpy as np
import pytest

# ── Helpers (duplicated from notebook for test isolation) ────────────────────

def l2_normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-8)


def cosine_similarity(a, b):
    a = l2_normalize(np.array(a, dtype=float))
    b = l2_normalize(np.array(b, dtype=float))
    return float(np.dot(a, b))


def slerp(v0, v1, t):
    v0 = l2_normalize(v0)
    v1 = l2_normalize(v1)
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    omega = np.arccos(dot)
    if abs(omega) < 1e-6:
        return l2_normalize((1 - t) * v0 + t * v1)
    sin_omega = np.sin(omega)
    return (np.sin((1 - t) * omega) / sin_omega) * v0 + (np.sin(t * omega) / sin_omega) * v1


def simulate_prior(text_embedding, noise_scale=0.1, n_diffusion_steps=20, seed=7):
    np.random.seed(seed)
    x = np.random.randn(len(text_embedding))
    x = l2_normalize(x)
    history = [x.copy()]
    for step in range(n_diffusion_steps):
        alpha = (step + 1) / n_diffusion_steps
        noise = np.random.randn(len(text_embedding)) * noise_scale * (1 - alpha)
        x = l2_normalize((1 - alpha * 0.3) * x + alpha * 0.3 * text_embedding + noise)
        history.append(x.copy())
    return x, history


def simulate_decoder(image_embedding, target_size=8, n_steps=20, guidance_scale=3.0, seed=3):
    np.random.seed(seed)
    n_pixels = target_size * target_size
    repeats = (n_pixels // len(image_embedding)) + 1
    target_pattern = np.tile(image_embedding, repeats)[:n_pixels].reshape(target_size, target_size)
    target_pattern = (target_pattern - target_pattern.min()) / (target_pattern.max() - target_pattern.min() + 1e-8)
    x = np.random.randn(target_size, target_size)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    history = [x.copy()]
    losses = []
    for step in range(n_steps):
        t = 1.0 - (step + 1) / n_steps
        noise = np.random.randn(target_size, target_size) * t * 0.3
        guidance = guidance_scale * (target_pattern - x)
        x = np.clip(x + 0.05 * guidance + noise, 0, 1)
        losses.append(float(np.mean((x - target_pattern) ** 2)))
        history.append(x.copy())
    return x, target_pattern, history, losses


# ── TestL2Normalize ──────────────────────────────────────────────────────────

class TestL2Normalize:
    def test_unit_length(self):
        v = np.array([3.0, 4.0])
        norm = np.linalg.norm(l2_normalize(v))
        assert abs(norm - 1.0) < 1e-6

    def test_already_normalized(self):
        v = np.array([1.0, 0.0, 0.0])
        result = l2_normalize(v)
        np.testing.assert_allclose(result, v, atol=1e-6)

    def test_batch_normalize(self):
        vs = np.array([[3.0, 4.0], [5.0, 12.0]])
        norms = np.linalg.norm(l2_normalize(vs), axis=-1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)

    def test_zero_vector_safe(self):
        v = np.zeros(4)
        result = l2_normalize(v)
        assert not np.any(np.isnan(result))


# ── TestCosineSimilarity ─────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_opposite_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert abs(cosine_similarity(v, -v) - (-1.0)) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_range(self):
        a = np.random.randn(16)
        b = np.random.randn(16)
        sim = cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0

    def test_same_direction_high_sim(self):
        base = np.array([1.0, 0.5, -0.3, 0.8])
        a = base + np.random.randn(4) * 0.05
        b = base + np.random.randn(4) * 0.05
        assert cosine_similarity(a, b) > 0.9


# ── TestSemanticSpace ────────────────────────────────────────────────────────

class TestSemanticSpace:
    def setup_method(self):
        np.random.seed(0)
        D = 8
        cat_base = l2_normalize(np.array([1.0, 0.8, 0.2, -0.1, 0.3, 0.5, -0.2, 0.1]))
        dog_base = l2_normalize(np.array([0.9, 0.7, 0.3, 0.1, 0.4, 0.4, -0.1, 0.2]))
        plane_base = l2_normalize(np.array([-0.5, 0.2, 0.9, 0.7, -0.3, 0.1, 0.8, -0.4]))
        self.cat_img = l2_normalize(cat_base + np.random.randn(D) * 0.15)
        self.cat_txt = l2_normalize(cat_base + np.random.randn(D) * 0.15)
        self.dog_img = l2_normalize(dog_base + np.random.randn(D) * 0.15)
        self.dog_txt = l2_normalize(dog_base + np.random.randn(D) * 0.15)
        self.plane_txt = l2_normalize(plane_base + np.random.randn(D) * 0.15)

    def test_same_concept_img_txt_close(self):
        sim = cosine_similarity(self.cat_img, self.cat_txt)
        assert sim > 0.7, f"同一概念图文相似度应高，got {sim:.4f}"

    def test_cross_concept_img_txt_farther(self):
        sim_same = cosine_similarity(self.cat_img, self.cat_txt)
        sim_cross = cosine_similarity(self.cat_img, self.plane_txt)
        assert sim_same > sim_cross, "同类相似度应 > 跨类相似度"

    def test_animal_similarity_higher_than_plane(self):
        sim_animal = cosine_similarity(self.cat_txt, self.dog_txt)
        sim_plane = cosine_similarity(self.cat_txt, self.plane_txt)
        assert sim_animal > sim_plane


# ── TestPrior ────────────────────────────────────────────────────────────────

class TestPrior:
    def setup_method(self):
        np.random.seed(0)
        D = 8
        cat_base = l2_normalize(np.array([1.0, 0.8, 0.2, -0.1, 0.3, 0.5, -0.2, 0.1]))
        plane_base = l2_normalize(np.array([-0.5, 0.2, 0.9, 0.7, -0.3, 0.1, 0.8, -0.4]))
        self.cat_txt = l2_normalize(cat_base + np.random.randn(D) * 0.15)
        self.plane_txt = l2_normalize(plane_base + np.random.randn(D) * 0.15)

    def test_prior_converges_toward_target(self):
        pred, history = simulate_prior(self.cat_txt, seed=7)
        sim_initial = cosine_similarity(history[0], self.cat_txt)
        sim_final = cosine_similarity(history[-1], self.cat_txt)
        assert sim_final > sim_initial, "Prior 应使嵌入逐渐接近目标文字嵌入"

    def test_prior_semantic_alignment(self):
        pred, _ = simulate_prior(self.cat_txt, seed=7)
        sim_cat = cosine_similarity(pred, self.cat_txt)
        sim_plane = cosine_similarity(pred, self.plane_txt)
        assert sim_cat > sim_plane, "Prior 生成的嵌入应与输入概念更近"

    def test_prior_output_normalized(self):
        pred, _ = simulate_prior(self.cat_txt, seed=7)
        norm = np.linalg.norm(pred)
        assert abs(norm - 1.0) < 1e-4, f"Prior 输出应为单位向量，got norm={norm:.4f}"

    def test_prior_converges_overall(self):
        pred, history = simulate_prior(self.cat_txt, n_diffusion_steps=10, seed=7)
        sims = [cosine_similarity(h, self.cat_txt) for h in history]
        # 前半段均值 vs 后半段均值：整体趋势上升即可
        mid = len(sims) // 2
        assert np.mean(sims[mid:]) > np.mean(sims[:mid]), "Prior 过程后半段平均相似度应高于前半段"


# ── TestDecoder ──────────────────────────────────────────────────────────────

class TestDecoder:
    def setup_method(self):
        np.random.seed(0)
        D = 8
        cat_base = l2_normalize(np.array([1.0, 0.8, 0.2, -0.1, 0.3, 0.5, -0.2, 0.1]))
        self.cat_emb = l2_normalize(cat_base + np.random.randn(D) * 0.15)

    def test_decoder_reduces_mse(self):
        result, target, history, losses = simulate_decoder(self.cat_emb, target_size=8, n_steps=20, seed=3)
        initial_mse = np.mean((history[0] - target) ** 2)
        final_mse = np.mean((result - target) ** 2)
        assert final_mse < initial_mse, "解码过程应使 MSE 下降"

    def test_decoder_output_clipped(self):
        result, _, _, _ = simulate_decoder(self.cat_emb, target_size=8, n_steps=20, seed=3)
        assert result.min() >= 0.0 and result.max() <= 1.0, "输出应在 [0, 1] 范围内"

    def test_decoder_output_shape(self):
        result, _, _, _ = simulate_decoder(self.cat_emb, target_size=8, n_steps=20, seed=3)
        assert result.shape == (8, 8)

    def test_decoder_loss_trend(self):
        _, _, _, losses = simulate_decoder(self.cat_emb, target_size=8, n_steps=20, seed=3)
        # Last quarter of losses should be smaller than first quarter (overall trend)
        q1 = np.mean(losses[:5])
        q4 = np.mean(losses[-5:])
        assert q4 < q1, "后期损失应比初期更小"


# ── TestSlerp ────────────────────────────────────────────────────────────────

class TestSlerp:
    def test_t0_returns_v0(self):
        v0 = l2_normalize(np.array([1.0, 0.0, 0.0]))
        v1 = l2_normalize(np.array([0.0, 1.0, 0.0]))
        result = slerp(v0, v1, 0.0)
        np.testing.assert_allclose(result, v0, atol=1e-5)

    def test_t1_returns_v1(self):
        v0 = l2_normalize(np.array([1.0, 0.0, 0.0]))
        v1 = l2_normalize(np.array([0.0, 1.0, 0.0]))
        result = slerp(v0, v1, 1.0)
        np.testing.assert_allclose(result, v1, atol=1e-5)

    def test_midpoint_equidistant(self):
        v0 = l2_normalize(np.array([1.0, 0.0, 0.0]))
        v1 = l2_normalize(np.array([0.0, 1.0, 0.0]))
        mid = slerp(v0, v1, 0.5)
        sim0 = cosine_similarity(mid, v0)
        sim1 = cosine_similarity(mid, v1)
        assert abs(sim0 - sim1) < 0.01, "球面插值中点应与两端等距"

    def test_output_normalized(self):
        v0 = l2_normalize(np.array([1.0, 0.5, 0.2]))
        v1 = l2_normalize(np.array([0.3, 0.8, -0.5]))
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = slerp(v0, v1, t)
            assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_slerp_similarity_endpoints(self):
        np.random.seed(0)
        D = 8
        cat_base = l2_normalize(np.array([1.0, 0.8, 0.2, -0.1, 0.3, 0.5, -0.2, 0.1]))
        dog_base = l2_normalize(np.array([0.9, 0.7, 0.3, 0.1, 0.4, 0.4, -0.1, 0.2]))
        cat_txt = l2_normalize(cat_base + np.random.randn(D) * 0.15)
        dog_txt = l2_normalize(dog_base + np.random.randn(D) * 0.15)
        t_values = np.linspace(0, 1, 10)
        interp = [slerp(cat_txt, dog_txt, t) for t in t_values]
        sims_cat = [cosine_similarity(v, cat_txt) for v in interp]
        sims_dog = [cosine_similarity(v, dog_txt) for v in interp]
        # 端点行为：t=0 与猫最近，t=1 与狗最近
        assert sims_cat[0] > sims_cat[-1], "t=0 时应与猫更近"
        assert sims_dog[-1] > sims_dog[0], "t=1 时应与狗更近"
        # 前半段更接近猫，后半段更接近狗
        mid = len(t_values) // 2
        assert np.mean(sims_cat[:mid]) > np.mean(sims_cat[mid:]), "前半段应整体更接近猫"
        assert np.mean(sims_dog[mid:]) > np.mean(sims_dog[:mid]), "后半段应整体更接近狗"

    def test_slerp_similarity_strictly_monotone(self):
        # SLERP cosine similarity is mathematically guaranteed monotone:
        # slerp(v0,v1,t) similarity to v0 = cos(t*omega), strictly decreasing.
        # With deterministic inputs (no noise), this must hold exactly.
        v0 = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        v1 = l2_normalize(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        t_values = np.linspace(0, 1, 11)
        interp = [slerp(v0, v1, t) for t in t_values]
        sims_v0 = [cosine_similarity(v, v0) for v in interp]
        sims_v1 = [cosine_similarity(v, v1) for v in interp]
        # Strictly decreasing similarity to v0
        for i in range(len(sims_v0) - 1):
            assert sims_v0[i] >= sims_v0[i + 1] - 1e-9, (
                f"SLERP similarity to v0 should be monotone: step {i}→{i+1}: "
                f"{sims_v0[i]:.6f} → {sims_v0[i+1]:.6f}"
            )
        # Strictly increasing similarity to v1
        for i in range(len(sims_v1) - 1):
            assert sims_v1[i] <= sims_v1[i + 1] + 1e-9, (
                f"SLERP similarity to v1 should be monotone: step {i}→{i+1}: "
                f"{sims_v1[i]:.6f} → {sims_v1[i+1]:.6f}"
            )


# ── TestDocumentStructure ────────────────────────────────────────────────────

class TestDocumentStructure:
    DOC_PATH = os.path.join(os.path.dirname(__file__), "../docs/20-dalle2-2022.md")

    def _read_doc(self):
        with open(self.DOC_PATH, encoding="utf-8") as f:
            return f.read()

    def test_doc_exists(self):
        assert os.path.exists(self.DOC_PATH), "docs/20-dalle2-2022.md 应存在"

    def test_has_arxiv_citation(self):
        doc = self._read_doc()
        assert "2204.06125" in doc, "文档应包含 DALL-E 2 的 arXiv ID"

    def test_has_dalle1_reference(self):
        doc = self._read_doc()
        assert "DALL-E 1" in doc or "DALL-E1" in doc or "ramesh2021" in doc.lower() or "2102.12092" in doc

    def test_has_prior_section(self):
        doc = self._read_doc()
        assert "Prior" in doc or "prior" in doc, "文档应包含 Prior 的介绍"

    def test_historical_ordering_note(self):
        doc = self._read_doc()
        assert "2022" in doc and ("2021" in doc), "文档应包含历史时序（2021 CLIP, 2022 DALL-E 2）"

    def test_has_clip_connection(self):
        doc = self._read_doc()
        assert "CLIP" in doc, "文档应包含 CLIP 的连接说明"
