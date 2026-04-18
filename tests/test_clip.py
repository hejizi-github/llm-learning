"""Tests for CLIP node 19: contrastive learning core mechanisms."""

import numpy as np
import pytest


# ── helpers (mirrors notebook implementations) ──────────────────────────

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def l2_normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-8)


def similarity_matrix(image_vecs, text_vecs):
    img = l2_normalize(np.array(image_vecs, dtype=float))
    txt = l2_normalize(np.array(text_vecs, dtype=float))
    return img @ txt.T


def infonce_loss(image_vecs, text_vecs, temperature=0.07):
    sim_mat = similarity_matrix(image_vecs, text_vecs) / temperature
    N = len(image_vecs)
    # log-sum-exp trick for numerical stability
    max_r = sim_mat.max(axis=1, keepdims=True)
    lse_r = np.log(np.sum(np.exp(sim_mat - max_r), axis=1, keepdims=True)) + max_r
    log_softmax_rows = sim_mat - lse_r
    loss_i2t = -log_softmax_rows[np.arange(N), np.arange(N)].mean()

    max_c = sim_mat.T.max(axis=1, keepdims=True)
    lse_c = np.log(np.sum(np.exp(sim_mat.T - max_c), axis=1, keepdims=True)) + max_c
    log_softmax_cols = sim_mat.T - lse_c
    loss_t2i = -log_softmax_cols[np.arange(N), np.arange(N)].mean()

    return (loss_i2t + loss_t2i) / 2.0


# ── cosine similarity ──────────────────────────────────────────────────

class TestCosineSimilarity:

    def test_same_vector_is_one(self):
        v = np.array([3.0, 4.0, 0.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_opposite_vectors_is_minus_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(v, -v) - (-1.0)) < 1e-6

    def test_orthogonal_is_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_scale_invariant(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])   # 2×a
        assert abs(cosine_similarity(a, b) - 1.0) < 1e-6

    def test_range_within_minus_one_to_one(self):
        rng = np.random.RandomState(0)
        for _ in range(20):
            a = rng.randn(16)
            b = rng.randn(16)
            sim = cosine_similarity(a, b)
            assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity(np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0])) == 0.0


# ── l2_normalize ──────────────────────────────────────────────────────

class TestL2Normalize:

    def test_unit_norm(self):
        vecs = np.random.RandomState(1).randn(5, 8)
        normalized = l2_normalize(vecs)
        norms = np.linalg.norm(normalized, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_direction_preserved(self):
        v = np.array([[3.0, 4.0]])
        n = l2_normalize(v)
        assert abs(cosine_similarity(v[0], n[0]) - 1.0) < 1e-6

    def test_2d_input_shape_preserved(self):
        v = np.random.randn(4, 16)
        assert l2_normalize(v).shape == (4, 16)


# ── similarity_matrix ─────────────────────────────────────────────────

class TestSimilarityMatrix:

    def test_shape_N_by_N(self):
        img = np.random.randn(4, 8)
        txt = np.random.randn(4, 8)
        assert similarity_matrix(img, txt).shape == (4, 4)

    def test_diagonal_high_when_vectors_aligned(self):
        """正确配对的图文向量方向相近时，对角线应是该行最大值。"""
        rng = np.random.RandomState(7)
        base = rng.randn(3, 8)
        img = l2_normalize(base + 0.01 * rng.randn(3, 8))
        txt = l2_normalize(base + 0.01 * rng.randn(3, 8))
        mat = similarity_matrix(img, txt)
        for i in range(3):
            assert mat[i, i] == mat[i].max(), f"row {i}: diagonal not maximum"

    def test_all_values_in_minus_one_to_one(self):
        img = np.random.randn(5, 6)
        txt = np.random.randn(5, 6)
        mat = similarity_matrix(img, txt)
        assert mat.min() >= -1.0 - 1e-6
        assert mat.max() <= 1.0 + 1e-6

    def test_identity_input_is_identity_matrix(self):
        eye = np.eye(4, 4)
        mat = similarity_matrix(eye, eye)
        np.testing.assert_allclose(mat, np.eye(4), atol=1e-6)


# ── infonce_loss ──────────────────────────────────────────────────────

class TestInfoNCELoss:

    def test_perfect_match_near_zero_loss(self):
        """完全匹配时损失接近 0。"""
        eye = np.eye(4, 4)
        loss = infonce_loss(eye, eye, temperature=0.07)
        assert loss < 0.01, f"完全匹配损失应 < 0.01，得到 {loss:.4f}"

    def test_random_loss_near_log_N(self):
        """随机初始化时，损失应接近 log(N)（均匀分布的理论值）。"""
        rng = np.random.RandomState(42)
        N = 6
        img = rng.randn(N, 32)
        txt = rng.randn(N, 32)
        loss = infonce_loss(img, txt, temperature=1.0)
        expected = np.log(N)
        assert abs(loss - expected) < 0.5, f"随机损失 {loss:.3f}，期望约 log({N})={expected:.3f}"

    def test_loss_decreases_when_positive_pair_improves(self):
        """正样本对相似度提高后，损失应下降。"""
        rng = np.random.RandomState(99)
        img = rng.randn(3, 8)
        txt = rng.randn(3, 8)
        loss_before = infonce_loss(img, txt)
        # 让图文对更像：把文字向量往图像方向拉近
        txt_better = txt + 2.0 * img
        loss_after = infonce_loss(img, txt_better)
        assert loss_after < loss_before, f"损失应下降：前={loss_before:.4f}，后={loss_after:.4f}"

    def test_loss_nonnegative(self):
        img = np.random.randn(4, 8)
        txt = np.random.randn(4, 8)
        assert infonce_loss(img, txt) >= 0.0

    def test_temperature_scaling(self):
        """低温度让分布更尖锐：相同数据下，低温度损失 < 高温度损失。"""
        eye = np.eye(4, 4)
        loss_low_t = infonce_loss(eye, eye, temperature=0.01)
        loss_high_t = infonce_loss(eye, eye, temperature=1.0)
        # 低温度使 logit 差异放大，分布更尖锐，所以损失更低
        assert loss_low_t < loss_high_t, (
            f"低温损失({loss_low_t:.4f}) 应 < 高温损失({loss_high_t:.4f})")


# ── zero-shot prediction ──────────────────────────────────────────────

class TestZeroShotPrediction:

    def _make_class_embeddings(self, n_classes=3, dim=16, seed=0):
        """生成模拟的类别原型向量。"""
        rng = np.random.RandomState(seed)
        # 每个类别一个"方向"
        directions = rng.randn(n_classes, dim)
        directions = l2_normalize(directions)
        return directions

    def zero_shot_predict(self, image_emb, text_embs):
        sims = np.array([cosine_similarity(image_emb, t) for t in text_embs])
        return int(np.argmax(sims))

    def test_predicts_correct_class(self):
        """与正确类别嵌入方向相近的图片，应被预测为该类别。"""
        dirs = self._make_class_embeddings(n_classes=3)
        rng = np.random.RandomState(5)
        text_embs = l2_normalize(dirs + 0.05 * rng.randn(3, 16))

        # 生成 3 张图，各属于一个类别
        for true_class in range(3):
            img_emb = l2_normalize((dirs[true_class] + 0.05 * rng.randn(16)).reshape(1, -1))[0]
            pred = self.zero_shot_predict(img_emb, text_embs)
            assert pred == true_class, f"true={true_class}, pred={pred}"

    def test_top_sim_is_correct_pair(self):
        """正确的图文对，余弦相似度应是该图与所有文字中最高的。"""
        dirs = self._make_class_embeddings(n_classes=4)
        rng = np.random.RandomState(9)
        text_embs = l2_normalize(dirs + 0.02 * rng.randn(4, 16))
        for i in range(4):
            img_emb = l2_normalize((dirs[i] + 0.02 * rng.randn(16)).reshape(1, -1))[0]
            sims = [cosine_similarity(img_emb, t) for t in text_embs]
            assert np.argmax(sims) == i, f"类别{i}的最高相似度不是正确配对"

    def test_accuracy_above_80_percent(self):
        """9 张图（各 3 个类别各 3 张），Zero-shot 准确率应 >= 80%。"""
        dirs = self._make_class_embeddings(n_classes=3, dim=16, seed=99)
        rng = np.random.RandomState(11)
        text_embs = l2_normalize(dirs + 0.05 * rng.randn(3, 16))

        correct = 0
        total = 9
        for true_class in range(3):
            for _ in range(3):
                img_emb = l2_normalize((dirs[true_class] + 0.05 * rng.randn(16)).reshape(1, -1))[0]
                pred = self.zero_shot_predict(img_emb, text_embs)
                if pred == true_class:
                    correct += 1

        accuracy = correct / total
        assert accuracy >= 0.8, f"Zero-shot 准确率 {accuracy:.0%} < 80%"


# ── document linkage check ────────────────────────────────────────────

class TestDocumentStructure:

    def test_doc_file_exists(self):
        import pathlib
        doc = pathlib.Path("docs/19-clip-2021.md")
        assert doc.exists(), "文档文件 docs/19-clip-2021.md 不存在"

    def test_doc_has_arxiv_ref(self):
        import pathlib
        content = pathlib.Path("docs/19-clip-2021.md").read_text(encoding="utf-8")
        assert "2103.00020" in content, "文档应包含 arXiv ID 2103.00020"

    def test_doc_has_citation_key(self):
        import pathlib
        content = pathlib.Path("docs/19-clip-2021.md").read_text(encoding="utf-8")
        assert "radford2021clip" in content, "文档应包含引用键 radford2021clip"

    def test_bib_entry_exists(self):
        import pathlib
        bib = pathlib.Path("refs/references.bib").read_text(encoding="utf-8")
        assert "radford2021clip" in bib, "references.bib 应包含 CLIP 引用条目"

    def test_notebook_file_exists(self):
        import pathlib
        nb = pathlib.Path("notebooks/19-clip-2021.ipynb")
        assert nb.exists(), "notebook 文件不存在"
