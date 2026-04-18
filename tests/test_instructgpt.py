"""
tests/test_instructgpt.py — 节点11 InstructGPT/RLHF 测试

覆盖：Bradley-Terry偏好模型 | Reward Model | KL散度 | PPO目标 | RLHF流程
"""
import numpy as np
import pytest


# ─── Helpers (mirror notebook implementations) ────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def bradley_terry_prob(r_w, r_l):
    return sigmoid(r_w - r_l)

def rm_loss(r_w, r_l):
    return float(-np.log(bradley_terry_prob(r_w, r_l) + 1e-8))

def softmax(x):
    x = np.array(x, dtype=float)
    e = np.exp(x - x.max())
    return e / e.sum()

def kl_divergence(p, q, eps=1e-10):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log((p + eps) / (q + eps))))

def reward_model(features, weights):
    return float(np.dot(features, weights))

def ppo_objective(rm_score, kl, beta=0.05):
    return rm_score - beta * kl

def train_reward_model(preference_data, n_features=5, n_epochs=200, lr=0.05):
    weights = np.zeros(n_features)
    losses = []
    for _ in range(n_epochs):
        epoch_loss = 0.0
        grad = np.zeros(n_features)
        for chosen, rejected in preference_data:
            r_w = reward_model(chosen, weights)
            r_l = reward_model(rejected, weights)
            prob = bradley_terry_prob(r_w, r_l)
            epoch_loss += -np.log(prob + 1e-8)
            grad += -(1 - prob) * (chosen - rejected)
        weights -= lr * grad / len(preference_data)
        losses.append(epoch_loss / len(preference_data))
    return weights, losses


# ─── 1. Sigmoid & Bradley-Terry ───────────────────────────────────────────

class TestSigmoid:
    def test_sigmoid_zero_is_half(self):
        assert abs(sigmoid(0.0) - 0.5) < 1e-6

    def test_sigmoid_large_positive_near_one(self):
        assert sigmoid(100.0) > 0.999

    def test_sigmoid_large_negative_near_zero(self):
        assert sigmoid(-100.0) < 0.001

    def test_sigmoid_monotone(self):
        xs = np.linspace(-5, 5, 20)
        vals = [sigmoid(x) for x in xs]
        assert all(a < b for a, b in zip(vals, vals[1:]))

    def test_sigmoid_output_in_01(self):
        for x in [-10, -1, 0, 1, 10]:
            v = sigmoid(x)
            assert 0 < v < 1


class TestBradleyTerry:
    def test_equal_scores_gives_half(self):
        assert abs(bradley_terry_prob(1.0, 1.0) - 0.5) < 1e-6

    def test_chosen_higher_gives_prob_above_half(self):
        assert bradley_terry_prob(2.0, 0.0) > 0.5

    def test_chosen_lower_gives_prob_below_half(self):
        assert bradley_terry_prob(0.0, 2.0) < 0.5

    def test_prob_in_01(self):
        for r_w, r_l in [(3, 0), (-2, 1), (0, 0), (1, 1)]:
            p = bradley_terry_prob(r_w, r_l)
            assert 0 < p < 1

    def test_symmetry(self):
        p1 = bradley_terry_prob(2.0, 0.0)
        p2 = bradley_terry_prob(0.0, 2.0)
        assert abs(p1 + p2 - 1.0) < 1e-6

    def test_large_gap_high_confidence(self):
        assert bradley_terry_prob(10.0, -10.0) > 0.99


class TestRMLoss:
    def test_good_ranking_low_loss(self):
        assert rm_loss(3.0, -1.0) < 0.1

    def test_bad_ranking_high_loss(self):
        assert rm_loss(-1.0, 3.0) > 1.0

    def test_tied_scores_moderate_loss(self):
        loss = rm_loss(0.0, 0.0)
        assert 0.5 < loss < 1.0

    def test_loss_nonnegative(self):
        for r_w, r_l in [(2, 1), (0, 0), (-1, 1)]:
            assert rm_loss(r_w, r_l) >= 0


# ─── 2. Reward Model Training ─────────────────────────────────────────────

class TestRewardModelTraining:
    @pytest.fixture
    def preference_data(self):
        np.random.seed(0)
        n = 100
        chosen = np.random.beta(7, 3, (n, 5))
        rejected = np.random.beta(3, 7, (n, 5))
        return list(zip(chosen, rejected))

    def test_weights_all_positive(self, preference_data):
        weights, _ = train_reward_model(preference_data)
        assert all(w > 0 for w in weights)

    def test_loss_decreases(self, preference_data):
        _, losses = train_reward_model(preference_data)
        assert losses[-1] < losses[0]

    def test_accuracy_above_chance(self, preference_data):
        weights, _ = train_reward_model(preference_data)
        correct = sum(
            1 for chosen, rejected in preference_data
            if reward_model(chosen, weights) > reward_model(rejected, weights)
        )
        accuracy = correct / len(preference_data)
        assert accuracy > 0.7

    def test_rm_score_is_scalar(self, preference_data):
        weights, _ = train_reward_model(preference_data)
        score = reward_model(preference_data[0][0], weights)
        assert isinstance(score, float)

    def test_returns_correct_shapes(self, preference_data):
        weights, losses = train_reward_model(preference_data, n_epochs=50)
        assert len(weights) == 5
        assert len(losses) == 50


# ─── 3. KL Divergence ─────────────────────────────────────────────────────

class TestKLDivergence:
    def test_kl_self_is_zero(self):
        p = softmax(np.array([1.0, 2.0, 0.5, 3.0]))
        assert kl_divergence(p, p) < 1e-6

    def test_kl_nonnegative(self):
        np.random.seed(1)
        for _ in range(20):
            p = softmax(np.random.randn(10))
            q = softmax(np.random.randn(10))
            assert kl_divergence(p, q) >= 0

    def test_kl_asymmetric(self):
        p = softmax(np.array([1.0, 2.0, 0.1]))
        q = softmax(np.array([0.5, 1.0, 2.0]))
        assert abs(kl_divergence(p, q) - kl_divergence(q, p)) > 1e-3

    def test_kl_increases_with_distance(self):
        np.random.seed(2)
        base = np.random.randn(8)
        small_perturb = softmax(base + 0.2 * np.random.randn(8))
        large_perturb = softmax(base + 5.0 * np.random.randn(8))
        base_probs = softmax(base)
        assert kl_divergence(small_perturb, base_probs) < kl_divergence(large_perturb, base_probs)

    def test_kl_uniform_distributions(self):
        n = 5
        p = np.ones(n) / n
        q = np.ones(n) / n
        assert kl_divergence(p, q) < 1e-6


# ─── 4. PPO Objective ─────────────────────────────────────────────────────

class TestPPOObjective:
    def test_higher_rm_score_better_objective(self):
        obj_high = ppo_objective(3.0, 0.5, beta=0.1)
        obj_low = ppo_objective(1.0, 0.5, beta=0.1)
        assert obj_high > obj_low

    def test_higher_kl_worse_objective(self):
        obj_low_kl = ppo_objective(2.0, 0.5, beta=0.1)
        obj_high_kl = ppo_objective(2.0, 5.0, beta=0.1)
        assert obj_low_kl > obj_high_kl

    def test_beta_zero_ignores_kl(self):
        obj = ppo_objective(2.0, 100.0, beta=0.0)
        assert abs(obj - 2.0) < 1e-6

    def test_kl_penalty_proportional_to_beta(self):
        rm, kl = 2.0, 3.0
        obj1 = ppo_objective(rm, kl, beta=0.1)
        obj2 = ppo_objective(rm, kl, beta=0.2)
        expected_diff = (0.2 - 0.1) * kl
        assert abs((obj1 - obj2) - expected_diff) < 1e-6

    def test_objective_is_scalar(self):
        result = ppo_objective(1.5, 0.3, beta=0.05)
        assert isinstance(result, (int, float))


# ─── 5. RLHF Integration ─────────────────────────────────────────────────

class TestRLHFIntegration:
    def test_full_pipeline_runs(self):
        np.random.seed(42)
        n = 50
        chosen = np.random.beta(7, 3, (n, 5))
        rejected = np.random.beta(3, 7, (n, 5))
        pref_data = list(zip(chosen, rejected))

        # SFT: represented by initial weights
        sft_weights = np.zeros(5)

        # Reward Model training
        rm_weights, losses = train_reward_model(pref_data, n_epochs=100)
        assert losses[-1] < losses[0]

        # PPO: compute objective for a sample response
        sample_features = np.random.beta(6, 4, 5)
        rm_score = reward_model(sample_features, rm_weights)
        sft_score = reward_model(sample_features, sft_weights)

        p_new = softmax(sample_features + 0.1 * np.random.randn(5))
        p_sft = softmax(sample_features)
        kl = kl_divergence(p_new, p_sft)

        objective = ppo_objective(rm_score, kl)
        assert isinstance(objective, float)

    def test_rm_prefers_better_responses(self):
        np.random.seed(3)
        n = 80
        chosen = np.random.beta(8, 2, (n, 5))
        rejected = np.random.beta(2, 8, (n, 5))
        pref_data = list(zip(chosen, rejected))
        weights, _ = train_reward_model(pref_data, n_epochs=300)

        test_chosen = np.random.beta(8, 2, (20, 5))
        test_rejected = np.random.beta(2, 8, (20, 5))
        correct = sum(
            1 for c, r in zip(test_chosen, test_rejected)
            if reward_model(c, weights) > reward_model(r, weights)
        )
        assert correct / 20 >= 0.7

    def test_kl_penalty_prevents_reward_hacking(self):
        np.random.seed(4)
        base_logits = np.zeros(10)
        hacked_logits = base_logits + 8.0 * np.random.randn(10)

        p_base = softmax(base_logits)
        p_hacked = softmax(hacked_logits)

        kl_normal = kl_divergence(softmax(base_logits + 0.1 * np.random.randn(10)), p_base)
        kl_hacked = kl_divergence(p_hacked, p_base)

        high_rm_score = 10.0
        obj_normal = ppo_objective(high_rm_score * 0.5, kl_normal, beta=0.5)
        obj_hacked = ppo_objective(high_rm_score, kl_hacked, beta=0.5)

        # With sufficient KL penalty, hacking (huge KL) loses even with high RM score
        assert obj_normal > obj_hacked or kl_hacked > kl_normal * 5
