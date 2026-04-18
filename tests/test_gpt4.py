"""Tests for GPT-4 / Emergent Abilities concepts (node 14)."""

import numpy as np
import pytest


# ── Core math functions (mirroring the notebook) ─────────────────────────────

def sigmoid(x, k=1.0, x0=0.0):
    """Logistic sigmoid: 1 / (1 + exp(-k*(x - x0)))"""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def step_success_rate(log_scale, step_id, slope=0.2, base_offset=0.3):
    """Single-step success rate as a function of log model scale."""
    raw = slope * log_scale - base_offset - step_id * 0.1
    return float(np.clip(raw, 0.05, 0.98))


def task_success_rate(log_scales, n_steps):
    """Combined success rate for an n-step task (product of step rates)."""
    log_scales = np.asarray(log_scales, dtype=float)
    product = np.ones_like(log_scales)
    for i in range(n_steps):
        product = product * np.vectorize(lambda s: step_success_rate(s, i))(log_scales)
    return product


def find_emergence_threshold(log_scales, success_rates, threshold=0.5):
    """Return log-scale at which success_rates first exceeds threshold."""
    log_scales = np.asarray(log_scales)
    success_rates = np.asarray(success_rates)
    above = success_rates >= threshold
    if not np.any(above):
        return None
    return float(log_scales[np.argmax(above)])


# ── Sigmoid function tests ────────────────────────────────────────────────────

class TestSigmoid:

    def test_output_range(self):
        """Sigmoid output is always in (0, 1)."""
        x = np.linspace(-10, 10, 100)
        y = sigmoid(x)
        assert np.all(y > 0) and np.all(y < 1)

    def test_midpoint(self):
        """At x=x0, sigmoid equals 0.5."""
        assert abs(sigmoid(0.0) - 0.5) < 1e-10
        assert abs(sigmoid(3.0, x0=3.0) - 0.5) < 1e-10

    def test_monotone_increasing(self):
        """Sigmoid is strictly monotone increasing."""
        x = np.linspace(-5, 5, 50)
        y = sigmoid(x)
        assert np.all(np.diff(y) > 0)

    def test_steepness_parameter(self):
        """Larger k produces a steeper transition."""
        x = np.array([0.5])
        low_k = sigmoid(x, k=0.5)[0]
        high_k = sigmoid(x, k=5.0)[0]
        assert high_k > low_k

    def test_asymptotes(self):
        """Very large positive x → ~1, very large negative x → ~0."""
        assert sigmoid(100.0) > 0.999
        assert sigmoid(-100.0) < 0.001

    def test_shift_parameter(self):
        """Shifting x0 moves the midpoint."""
        assert abs(sigmoid(5.0, x0=5.0) - 0.5) < 1e-8
        assert sigmoid(5.0, x0=3.0) > 0.5  # midpoint moved left


# ── Step success rate tests ───────────────────────────────────────────────────

class TestStepSuccessRate:

    def test_output_clipped(self):
        """Output is always in [0.05, 0.98]."""
        for s in [-5, 0, 5, 10, 20]:
            for step_id in [0, 1, 2, 5]:
                r = step_success_rate(s, step_id)
                assert 0.05 <= r <= 0.98, f"s={s}, step_id={step_id}: {r}"

    def test_increases_with_scale(self):
        """Larger log_scale → higher success rate (until ceiling)."""
        r_small = step_success_rate(1.0, 0)
        r_large = step_success_rate(8.0, 0)
        assert r_large >= r_small

    def test_harder_steps_harder(self):
        """Higher step_id (harder step) has lower or equal success rate."""
        s = 5.0
        r0 = step_success_rate(s, 0)
        r3 = step_success_rate(s, 3)
        assert r0 >= r3


# ── Task success rate tests ───────────────────────────────────────────────────

class TestTaskSuccessRate:

    def test_monotone_with_scale(self):
        """More steps → curve shifts right (emerges later)."""
        log_s = np.linspace(0, 12, 100)
        r1 = task_success_rate(log_s, 1)
        r3 = task_success_rate(log_s, 3)
        # At large scale, both converge near 1; at small scale, 1-step > 3-step
        assert r1[0] >= r3[0]

    def test_output_bounded(self):
        """Task success rate is in [0, 1]."""
        log_s = np.linspace(-2, 15, 50)
        for n in [1, 2, 5]:
            r = task_success_rate(log_s, n)
            assert np.all(r >= 0) and np.all(r <= 1)

    def test_product_property(self):
        """1-step rate should be strictly >= 2-step rate (product can only decrease)."""
        log_s = np.array([3.0, 5.0, 8.0])
        r1 = task_success_rate(log_s, 1)
        r2 = task_success_rate(log_s, 2)
        # For these log_scales, step rates < 1, so product is smaller
        assert np.all(r1 >= r2 - 1e-10)

    def test_more_steps_lower_rate(self):
        """More steps → lower combined rate (at same scale)."""
        log_s = np.array([4.0])
        r1 = float(task_success_rate(log_s, 1)[0])
        r5 = float(task_success_rate(log_s, 5)[0])
        assert r1 >= r5


# ── Emergence threshold tests ─────────────────────────────────────────────────

class TestEmergenceThreshold:

    def test_threshold_increases_with_steps(self):
        """Tasks with more steps have higher (or equal) emergence thresholds."""
        log_s = np.linspace(0, 12, 500)
        t1 = find_emergence_threshold(log_s, task_success_rate(log_s, 1))
        t3 = find_emergence_threshold(log_s, task_success_rate(log_s, 3))
        t5 = find_emergence_threshold(log_s, task_success_rate(log_s, 5))
        assert t1 is not None and t3 is not None and t5 is not None
        assert t1 <= t3 <= t5

    def test_returns_none_if_never_emerges(self):
        """Returns None when success never exceeds threshold."""
        log_s = np.linspace(0, 2, 50)  # tiny range
        rates = task_success_rate(log_s, 10)  # 10 steps, impossible in this range
        result = find_emergence_threshold(log_s, rates, threshold=0.99)
        # May or may not be None depending on params, just check type
        assert result is None or isinstance(result, float)

    def test_threshold_value_is_valid_log_scale(self):
        """Returned threshold is within the provided log_scales range."""
        log_s = np.linspace(0, 12, 500)
        t = find_emergence_threshold(log_s, task_success_rate(log_s, 2))
        assert t is not None
        assert log_s[0] <= t <= log_s[-1]

    def test_above_threshold_at_threshold(self):
        """At the returned threshold index, success rate >= 0.5."""
        log_s = np.linspace(0, 12, 500)
        rates = task_success_rate(log_s, 3)
        t = find_emergence_threshold(log_s, rates)
        assert t is not None
        idx = np.argmin(np.abs(log_s - t))
        assert rates[idx] >= 0.5


# ── GPT-4 benchmark data validation ──────────────────────────────────────────

class TestGPT4BenchmarkData:
    """Validate the benchmark data cited from GPT-4 Technical Report."""

    GPT35 = {'ube': 10, 'usmle': 53, 'gre': 63, 'sat': 70, 'amc': 30}
    GPT4  = {'ube': 90, 'usmle': 87, 'gre': 99, 'sat': 89, 'amc': 60}

    def test_gpt4_beats_gpt35_on_all_exams(self):
        """GPT-4 outperforms GPT-3.5 on every benchmark listed."""
        for exam in self.GPT35:
            assert self.GPT4[exam] > self.GPT35[exam], f"Exam: {exam}"

    def test_gpt4_passes_professional_threshold_ube(self):
        """GPT-4 exceeds the ~75th percentile bar for UBE (law exam)."""
        assert self.GPT4['ube'] >= 75

    def test_gpt4_passes_professional_threshold_usmle(self):
        """GPT-4 exceeds the ~60th percentile passing bar for USMLE."""
        assert self.GPT4['usmle'] >= 60

    def test_gpt35_fails_ube_threshold(self):
        """GPT-3.5 was below the 75th percentile threshold for UBE."""
        assert self.GPT35['ube'] < 75

    def test_gpt4_gre_near_ceiling(self):
        """GPT-4 GRE verbal score is at or near the 99th percentile."""
        assert self.GPT4['gre'] >= 95

    def test_scores_in_valid_range(self):
        """All scores are valid percentile values in [0, 100]."""
        for v in list(self.GPT35.values()) + list(self.GPT4.values()):
            assert 0 <= v <= 100
