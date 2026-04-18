"""
Tests for node 04 LeNet 1989 — conv2d, max_pool2d, and end-to-end forward dimensions.

Inline implementation mirrors notebooks/04-lenet-1989.ipynb so tests stay
independent of notebook execution environment.
"""
import numpy as np
import pytest


# ── Core functions (mirrored from notebooks/04-lenet-1989.ipynb) ─────────────

def conv2d(image, kernel, stride=1, padding=0):
    H, W = image.shape
    kH, kW = kernel.shape
    if padding > 0:
        image = np.pad(image, padding, mode='constant')
        H, W = image.shape
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            patch = image[i*stride:i*stride+kH, j*stride:j*stride+kW]
            output[i, j] = np.sum(patch * kernel)
    return output


def max_pool2d(feature_map, pool_size=2, stride=2):
    H, W = feature_map.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            patch = feature_map[i*stride:i*stride+pool_size,
                                j*stride:j*stride+pool_size]
            output[i, j] = np.max(patch)
    return output


# ── Tests: conv2d output size ─────────────────────────────────────────────────

def test_conv2d_output_size_basic():
    """(H-kH)//stride+1 x (W-kW)//stride+1 —— 无 padding, stride=1"""
    img = np.zeros((8, 8))
    k = np.zeros((3, 3))
    out = conv2d(img, k)
    assert out.shape == (6, 6), f"期望 (6,6), 实际 {out.shape}"


def test_conv2d_output_size_stride2():
    """stride=2 时输出尺寸减半"""
    img = np.zeros((8, 8))
    k = np.zeros((3, 3))
    out = conv2d(img, k, stride=2)
    assert out.shape == (3, 3), f"期望 (3,3), 实际 {out.shape}"


def test_conv2d_output_size_with_padding():
    """padding=1 时输出与输入同尺寸（same padding，stride=1, kernel=3）"""
    img = np.zeros((6, 6))
    k = np.zeros((3, 3))
    out = conv2d(img, k, stride=1, padding=1)
    assert out.shape == (6, 6), f"期望 (6,6), 实际 {out.shape}"


def test_conv2d_output_size_non_square():
    """非正方形输入/核"""
    img = np.zeros((10, 6))
    k = np.zeros((3, 2))
    out = conv2d(img, k)
    assert out.shape == (8, 5), f"期望 (8,5), 实际 {out.shape}"


# ── Tests: conv2d numerical correctness ──────────────────────────────────────

def test_conv2d_identity_kernel():
    """2x2 核 [[1,0],[0,0]] 对 3x3 图片，输出应等于图片左上角各块的第一个元素"""
    img = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]], dtype=float)
    k = np.array([[1, 0], [0, 0]], dtype=float)
    out = conv2d(img, k)
    expected = np.array([[1, 2], [4, 5]], dtype=float)
    np.testing.assert_array_equal(out, expected)


def test_conv2d_known_values():
    """notebook 中的验证用例：[[1,0],[0,-1]] 核"""
    img = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]], dtype=float)
    k = np.array([[1, 0], [0, -1]], dtype=float)
    out = conv2d(img, k)
    expected = np.array([[1-5, 2-6], [4-8, 5-9]], dtype=float)
    np.testing.assert_array_equal(out, expected)


def test_conv2d_horizontal_edge_detector():
    """横线检测核在含横线的图片上输出较大正数"""
    img = np.zeros((7, 7))
    img[3, :] = 1.0   # 中间一条横线
    k_h = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]], dtype=float)
    out = conv2d(img, k_h)
    # 横线所在行的输出应显著大于其他行
    assert out.max() > 0, "横线检测核应在横线处产生正输出"


def test_conv2d_all_zeros_kernel():
    """全零核输出应为全零"""
    img = np.arange(16, dtype=float).reshape(4, 4)
    k = np.zeros((2, 2))
    out = conv2d(img, k)
    np.testing.assert_array_equal(out, np.zeros((3, 3)))


# ── Tests: max_pool2d output size ─────────────────────────────────────────────

def test_max_pool2d_output_size_default():
    """默认 pool_size=2, stride=2 时，4x4→2x2"""
    fm = np.zeros((4, 4))
    out = max_pool2d(fm)
    assert out.shape == (2, 2), f"期望 (2,2), 实际 {out.shape}"


def test_max_pool2d_output_size_6x6():
    """6x6 输入，pool=2, stride=2 → 3x3"""
    fm = np.zeros((6, 6))
    out = max_pool2d(fm, pool_size=2, stride=2)
    assert out.shape == (3, 3), f"期望 (3,3), 实际 {out.shape}"


def test_max_pool2d_output_size_stride1():
    """stride=1 时 4x4 → 3x3"""
    fm = np.zeros((4, 4))
    out = max_pool2d(fm, pool_size=2, stride=1)
    assert out.shape == (3, 3), f"期望 (3,3), 实际 {out.shape}"


# ── Tests: max_pool2d numerical correctness ───────────────────────────────────

def test_max_pool2d_known_values():
    """notebook 中的验证用例"""
    fm = np.array([[1, 3, 2, 4],
                   [5, 6, 1, 2],
                   [3, 2, 4, 7],
                   [1, 0, 5, 3]], dtype=float)
    out = max_pool2d(fm)
    expected = np.array([[6, 4], [3, 7]], dtype=float)
    np.testing.assert_array_equal(out, expected)


def test_max_pool2d_selects_maximum():
    """池化确实取每个窗口的最大值"""
    fm = np.array([[9, 1], [2, 3]], dtype=float)
    out = max_pool2d(fm, pool_size=2, stride=2)
    assert out[0, 0] == 9.0


# ── Tests: end-to-end CNN forward dimensions ──────────────────────────────────

def test_lenet_forward_dimensions():
    """端到端前向维度：16×16 → conv(3×3) → pool(2×2) → flatten"""
    # 输入图片
    img = np.zeros((16, 16))

    # 两个卷积核
    k1 = np.ones((3, 3)) / 9.0
    k2 = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]], dtype=float)

    fmap1 = conv2d(img, k1)   # 期望 14×14
    fmap2 = conv2d(img, k2)   # 期望 14×14
    assert fmap1.shape == (14, 14), f"卷积输出应为 14×14, 实际 {fmap1.shape}"
    assert fmap2.shape == (14, 14)

    pool1 = max_pool2d(fmap1)  # 期望 7×7
    pool2 = max_pool2d(fmap2)  # 期望 7×7
    assert pool1.shape == (7, 7), f"池化输出应为 7×7, 实际 {pool1.shape}"
    assert pool2.shape == (7, 7)

    flat = np.concatenate([pool1.flatten(), pool2.flatten()])
    assert flat.shape[0] == 7 * 7 * 2, f"拼接后应为 98, 实际 {flat.shape[0]}"


def test_param_count_conv_vs_fc():
    """权重共享：卷积参数量远少于全连接层"""
    H, W = 32, 32
    n_filters = 8
    kH, kW = 5, 5
    n_fc_neurons = 100

    fc_params = H * W * n_fc_neurons       # 102400
    conv_params = kH * kW * n_filters      # 200

    assert conv_params < fc_params, "卷积参数量应少于全连接"
    assert conv_params == 200
    assert fc_params == 102400
    assert conv_params / fc_params < 0.002, "卷积参数量应不足全连接的 0.2%"
