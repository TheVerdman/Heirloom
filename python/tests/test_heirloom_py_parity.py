import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import heirloom_py as h


def as_numpy(tensor):
    return np.array(tensor.tolist(), dtype=np.float64).reshape(tensor.shape)


def assert_tensor_close(heirloom_tensor, torch_tensor, atol=1e-5, rtol=1e-5):
    np.testing.assert_allclose(
        as_numpy(heirloom_tensor),
        torch_tensor.detach().cpu().numpy(),
        atol=atol,
        rtol=rtol,
    )


def assert_grad_close(heirloom_tensor, torch_tensor, atol=1e-5, rtol=1e-5):
    grad = heirloom_tensor.grad()
    assert grad is not None
    np.testing.assert_allclose(
        as_numpy(grad),
        torch_tensor.grad.detach().cpu().numpy(),
        atol=atol,
        rtol=rtol,
    )


def tensor_from_torch(torch_tensor, requires_grad=False):
    return h.tensor(
        torch_tensor.detach().cpu().reshape(-1).tolist(),
        list(torch_tensor.shape),
        dtype="f32",
        requires_grad=requires_grad,
    )


def test_numpy_constructor_and_materialization_round_trip():
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    x = h.from_numpy(array, requires_grad=True)

    assert x.shape == [2, 3]
    assert x.dtype == "f32"
    assert x.device == "cpu"
    np.testing.assert_allclose(x.numpy(), array)


def test_matmul_relu_mean_backward_matches_torch():
    x_t = torch.tensor(
        [[-1.0, 0.5, 2.0], [3.0, -0.25, 1.5]], dtype=torch.float32, requires_grad=True
    )
    w_t = torch.tensor(
        [[0.25, -0.5], [1.0, 2.0], [-1.5, 0.75]], dtype=torch.float32, requires_grad=True
    )
    x_h = tensor_from_torch(x_t, requires_grad=True)
    w_h = tensor_from_torch(w_t, requires_grad=True)

    loss_h = x_h.matmul(w_h).relu().mean()
    loss_t = torch.relu(x_t @ w_t).mean()

    assert_tensor_close(loss_h, loss_t)
    loss_h.backward()
    loss_t.backward()
    assert_grad_close(x_h, x_t)
    assert_grad_close(w_h, w_t)


def test_broadcast_transpose_sum_backward_matches_torch():
    x_t = torch.tensor(
        [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], dtype=torch.float32, requires_grad=True
    )
    b_t = torch.tensor([0.5, -1.0, 1.5], dtype=torch.float32, requires_grad=True)
    x_h = tensor_from_torch(x_t, requires_grad=True)
    b_h = tensor_from_torch(b_t, requires_grad=True)

    loss_h = (x_h + b_h).transpose().sum()
    loss_t = (x_t + b_t).t().sum()

    assert_tensor_close(loss_h, loss_t)
    loss_h.backward()
    loss_t.backward()
    assert_grad_close(x_h, x_t)
    assert_grad_close(b_h, b_t)


def test_layer_norm_forward_backward_matches_torch():
    x_t = torch.tensor(
        [
            [[-0.5, 1.0, 2.0, -1.5], [0.25, -0.75, 1.25, 2.5]],
            [[1.5, -2.0, 0.0, 0.5], [2.25, -1.25, 0.75, -0.25]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    weight_t = torch.tensor([1.0, 0.75, -0.5, 1.25], dtype=torch.float32, requires_grad=True)
    bias_t = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float32, requires_grad=True)
    x_h = tensor_from_torch(x_t, requires_grad=True)
    weight_h = tensor_from_torch(weight_t, requires_grad=True)
    bias_h = tensor_from_torch(bias_t, requires_grad=True)

    y_h = x_h.layer_norm_last_dim(weight_h, bias_h, 1e-5)
    y_t = F.layer_norm(x_t, (4,), weight_t, bias_t, eps=1e-5)
    assert_tensor_close(y_h, y_t, atol=2e-5, rtol=2e-5)

    y_h.sum().backward()
    y_t.sum().backward()
    assert_grad_close(x_h, x_t, atol=2e-5, rtol=2e-5)
    assert_grad_close(weight_h, weight_t, atol=2e-5, rtol=2e-5)
    assert_grad_close(bias_h, bias_t, atol=2e-5, rtol=2e-5)


def test_embedding_backward_accumulates_repeated_rows_like_torch():
    indices_h = h.tensor([0, 2, 0, 3], [4], dtype="i64")
    weight_t = torch.tensor(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
        dtype=torch.float32,
        requires_grad=True,
    )
    weight_h = tensor_from_torch(weight_t, requires_grad=True)
    indices_t = torch.tensor([0, 2, 0, 3], dtype=torch.long)

    loss_h = indices_h.embedding(weight_h).sum()
    loss_t = F.embedding(indices_t, weight_t).sum()

    assert_tensor_close(loss_h, loss_t)
    loss_h.backward()
    loss_t.backward()
    assert_grad_close(weight_h, weight_t)


def test_cross_entropy_forward_backward_matches_torch():
    logits_t = torch.tensor(
        [[1.0, -0.5, 0.25], [-1.25, 0.75, 2.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets_t = torch.tensor([0, 2], dtype=torch.long)
    logits_h = tensor_from_torch(logits_t, requires_grad=True)
    targets_h = h.tensor(targets_t.tolist(), [2], dtype="i64")

    loss_h = logits_h.cross_entropy_for_logits(targets_h)
    loss_t = F.cross_entropy(logits_t, targets_t)

    assert_tensor_close(loss_h, loss_t, atol=2e-5, rtol=2e-5)
    loss_h.backward()
    loss_t.backward()
    assert_grad_close(logits_h, logits_t, atol=2e-5, rtol=2e-5)


def torch_causal_self_attention(query, key, value, n_heads):
    batch, time, channels = query.shape
    head_dim = channels // n_heads
    q = query.reshape(batch, time, n_heads, head_dim).permute(0, 2, 1, 3)
    k = key.reshape(batch, time, n_heads, head_dim).permute(0, 2, 1, 3)
    v = value.reshape(batch, time, n_heads, head_dim).permute(0, 2, 1, 3)
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
    mask = torch.triu(torch.ones(time, time, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v).permute(0, 2, 1, 3).reshape(batch, time, channels)


def test_causal_self_attention_forward_backward_matches_torch():
    q_t = torch.tensor(
        [
            [
                [0.1, -0.2, 0.3, 0.4],
                [0.5, 0.6, -0.7, 0.8],
                [-0.9, 1.0, 1.1, -1.2],
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    k_t = torch.tensor(
        [
            [
                [0.2, 0.1, -0.4, 0.3],
                [-0.5, 0.7, 0.6, -0.8],
                [0.9, -1.0, 1.2, 1.1],
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    v_t = torch.tensor(
        [
            [
                [0.3, -0.1, 0.2, 0.5],
                [0.6, -0.4, 0.7, -0.8],
                [-0.9, 1.0, -1.1, 1.2],
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    q_h = tensor_from_torch(q_t, requires_grad=True)
    k_h = tensor_from_torch(k_t, requires_grad=True)
    v_h = tensor_from_torch(v_t, requires_grad=True)

    y_h = q_h.causal_self_attention(k_h, v_h, 2)
    y_t = torch_causal_self_attention(q_t, k_t, v_t, 2)
    assert_tensor_close(y_h, y_t, atol=3e-5, rtol=3e-5)

    y_h.sum().backward()
    y_t.sum().backward()
    assert_grad_close(q_h, q_t, atol=5e-5, rtol=5e-5)
    assert_grad_close(k_h, k_t, atol=5e-5, rtol=5e-5)
    assert_grad_close(v_h, v_t, atol=5e-5, rtol=5e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity requires a local CUDA GPU")
def test_cuda_device_round_trip_when_available():
    x = h.tensor([1.0, -2.0, 3.0], [3], requires_grad=True, device="cuda:0")
    y = x.relu().sum()
    y.backward()

    assert x.device == "cuda:0"
    assert y.cpu().tolist() == [4.0]
    assert x.grad().cpu().tolist() == [1.0, 0.0, 1.0]
