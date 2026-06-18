#!/usr/bin/env python3
"""Generate tests/fixtures/pytorch_parity.json from live PyTorch.

This script intentionally fails with a clear message if torch is not installed.
It keeps cargo tests independent from Python, while making fixture refreshes
traceable to real PyTorch when the dependency is available.
"""

from __future__ import annotations

import json
from pathlib import Path


def tensor_case(name, shape, data, requires_grad=True, dtype="f32"):
    return {
        "name": name,
        "shape": shape,
        "data": [float(v) if dtype == "f32" else int(v) for v in data],
        "requires_grad": requires_grad,
        "dtype": dtype,
    }


def run_case(case, torch):
    tensors = {}
    for item in case["inputs"]:
        dtype = item.get("dtype", "f32")
        torch_dtype = torch.float32 if dtype == "f32" else torch.long
        tensor = torch.tensor(item["data"], dtype=torch_dtype).reshape(item["shape"])
        if item["requires_grad"]:
            tensor.requires_grad_()
        tensors[item["name"]] = tensor
    for item in case["inputs"]:
        if item["requires_grad"]:
            tensors[item["name"]].retain_grad()

    op = case["op"]
    if op == "add_sum":
        output = (tensors["x"] + tensors["bias"]).sum()
    elif op == "matmul_mean":
        output = tensors["left"].matmul(tensors["right"]).mean()
    elif op == "batched_matmul_mean":
        output = tensors["left"].matmul(tensors["right"]).mean()
    elif op == "relu_mean":
        output = tensors["x"].relu().mean()
    elif op == "gelu_tanh_mean":
        output = torch.nn.functional.gelu(tensors["x"], approximate="tanh").mean()
    elif op == "layer_norm_square_sum":
        output_tensor = torch.nn.functional.layer_norm(
            tensors["x"],
            [case["features"]],
            tensors["weight"],
            tensors["bias"],
            eps=case["eps"],
        )
        output = (output_tensor * output_tensor).sum()
    elif op == "embedding_sum":
        output = torch.nn.functional.embedding(tensors["indices"], tensors["weight"]).sum()
    elif op == "softmax_square_sum":
        probs = tensors["logits"].softmax(dim=1)
        output = (probs * probs).sum()
    elif op == "cross_entropy":
        targets = torch.tensor(case["targets"], dtype=torch.long)
        output = torch.nn.functional.cross_entropy(tensors["logits"], targets)
    elif op == "causal_attention_square_sum":
        attended = causal_self_attention(
            tensors["query"],
            tensors["key"],
            tensors["value"],
            case["n_heads"],
            torch,
        )
        output = (attended * attended).sum()
    else:
        raise ValueError(f"unknown op: {op}")

    output.backward()
    case["output_shape"] = list(output.shape)
    case["output_data"] = [float(v) for v in output.detach().reshape(-1).tolist()]
    case["grads"] = {
        name: [float(v) for v in tensor.grad.detach().reshape(-1).tolist()]
        for name, tensor in tensors.items()
        if tensor.requires_grad
    }
    return case


def causal_self_attention(query, key, value, n_heads, torch):
    batch, time, channels = query.shape
    head_dim = channels // n_heads
    q = query.reshape(batch, time, n_heads, head_dim).transpose(1, 2)
    k = key.reshape(batch, time, n_heads, head_dim).transpose(1, 2)
    v = value.reshape(batch, time, n_heads, head_dim).transpose(1, 2)
    scores = q.matmul(k.transpose(-2, -1)) / (head_dim**0.5)
    mask = torch.triu(torch.ones(time, time, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    attention = scores.softmax(dim=-1)
    output = attention.matmul(v).transpose(1, 2).reshape(batch, time, channels)
    return output


def main():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install torch to refresh live parity fixtures."
        ) from exc

    cases = [
        {
            "name": "add_broadcast_sum",
            "op": "add_sum",
            "inputs": [
                tensor_case("x", [2, 3], [0, 1, 2, 3, 4, 5]),
                tensor_case("bias", [3], [10, 20, 30]),
            ],
        },
        {
            "name": "matmul_mean",
            "op": "matmul_mean",
            "inputs": [
                tensor_case("left", [2, 3], [1, -2, 3, 4, 0.5, -1]),
                tensor_case("right", [3, 2], [0.5, -1, 2, 1.5, -0.25, 0.75]),
            ],
        },
        {
            "name": "batched_matmul_mean",
            "op": "batched_matmul_mean",
            "inputs": [
                tensor_case(
                    "left",
                    [2, 2, 3],
                    [1, 2, 3, 4, 5, 6, -1, -2, -3, 2, 1, 0],
                ),
                tensor_case("right", [3, 2], [1, 0.5, -1, 2, 0.25, -0.5]),
            ],
        },
        {
            "name": "relu_mean",
            "op": "relu_mean",
            "inputs": [tensor_case("x", [5], [-2, -0.0, 0.0, 3, 5])],
        },
        {
            "name": "gelu_tanh_mean",
            "op": "gelu_tanh_mean",
            "inputs": [tensor_case("x", [5], [-2, -0.5, 0.0, 1.5, 3.0])],
        },
        {
            "name": "layer_norm_square_sum",
            "op": "layer_norm_square_sum",
            "features": 4,
            "eps": 1e-5,
            "inputs": [
                tensor_case("x", [2, 4], [0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8]),
                tensor_case("weight", [4], [1.0, 0.5, -1.0, 2.0]),
                tensor_case("bias", [4], [0.0, 0.1, -0.2, 0.3]),
            ],
        },
        {
            "name": "embedding_sum",
            "op": "embedding_sum",
            "inputs": [
                tensor_case("indices", [4], [0, 1, 0, 2], requires_grad=False, dtype="i64"),
                tensor_case("weight", [3, 2], [1.0, 2.0, 3.0, 4.0, -1.0, -2.0]),
            ],
        },
        {
            "name": "softmax_square_sum",
            "op": "softmax_square_sum",
            "inputs": [tensor_case("logits", [2, 3], [1, -0.5, 0.25, 0.7, 0.2, -1])],
        },
        {
            "name": "cross_entropy",
            "op": "cross_entropy",
            "targets": [2, 0],
            "inputs": [tensor_case("logits", [2, 3], [0.2, -1, 1.7, 1.2, 0.4, -0.3])],
        },
        {
            "name": "causal_attention_square_sum",
            "op": "causal_attention_square_sum",
            "n_heads": 2,
            "inputs": [
                tensor_case(
                    "query",
                    [1, 3, 4],
                    [0.1, 0.2, -0.3, 0.4, 0.2, -0.1, 0.5, 0.3, -0.4, 0.6, 0.7, -0.2],
                ),
                tensor_case(
                    "key",
                    [1, 3, 4],
                    [-0.2, 0.3, 0.4, 0.1, 0.6, -0.5, 0.2, 0.8, 0.1, 0.0, -0.3, 0.5],
                ),
                tensor_case(
                    "value",
                    [1, 3, 4],
                    [0.7, -0.1, 0.2, 0.3, -0.4, 0.5, 0.6, -0.2, 0.1, 0.8, -0.6, 0.4],
                ),
            ],
        },
    ]

    rendered = {
        "generated_by": f"torch {torch.__version__}",
        "cases": [run_case(case, torch) for case in cases],
    }
    output_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "pytorch_parity.json"
    output_path.write_text(json.dumps(rendered, indent=2) + "\n")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
